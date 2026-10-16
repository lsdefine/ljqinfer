"""Request-owned RNG and reusable target selection. No cache/draft changes."""
import secrets
import math
import torch
import torch.distributed as dist
from .glm53_sampling import NucleusSampler


class SamplingRequest:
    def __init__(self, max_tokens, device, *, temperature=0., top_p=.95, seed=None):
        if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<0:
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = float(temperature)
        if top_p != .95:
            raise ValueError('sampling requires top_p=0.95')
        self.uniforms = None
        self.seed = seed
        if temperature == 0.:
            return
        if seed is None:
            value = secrets.randbits(63) if not dist.is_initialized() or dist.get_rank()==0 else 0
            value = torch.tensor([value],device=device,dtype=torch.int64)
            if dist.is_initialized():dist.broadcast(value,src=0)
            seed = int(value.item())
        if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        self.seed = seed
        # Generate once, before the timed decode loop; identical on every TP rank.
        rng = torch.Generator(device='cpu').manual_seed(seed)
        self.uniforms = torch.rand(max_tokens+8,generator=rng,dtype=torch.float64).to(device)

    def at(self, position, count):
        if self.uniforms is None:return None
        if position<0 or position+count>self.uniforms.numel():
            raise ValueError('sampling position out of request range')
        return self.uniforms[position:position+count]


class TargetSelector:
    def __init__(self, device):
        self.device=device
        self.plans={}

    def _prepare(self, rows, width):
        key=(rows,width)
        if key not in self.plans:
            sampler=NucleusSampler(rows,width,self.device)
            logits=torch.zeros((rows,width),device=self.device,dtype=torch.float32)
            uniforms=torch.full((rows,),.5,device=self.device,dtype=torch.float64)
            for _ in range(2):sampler.run(logits,uniforms)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=sampler.run(logits,uniforms)
            self.plans[key]=(sampler,logits,uniforms,graph,out)
        return self.plans[key]

    def select(self, logits, requests, positions):
        """[B,Q,V], request/position ordered by logical active request, not slot."""
        b,q,v=logits.shape
        if len(requests)!=b or len(positions)!=b:raise ValueError('sampling row mismatch')
        if all(r.uniforms is None for r in requests):return logits.argmax(-1)
        sampler,inputs,uniforms,graph,out=self._prepare(b*q,v)
        inputs.copy_(logits.reshape(b*q,v))
        for i,(request,position) in enumerate(zip(requests,positions)):
            u=request.at(position,q)
            sampler.temperatures[i*q:(i+1)*q].fill_(request.temperature if request.temperature>0 else 1.)
            if u is None:uniforms[i*q:(i+1)*q].zero_()
            else:
                uniforms[i*q:(i+1)*q].copy_(u)
                # GLM single-turn chat constraint, not raw unconstrained top-p:
                # a new <|assistant|> header is not content of this reply.
                # Mask only the private sampling buffer, before normalization.
                # Original model logits, EOS and greedy rows remain untouched.
                if v > 154828:
                    inputs[i*q:(i+1)*q,154828] = -float('inf')
        graph.replay()
        # Greedy rows in a mixed batch must retain exact argmax tie policy.
        result=out.view(b,q)
        for i,request in enumerate(requests):
            if request.uniforms is None:result[i].copy_(logits[i].argmax(-1))
        return result

    def close(self):
        for _,_,_,graph,_ in self.plans.values():graph.reset()
        self.plans.clear()
