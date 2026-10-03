"""GLM53 FP8 checkpoint -> TP8 single-layer weights for the existing MLA ABI.
Source FP8 attention is decoded to FP16 without additional quantization.
"""
from pathlib import Path
import json
import struct
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from model.int4_quantize import quantize_mse


class Source:
    def __init__(self, root, layer):
        self.root = Path(root)
        self.prefix = f'model.layers.{layer}.'
        self.by = {}
        for path in sorted(self.root.glob('*.safetensors')):
            with path.open('rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(n))
            required = max((v['data_offsets'][1] for k,v in header.items() if k != '__metadata__'), default=0)
            if path.stat().st_size != 8+n+required:
                continue
            for key in header:
                if key.startswith(self.prefix):
                    if key in self.by:
                        raise ValueError(f'duplicate tensor {key}')
                    self.by[key] = path

    def get(self, name, rows=None, cols=None):
        key = self.prefix+name
        with safe_open(self.by[key], framework='pt', device='cpu') as f:
            if rows is None and cols is None:
                return f.get_tensor(key)
            return f.get_slice(key)[rows if rows is not None else slice(None),
                                    cols if cols is not None else slice(None)].contiguous()

    def matrix(self, name, device, rows=None, cols=None, raw=False):
        w = self.get(name+'.weight', rows, cols).to(device)
        def blocks(s):
            if s is None: return None
            assert s.start % 128 == 0 and s.stop % 128 == 0
            return slice(s.start//128, s.stop//128)
        scale = self.get(name+'.weight_scale_inv', blocks(rows), blocks(cols)).to(device)
        if raw: return w.view(torch.uint8).contiguous(), scale.contiguous()
        return w.float()*scale.repeat_interleave(128,0).repeat_interleave(128,1)[:w.shape[0],:w.shape[1]]


def convert_rank(root, output, layer, rank, group=64, *, include_fp8_down=True, verify=False):
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    source = Source(root, layer)
    result = {}
    def put(name, tensor): result[name] = tensor.detach().cpu().contiguous()
    put('ffn_norm',source.get('post_attention_layernorm.weight').half())
    put('router',source.get('mlp.gate.weight').float())
    put('bias',source.get('mlp.gate.e_score_correction_bias').float())
    sl = slice(rank*256,(rank+1)*256)
    g=source.matrix('mlp.shared_experts.gate_proj',device,rows=sl)
    u=source.matrix('mlp.shared_experts.up_proj',device,rows=sl)
    put('shared_gu',torch.cat((g,u)).bfloat16())
    del g,u
    put('shared_down',source.matrix('mlp.shared_experts.down_proj',device,cols=sl).bfloat16())
    names = ['gu','gu_scale','down_int4','down_int4_scale']
    if include_fp8_down: names += ['down_fp8','down_fp8_scale']
    fields={k:[] for k in names}
    for e in range(256):
        base=f'mlp.experts.{e}.'
        gu=torch.cat([source.matrix(base+n,device,rows=sl) for n in ['gate_proj','up_proj']])
        a,b=quantize_mse(gu,group,chunk_rows=256)
        fields['gu'].append(a.cpu());fields['gu_scale'].append(b.cpu())
        del gu,a,b
        raw,scale=source.matrix(base+'down_proj',device,cols=sl,raw=True)
        if include_fp8_down:
            fields['down_fp8'].append(raw.cpu());fields['down_fp8_scale'].append(scale.cpu())
        w=raw.view(torch.float8_e4m3fn).float()*scale.repeat_interleave(128,0).repeat_interleave(128,1)
        a,b=quantize_mse(w,group,chunk_rows=1024)
        fields['down_int4'].append(a.cpu());fields['down_int4_scale'].append(b.cpu())
        del raw,scale,w,a,b
        if e%32==0: print('CONVERT',rank,e,flush=True)
    result.update({k:torch.stack(v) for k,v in fields.items()})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    target=out/f'layer{layer}_g{group}_rank{rank}.safetensors'
    tmp=target.with_suffix('.tmp')
    save_file(result,str(tmp),metadata={'layer':str(layer),'rank':str(rank),'group':str(group),
                                       'attention':'loaded from original checkpoint','source':str(root),
                                       'format':'int4_sym_mse_v1','include_fp8_down':str(include_fp8_down)})
    if verify:
        with safe_open(str(tmp), framework='pt', device='cpu') as check:
            assert set(check.keys()) == set(result)
            for key, expected in result.items():
                actual = check.get_tensor(key)
                assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
                assert torch.equal(actual, expected), key
                if actual.is_floating_point(): assert torch.isfinite(actual).all(), key
        with tmp.open('rb') as handle:
            import os
            os.fsync(handle.fileno())
    tmp.replace(target)
    print('SAVED',target,target.stat().st_size,flush=True)


def load_attention(root, layer, rank, device):
    """Decode source FP8 once on load, without additional quantization."""
    source=Source(root,layer)
    a={}
    for target,name in [('norm','input_layernorm'),('q_a_norm','self_attn.q_a_layernorm'),('kv_a_norm','self_attn.kv_a_layernorm')]:
        a[target]=source.get(name+'.weight').to(device=device,dtype=torch.float16)
    for target,name in [('q_a','q_a_proj'),('kv_a','kv_a_proj_with_mqa')]:
        a[target]=source.matrix('self_attn.'+name,device).half()
    a['q_b']=source.matrix('self_attn.q_b_proj',device,rows=slice(rank*2048,(rank+1)*2048)).half()
    kvb=source.matrix('self_attn.kv_b_proj',device,rows=slice(rank*3584,(rank+1)*3584)).reshape(8,448,512)
    a['k_b']=kvb[:,:192,:].transpose(1,2).half().contiguous()
    a['v_b']=kvb[:,192:,:].half().contiguous()
    a['o']=source.matrix('self_attn.o_proj',device,cols=slice(rank*2048,(rank+1)*2048)).half()
    return a


def load_rank(path, device, down_dtype, *, source_root, layer, rank):
    from ops.moe_layer import MoELayerWeights
    if down_dtype not in ('int4','fp8'):raise ValueError(down_dtype)
    data=load_file(str(path),device='cpu')
    attn=load_attention(source_root,layer,rank,device)
    w=MoELayerWeights(*[data[k].to(device) for k in ['router','bias','gu','gu_scale',
                       'down_'+down_dtype,'down_'+down_dtype+'_scale','shared_gu','shared_down']])
    return attn,data['ffn_norm'].to(device),w


if __name__=='__main__':
    import os,sys
    convert_rank(sys.argv[1],sys.argv[2],12,int(os.environ['LOCAL_RANK']))


def load_indexer(root, layer, rank, device):
    # Four TP-local index heads; shared layers reuse selected IDs.
    config=json.loads((Path(root)/'config.json').read_text())
    kind=config['indexer_types'][layer]
    if kind=='shared':return None
    if kind!='full' or not 0<=rank<8:raise ValueError((kind,rank))
    if not config.get('indexer_rope_interleave',False):
        raise ValueError('GLM53 binding requires interleaved index RoPE')
    if (config['index_n_heads'],config['index_head_dim'])!=(32,128):
        raise ValueError('expected 32x128 indexer')
    source=Source(root,layer);base='self_attn.indexer.'
    return dict(
        wq_b=source.matrix(base+'wq_b',device,rows=slice(rank*512,(rank+1)*512)).half(),
        wk=source.matrix(base+'wk',device).half(),
        k_norm=source.get(base+'k_norm.weight').to(device=device,dtype=torch.float16),
        k_bias=source.get(base+'k_norm.bias').to(device=device,dtype=torch.float16),
        weights_proj=source.get(base+'weights_proj.weight')[rank*4:(rank+1)*4].to(device=device,dtype=torch.float32).contiguous())
