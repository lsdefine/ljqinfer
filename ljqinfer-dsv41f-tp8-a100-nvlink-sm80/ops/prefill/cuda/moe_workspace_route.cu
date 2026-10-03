// Device-side MoE routing: stable counting sort, no host readback, graph-capturable.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using B=__nv_bfloat16;
#define TILE 1024

__global__ void hist_kernel(const int64_t* __restrict__ ids, int64_t total, int topk,
                            int64_t base, int count, int64_t* __restrict__ blockcount){
  extern __shared__ int sh[];
  for(int e=threadIdx.x;e<count;e+=blockDim.x) sh[e]=0;
  __syncthreads();
  int64_t start=(int64_t)blockIdx.x*TILE;
  for(int64_t i=start+threadIdx.x;i<(start+TILE<total?start+TILE:total);i+=blockDim.x){
    int64_t e=ids[i]-base;
    if(e>=0&&e<count) atomicAdd(&sh[e],1);
  }
  __syncthreads();
  for(int e=threadIdx.x;e<count;e+=blockDim.x) blockcount[(int64_t)blockIdx.x*count+e]=sh[e];
}

// single block: per-expert totals, expert bases, per-block bases (stable order)
__global__ void scan_kernel(int64_t* __restrict__ blockcount, int nblocks, int count,
                            int64_t* __restrict__ counts, int64_t* __restrict__ offsets){
  __shared__ int64_t sbase[512];
  for(int e=threadIdx.x;e<count;e+=blockDim.x){
    int64_t tot=0;
    for(int b=0;b<nblocks;++b) tot+=blockcount[(int64_t)b*count+e];
    counts[e]=tot;
  }
  __syncthreads();
  // exclusive scan over experts (serial on thread 0 for determinism)
  if(threadIdx.x==0){
    int64_t acc=0;
    for(int j=0;j<count;++j){ sbase[j]=acc; acc+=counts[j]; }
    for(int j=0;j<count;++j) offsets[j]=sbase[j];
    offsets[count]=acc;
  }
  __syncthreads();
  for(int e=threadIdx.x;e<count;e+=blockDim.x){
    int64_t acc=sbase[e];
    for(int b=0;b<nblocks;++b){
      int64_t n=blockcount[(int64_t)b*count+e];
      blockcount[(int64_t)b*count+e]=acc;
      acc+=n;
    }
  }
}

__global__ void place_kernel(const int64_t* __restrict__ ids, int64_t total, int topk,
                             int64_t base, int count, const int64_t* __restrict__ blockbase,
                             int64_t* __restrict__ token, int64_t* __restrict__ choice){
  extern __shared__ int64_t fill[];
  for(int e=threadIdx.x;e<count;e+=blockDim.x) fill[e]=blockbase[(int64_t)blockIdx.x*count+e];
  __syncthreads();
  if(threadIdx.x==0){
    int64_t start=(int64_t)blockIdx.x*TILE, stop=(start+TILE<total?start+TILE:total);
    for(int64_t i=start;i<stop;++i){
      int64_t e=ids[i]-base;
      if(e>=0&&e<count){
        int64_t p=fill[e]++;
        token[p]=i/topk; choice[p]=i%topk;
      }
    }
  }
}

std::vector<torch::Tensor> route(torch::Tensor ids, int64_t base, int64_t count,
    torch::Tensor token, torch::Tensor choice, torch::Tensor counts,
    torch::Tensor offsets, torch::Tensor blockcount){
  TORCH_CHECK(ids.is_cuda()&&ids.is_contiguous()&&ids.scalar_type()==torch::kInt64&&ids.dim()==2,"ids [M,topk] int64 cuda");
  TORCH_CHECK(count>0&&count<=512,"1..512 experts per rank");
  for(auto& t:{token,choice,counts,offsets,blockcount})
    TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==torch::kInt64&&t.device()==ids.device(),"int64 cuda contiguous buffers");
  const int64_t M=ids.size(0), topk=ids.size(1), total=M*topk;
  TORCH_CHECK(token.numel()>=total&&choice.numel()>=total,"token/choice capacity");
  TORCH_CHECK(counts.numel()>=count&&offsets.numel()>=count+1,"counts/offsets capacity");
  c10::cuda::CUDAGuard guard(ids.device());
  const int nblocks=(int)((total+TILE-1)/TILE);
  TORCH_CHECK(nblocks>0,"empty routing input");
  TORCH_CHECK(blockcount.numel()>=(int64_t)nblocks*count,"blockcount capacity");
  auto stream=at::cuda::getCurrentCUDAStream();
  hist_kernel<<<nblocks,256,count*sizeof(int),stream>>>(ids.data_ptr<int64_t>(),total,(int)topk,base,(int)count,blockcount.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  scan_kernel<<<1,512,0,stream>>>(blockcount.data_ptr<int64_t>(),nblocks,(int)count,counts.data_ptr<int64_t>(),offsets.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  place_kernel<<<nblocks,64,count*sizeof(int64_t),stream>>>(ids.data_ptr<int64_t>(),total,(int)topk,base,(int)count,blockcount.data_ptr<int64_t>(),token.data_ptr<int64_t>(),choice.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {token,choice,counts,offsets};
}

// out[token[r]] += src[r], accumulated per token in routing-slot order.
// Scattering with one atomic per row let the topk contributions land in whatever
// order the hardware handed them over, so prefill was not reproducible: the fp32
// sums differed in the last bits, and by the deeper layers that was enough to flip
// a BF16 rounding and change the sampled token.  Gathering fixes the order without
// paying for it, since each output element is now owned by a single thread.
__global__ void invert_kernel(const int64_t* __restrict__ token, const int64_t* __restrict__ choice,
                              const int64_t* __restrict__ rows, int64_t* __restrict__ slot,
                              int64_t maxrows, int topk){
  int64_t n=*rows;
  for(int64_t r=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;r<maxrows;r+=(int64_t)blockDim.x*gridDim.x)
    if(r<n) slot[token[r]*topk+choice[r]]=r;
}

__global__ void gather_add_kernel(const B* __restrict__ src, const int64_t* __restrict__ slot,
                                  float* __restrict__ out, int64_t m, int k, int topk){
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<m*(int64_t)k;i+=(int64_t)blockDim.x*gridDim.x){
    int64_t t=i/k; int col=(int)(i%k); float sum=0.f;
    for(int j=0;j<topk;j++){
      int64_t r=slot[t*topk+j];
      if(r>=0) sum+=__bfloat162float(src[r*(int64_t)k+col]);
    }
    out[i]+=sum;
  }
}

torch::Tensor scatter_add(torch::Tensor src, torch::Tensor token, torch::Tensor choice,
                          torch::Tensor rows, torch::Tensor out, torch::Tensor slot,
                          int64_t maxrows, int64_t topk){
  TORCH_CHECK(src.is_cuda()&&src.is_contiguous()&&src.scalar_type()==torch::kBFloat16&&src.dim()==2,"src bf16 [R,d]");
  TORCH_CHECK(out.is_cuda()&&out.is_contiguous()&&out.scalar_type()==torch::kFloat32&&out.dim()==2,"out fp32 [M,d]");
  TORCH_CHECK(token.scalar_type()==torch::kInt64&&choice.scalar_type()==torch::kInt64
              &&rows.scalar_type()==torch::kInt64&&slot.scalar_type()==torch::kInt64,"int64 routing");
  TORCH_CHECK(out.size(1)==src.size(1),"width mismatch");
  TORCH_CHECK(maxrows>=0&&maxrows<=src.size(0)&&token.numel()>=maxrows&&choice.numel()>=maxrows,"maxrows capacity");
  TORCH_CHECK(topk>0&&slot.is_contiguous()&&slot.numel()>=out.size(0)*topk,"slot capacity");
  c10::cuda::CUDAGuard guard(src.device());
  if(maxrows==0) return out;
  int k=(int)src.size(1); int64_t m=out.size(0);
  auto stream=at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(slot.data_ptr<int64_t>(),0xFF,m*topk*sizeof(int64_t),stream));
  invert_kernel<<<(unsigned)std::min<int64_t>((maxrows+255)/256,65535),256,0,stream>>>(
      token.data_ptr<int64_t>(),choice.data_ptr<int64_t>(),rows.data_ptr<int64_t>(),
      slot.data_ptr<int64_t>(),maxrows,(int)topk);
  gather_add_kernel<<<(unsigned)std::min<int64_t>((m*k+255)/256,65535),256,0,stream>>>(
      (const B*)src.data_ptr(),slot.data_ptr<int64_t>(),out.data_ptr<float>(),m,k,(int)topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("route",&route);m.def("scatter_add",&scatter_add);}
