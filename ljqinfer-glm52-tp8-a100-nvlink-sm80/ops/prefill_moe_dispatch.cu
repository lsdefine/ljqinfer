#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

__global__ void moe_count256(const int64_t* e,int n,int64_t* cnt){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n) atomicAdd((unsigned long long*)&cnt[e[i]],1ULL);
}
__global__ void moe_scan256(const int64_t* cnt,int64_t* off,int64_t* pos){
 __shared__ int64_t a[256]; int t=threadIdx.x;
 a[t]=cnt[t]; __syncthreads();
 for(int d=1;d<256;d<<=1){ int64_t v=t>=d?a[t-d]:0; __syncthreads(); if(t>=d)a[t]+=v; __syncthreads(); }
 off[t+1]=a[t]; off[0]=0; pos[t]=t? a[t-1]:0;
}
__global__ void moe_scatter256(const int64_t* e,const float* w,int n,int64_t* pos,int64_t* tok,float* outw){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){int x=(int)e[i]; unsigned long long j=atomicAdd((unsigned long long*)&pos[x],1ULL); tok[j]=i>>3; outw[j]=w[i];}
}
// 256-bin counting dispatch: replaces argsort(N)+bincount(.cpu). Returns {tok[N],ww[N,1],off_cpu[257]}.
std::vector<torch::Tensor> dispatch_meta(torch::Tensor ei,torch::Tensor ew){
 TORCH_CHECK(ei.is_cuda()&&ew.is_cuda()&&ei.scalar_type()==torch::kInt64&&ew.scalar_type()==torch::kFloat32,"cuda int64/float32");
 TORCH_CHECK(ei.is_contiguous()&&ew.is_contiguous()&&ei.numel()==ew.numel()&&ei.dim()==2&&ei.size(1)==8,"shape");
 c10::cuda::CUDAGuard g(ei.device()); int n=ei.numel();
 auto cnt=torch::zeros({256},ei.options()); auto off=torch::empty({257},ei.options()); auto pos=torch::empty({256},ei.options());
 auto tok=torch::empty({n},ei.options()); auto ww=torch::empty({n,1},ew.options()); auto stream=at::cuda::getCurrentCUDAStream();
 moe_count256<<<(n+255)/256,256,0,stream>>>(ei.data_ptr<int64_t>(),n,cnt.data_ptr<int64_t>());
 moe_scan256<<<1,256,0,stream>>>(cnt.data_ptr<int64_t>(),off.data_ptr<int64_t>(),pos.data_ptr<int64_t>());
 moe_scatter256<<<(n+255)/256,256,0,stream>>>(ei.data_ptr<int64_t>(),ew.data_ptr<float>(),n,pos.data_ptr<int64_t>(),tok.data_ptr<int64_t>(),ww.data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {tok,ww,off.to(torch::kCPU)};
}
