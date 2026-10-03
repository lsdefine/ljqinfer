#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
using B=__nv_bfloat16;
__device__ float quantized(float v){
 float mx=fabsf(v);
 for(int d=16;d;d/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 float s=exp2f(ceilf(log2f(fmaxf(mx,1.e-4f)/448.f)));
 __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,v/s)));
 return float(q)*s;
}
__global__ void gather_quant_kernel(const B* x,const int64_t* tok,B* y,int64_t rows,int k){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*k;i+=int64_t(blockDim.x)*gridDim.x){
 float v=__bfloat162float(x[tok[i/k]*k+i%k]);
 y[i]=__float2bfloat16_rn(quantized(v));
 }
}
__global__ void glu_quant_kernel(const B* both,const float* p,const int64_t* tok,const int64_t* ch,B* y,int64_t rows,int k,int topk,float limit){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*k;i+=int64_t(blockDim.x)*gridDim.x){
 int64_t row=i/k,col=i%k;
 float g=__bfloat162float(both[row*(2*k)+col]);
 float u=__bfloat162float(both[row*(2*k)+k+col]);
 if(limit>0){g=fminf(g,limit);u=fmaxf(-limit,fminf(u,limit));}
 float v=(g/(1.f+expf(-g)))*u;
 v=v*p[tok[row]*topk+ch[row]];
 // Preserve original SwiGLU BF16 rounding before block-FP8 quantization.
 v=__bfloat162float(__float2bfloat16_rn(v));
 y[i]=__float2bfloat16_rn(quantized(v));
 }
}
void check(torch::Tensor x,at::ScalarType type){TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.scalar_type()==type,"CUDA contiguous dtype contract");}
torch::Tensor gather_quant(torch::Tensor x,torch::Tensor tok,torch::Tensor y){
 check(x,torch::kBFloat16);check(tok,torch::kInt64);
 TORCH_CHECK(x.dim()==2&&tok.dim()==1&&x.size(1)>0&&x.size(1)%32==0&&x.device()==tok.device(),"gather geometry/device");
 c10::cuda::CUDAGuard guard(x.device());
 check(y,torch::kBFloat16);
 TORCH_CHECK(y.device()==x.device() && y.dim()==2 && y.size(0)==tok.numel() && y.size(1)==x.size(1),"gather output geometry/device");
 TORCH_CHECK(!y.is_alias_of(x) && !y.is_alias_of(tok),"gather output must not alias inputs");
 if(y.numel())gather_quant_kernel<<<std::min<int64_t>((y.numel()+255)/256,65535),256,0,at::cuda::getCurrentCUDAStream()>>>((const B*)x.data_ptr(),tok.data_ptr<int64_t>(),(B*)y.data_ptr(),tok.numel(),x.size(1));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
torch::Tensor glu_quant(torch::Tensor both,torch::Tensor p,torch::Tensor tok,torch::Tensor ch,double limit,torch::Tensor y){
 check(both,torch::kBFloat16);check(p,torch::kFloat32);check(tok,torch::kInt64);check(ch,torch::kInt64);
 TORCH_CHECK(both.dim()==2&&p.dim()==2&&tok.dim()==1&&ch.dim()==1&&both.size(1)>0&&both.size(1)%64==0&&both.size(0)==tok.numel()&&ch.numel()==tok.numel(),"glu geometry");
 TORCH_CHECK(both.device()==p.device()&&both.device()==tok.device()&&both.device()==ch.device(),"glu devices");
 c10::cuda::CUDAGuard guard(both.device());int k=both.size(1)/2;
 check(y,torch::kBFloat16);
 TORCH_CHECK(y.device()==both.device() && y.dim()==2 && y.size(0)==both.size(0) && y.size(1)==k,"glu output geometry/device");
 TORCH_CHECK(!y.is_alias_of(both) && !y.is_alias_of(p) && !y.is_alias_of(tok) && !y.is_alias_of(ch),"glu output must not alias inputs");
 if(y.numel())glu_quant_kernel<<<std::min<int64_t>((y.numel()+255)/256,65535),256,0,at::cuda::getCurrentCUDAStream()>>>((const B*)both.data_ptr(),p.data_ptr<float>(),tok.data_ptr<int64_t>(),ch.data_ptr<int64_t>(),(B*)y.data_ptr(),both.size(0),k,p.size(1),limit);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gather_quant",&gather_quant);m.def("glu_quant",&glu_quant);}
