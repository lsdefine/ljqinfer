#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ float to_float(float x) {return x;}
__device__ float to_float(__nv_bfloat16 x) {return __bfloat162float(x);}

template<class T> __global__ void quant(const T* x,float* y,int64_t count) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
        float v=to_float(x[i]),mx=fabsf(v);
        for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
        float scale=exp2f(ceilf(log2f(fmaxf(mx,1.e-4f)/448.f)));
        __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,v/scale)));
        y[i]=float(q)*scale;
    }
}
torch::Tensor activation(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(),"contiguous CUDA input required");
    TORCH_CHECK(x.dim()>0 && x.size(-1)>0 && x.size(-1)%32==0,"K32 geometry");
    TORCH_CHECK(x.scalar_type()==torch::kBFloat16 || x.scalar_type()==torch::kFloat32,"BF16/FP32 input required");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(y.device()==x.device() && y.scalar_type()==torch::kFloat32 && y.is_contiguous() && y.sizes()==x.sizes(), "FP32 output geometry/device/contiguity");
    TORCH_CHECK(!y.is_alias_of(x), "output must not alias input");
    if(!x.numel())return y;
    int blocks=int(std::min<int64_t>((x.numel()+255)/256,65535));
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    if(x.scalar_type()==torch::kBFloat16)
        quant<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x.data_ptr(),y.data_ptr<float>(),x.numel());
    else quant<<<blocks,256,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("activation",&activation);}
