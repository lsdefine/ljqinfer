#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
// Reproduce FP32 contiguous torch mean's input-vector4 and ascending shuffle tree.
__global__ void exact_square_mean(const half* x,float* y,int n,int h,int stride){
 int row=blockIdx.x*blockDim.y+threadIdx.y;
 if(row>=n)return;
 int lane=threadIdx.x;
 float a=0,b=0,c=0,d=0;
 for(int j=lane*4;j<h;j+=128){
  float v0=__half2float(x[row*stride+j]);
  float v1=__half2float(x[row*stride+j+1]);
  float v2=__half2float(x[row*stride+j+2]);
  float v3=__half2float(x[row*stride+j+3]);
  a=__fadd_rn(a,__fmul_rn(v0,v0));
  b=__fadd_rn(b,__fmul_rn(v1,v1));
  c=__fadd_rn(c,__fmul_rn(v2,v2));
  d=__fadd_rn(d,__fmul_rn(v3,v3));
 }
 a=__fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);
 #pragma unroll
 for(int offset=1;offset<32;offset*=2)
  a=__fadd_rn(a,__shfl_down_sync(0xffffffff,a,offset));
 if(lane==0)y[row]=__fmul_rn(a,1.0f/h);
}
void forward(torch::Tensor x,torch::Tensor y){
 TORCH_CHECK(x.is_cuda()&&y.is_cuda()&&x.device()==y.device());
 TORCH_CHECK(x.scalar_type()==torch::kFloat16&&y.scalar_type()==torch::kFloat32);
 TORCH_CHECK(x.dim()==2&&x.stride(1)==1&&y.is_contiguous()&&y.numel()==x.size(0));
 TORCH_CHECK(x.size(0)>=16&&x.size(1)>128&&x.size(1)<=6144&&x.size(1)%128==0);
 c10::cuda::CUDAGuard guard(x.device());
 exact_square_mean<<<(x.size(0)+7)/8,dim3(32,8),0,at::cuda::getCurrentCUDAStream()>>>(
  (const half*)x.data_ptr(),y.data_ptr<float>(),x.size(0),x.size(1),x.stride(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
