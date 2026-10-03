#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
template<int H,int BW> __global__ void fused_rms(const half* x,const half* w,half* y,int n,int stride){
 int row=blockIdx.x;
 if(row>=n)return;
 int lane=threadIdx.x;
 uint2 cache[H/(BW*4)];
 float a=0,b=0,c=0,d=0;
 #pragma unroll
 for(int t=0;t<H/(BW*4);++t){
  int j=lane*4+t*(BW*4);
  uint2 v=*reinterpret_cast<const uint2*>(x+row*stride+j);cache[t]=v;
  half2 h0=*reinterpret_cast<half2*>(&v.x),h1=*reinterpret_cast<half2*>(&v.y);
  float2 f0=__half22float2(h0),f1=__half22float2(h1);
  a=__fadd_rn(a,__fmul_rn(f0.x,f0.x));b=__fadd_rn(b,__fmul_rn(f0.y,f0.y));
  c=__fadd_rn(c,__fmul_rn(f1.x,f1.x));d=__fadd_rn(d,__fmul_rn(f1.y,f1.y));
 }
 a=__fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);
 __shared__ float buf[BW];
 buf[lane]=a;
 #pragma unroll
 for(int offset=BW/2;offset>=32;offset/=2){
  __syncthreads();
  if(lane<offset){a=__fadd_rn(a,buf[lane+offset]);buf[lane]=a;}
 }
 __syncthreads();
 #pragma unroll
 for(int offset=1;offset<32;offset*=2)a=__fadd_rn(a,__shfl_down_sync(0xffffffff,a,offset));
 if(lane==0)buf[0]=a;
 __syncthreads();
 float mean=__fmul_rn(buf[0],1.0f/H);
 float inv=rsqrtf(__fadd_rn(mean,1.0e-5f));
 #pragma unroll
 for(int t=0;t<H/(BW*4);++t){
  int j=lane*4+t*(BW*4);uint2 v=cache[t];
  float2 f0=__half22float2(*reinterpret_cast<half2*>(&v.x)),f1=__half22float2(*reinterpret_cast<half2*>(&v.y));
  uint2 wv=*reinterpret_cast<const uint2*>(w+j);
  float2 w0=__half22float2(*reinterpret_cast<half2*>(&wv.x)),w1=__half22float2(*reinterpret_cast<half2*>(&wv.y));
  half2 o0=__floats2half2_rn(__fmul_rn(__fmul_rn(f0.x,inv),w0.x),__fmul_rn(__fmul_rn(f0.y,inv),w0.y));
  half2 o1=__floats2half2_rn(__fmul_rn(__fmul_rn(f1.x,inv),w1.x),__fmul_rn(__fmul_rn(f1.y,inv),w1.y));
  uint2 out=make_uint2(*reinterpret_cast<unsigned*>(&o0),*reinterpret_cast<unsigned*>(&o1));
  *reinterpret_cast<uint2*>(y+row*H+j)=out;
 }
}
void forward(torch::Tensor x,torch::Tensor w,torch::Tensor y){
 TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&y.is_cuda()&&x.device()==w.device()&&x.device()==y.device());
 TORCH_CHECK(x.scalar_type()==torch::kFloat16&&w.scalar_type()==x.scalar_type()&&y.scalar_type()==x.scalar_type());
 TORCH_CHECK(x.dim()==2&&x.stride(1)==1&&x.stride(0)%4==0&&y.is_contiguous()&&w.is_contiguous()&&y.sizes()==x.sizes()&&w.numel()==x.size(1));
 TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%8==0 && reinterpret_cast<uintptr_t>(w.data_ptr())%8==0 && reinterpret_cast<uintptr_t>(y.data_ptr())%8==0);
 int n=x.size(0),h=x.size(1),stride=x.stride(0);TORCH_CHECK(n>0&&n<16);
 c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 int bh=1;while(bh*2<=n)bh*=2;
 int bw=std::min(h/4,512/bh);
 #define L(H,BW) fused_rms<H,BW><<<n,BW,0,stream>>>((const half*)x.data_ptr(),(const half*)w.data_ptr(),(half*)y.data_ptr(),n,stride)
 #define LAUNCH(H) if(bw==64){L(H,64);}else if(bw==128){L(H,128);}else if(bw==256){if constexpr(H>=1024){L(H,256);}}else if(bw==512){if constexpr(H>=2048){L(H,512);}}else{TORCH_CHECK(false,"bad width");}
 if(h==512){LAUNCH(512);}else if(h==2048){LAUNCH(2048);}else if(h==6144){LAUNCH(6144);}else{TORCH_CHECK(false,"unsupported RMS width");}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
