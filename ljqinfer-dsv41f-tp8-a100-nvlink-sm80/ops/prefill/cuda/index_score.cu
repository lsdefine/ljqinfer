#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// V4.1: sum_h(relu(dot_fp32(q_h,k))*weight_h) / sqrt(D) / sqrt(H_global).
// No BF16 score rounding, causal mask, or TP reduction hidden in this leaf.
__global__ void score_kernel(const float* q,const float* k,const float* w,float* out,
                            int N,int H,float sd,float sh){
 __shared__ float qs[4*128];
 __shared__ float ws[4];
 int row=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
 for(int i=tid;i<H*128;i+=256) qs[i]=q[(long)row*H*128+i];
 if(tid<H) ws[tid]=w[(long)row*H+tid];
 __syncthreads();
 for(int j=0;j<8;j++){
  int key=blockIdx.y*64+warp+j*8;
  if(key>=N) continue;
  float kv[4];
  #pragma unroll
  for(int d=0;d<4;d++) kv[d]=k[(long)key*128+lane+32*d];
  float sum=0.f;
  for(int h=0;h<H;h++){
   float dot=0.f;
   #pragma unroll
   for(int d=0;d<4;d++) dot=__fadd_rn(dot,__fmul_rn(qs[h*128+lane+32*d],kv[d]));
   #pragma unroll
   for(int shift=16;shift>0;shift>>=1) dot=__fadd_rn(dot,__shfl_down_sync(0xffffffff,dot,shift));
   if(lane==0) sum=__fadd_rn(sum,__fmul_rn(fmaxf(dot,0.f),ws[h]));
  }
  if(lane==0) out[(long)row*N+key]=__fmul_rn(__fmul_rn(sum,sd),sh);
 }
}
void score_out(torch::Tensor q,torch::Tensor k,torch::Tensor w,torch::Tensor out,int64_t total_heads){
 TORCH_CHECK(q.is_cuda() && k.is_cuda() && w.is_cuda() && out.is_cuda(),"CUDA required");
 TORCH_CHECK(q.device()==k.device() && q.device()==w.device() && q.device()==out.device(),"device mismatch");
 TORCH_CHECK(q.scalar_type()==at::kFloat && k.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat && out.scalar_type()==at::kFloat,"FP32 required");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && w.is_contiguous() && out.is_contiguous(),"contiguous required");
 TORCH_CHECK(q.dim()==3 && k.dim()==2 && w.dim()==2 && out.dim()==2,"ranks");
 TORCH_CHECK(q.size(2)==128 && k.size(1)==128 && q.size(1)>0 && q.size(1)<=4,"D128 H1..4");
 TORCH_CHECK(w.size(0)==q.size(0) && w.size(1)==q.size(1) && out.size(0)==q.size(0) && out.size(1)==k.size(0) && total_heads>=q.size(1),"shapes");
 c10::cuda::CUDAGuard guard(q.device());
 if(!q.size(0)||!k.size(0)) return;
 score_kernel<<<dim3(q.size(0),(k.size(0)+63)/64),256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(),k.data_ptr<float>(),w.data_ptr<float>(),out.data_ptr<float>(),k.size(0),q.size(1),float(1.0/sqrt(128.0)),float(1.0/sqrt(double(total_heads))));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("score_out",&score_out);}
