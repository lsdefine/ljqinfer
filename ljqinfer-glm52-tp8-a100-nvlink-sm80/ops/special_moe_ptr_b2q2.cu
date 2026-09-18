#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
constexpr int D=6144,H=256,K=8;
__global__ void setup(const half*x,const int64_t*id,const half*wg,const half*wu,const half*wd,int64_t*p,int N){
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=N)return; int64_t e=id[i],t=i/K;
 p[0*N+i]=(int64_t)(wg+e*(long long)H*D); p[1*N+i]=(int64_t)(x+t*(long long)D); p[2*N+i]=(int64_t)(wu+e*(long long)H*D);
 p[3*N+i]=(int64_t)(wd+e*(long long)D*H);
}
__global__ void setup_outputs(int64_t*p,half*g,half*u,int N){int i=threadIdx.x;if(i<N){p[4*N+i]=(int64_t)(g+i*H);p[5*N+i]=(int64_t)(u+i*H);}}
__global__ void setup_down(int64_t*p,half*h,half*y,int N){int i=threadIdx.x;if(i<N){p[1*N+i]=(int64_t)(h+i*H);p[4*N+i]=(int64_t)(y+(long long)i*D);}}
__global__ void silu_mul(const half*g,const half*u,half*h,int n){int i=threadIdx.x+blockIdx.x*blockDim.x;if(i<n){float a=__half2float(g[i]);h[i]=__float2half_rn((a/(1.f+expf(-a)))*__half2float(u[i]));}}
__global__ void reduce(const half*y,const float*ew,float*out,int B){int j=threadIdx.x+blockIdx.x*blockDim.x,b=blockIdx.y;if(j<D&&b<B){float s=0;for(int k=0;k<K;k++)s+=__half2float(y[((b*K+k)*(long long)D)+j])*ew[b*K+k];out[b*(long long)D+j]=s;}}
torch::Tensor ptr_bgemm_cuda(torch::Tensor x,torch::Tensor ids,torch::Tensor ew,torch::Tensor wg,torch::Tensor wu,torch::Tensor wd){
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat16 && x.is_contiguous(), "x must be contiguous CUDA float16");
 TORCH_CHECK(x.dim()==2 && x.size(1)==D, "x must have shape [B, 6144]");
 const int64_t B=x.size(0), N=B*K;
 TORCH_CHECK(B>=1 && B<=4, "B must be in [1, 4]");
 TORCH_CHECK(ids.is_cuda() && ids.scalar_type()==torch::kInt64 && ids.is_contiguous(), "ids must be contiguous CUDA int64");
 TORCH_CHECK(ew.is_cuda() && ew.scalar_type()==torch::kFloat32 && ew.is_contiguous(), "ew must be contiguous CUDA float32");
 TORCH_CHECK(ids.dim()==2 && ids.size(0)==B && ids.size(1)==K, "ids must have shape [B, 8]");
 TORCH_CHECK(ew.dim()==2 && ew.size(0)==B && ew.size(1)==K, "ew must have shape [B, 8]");
 TORCH_CHECK(wg.is_cuda() && wu.is_cuda() && wd.is_cuda(), "weights must be CUDA tensors");
 TORCH_CHECK(wg.scalar_type()==torch::kFloat16 && wu.scalar_type()==torch::kFloat16 && wd.scalar_type()==torch::kFloat16, "weights must be float16");
 TORCH_CHECK(wg.is_contiguous() && wu.is_contiguous() && wd.is_contiguous(), "weights must be contiguous");
 TORCH_CHECK(wg.dim()==3 && wg.size(1)==H && wg.size(2)==D, "wg must have shape [E, 256, 6144]");
 TORCH_CHECK(wu.sizes()==wg.sizes(), "wu must have the same shape as wg");
 TORCH_CHECK(wd.dim()==3 && wd.size(0)==wg.size(0) && wd.size(1)==D && wd.size(2)==H, "wd must have shape [E, 6144, 256]");
 TORCH_CHECK(ids.device()==x.device() && ew.device()==x.device() && wg.device()==x.device() && wu.device()==x.device() && wd.device()==x.device(), "all tensors must be on the same CUDA device");
 c10::cuda::CUDAGuard guard(x.device());auto opt=x.options();auto gate=torch::empty({N,H},opt),up=torch::empty({N,H},opt),hid=torch::empty({N,H},opt),ye=torch::empty({N,D},opt),out=torch::empty({B,D},x.options().dtype(torch::kFloat32));auto pt=torch::empty({6,N},x.options().dtype(torch::kInt64));
 auto stream=at::cuda::getCurrentCUDAStream();setup<<<1,32,0,stream>>>((half*)x.data_ptr(),ids.data_ptr<int64_t>(),(half*)wg.data_ptr(),(half*)wu.data_ptr(),(half*)wd.data_ptr(),pt.data_ptr<int64_t>(),N);
 int64_t* p=pt.data_ptr<int64_t>();
 // p rows: wg,x,wu,wd; rows 4/5 are output pointers.
 setup_outputs<<<1,32,0,stream>>>(p,(half*)gate.data_ptr(),(half*)up.data_ptr(),N);
 auto handle=at::cuda::getCurrentCUDABlasHandle();float alpha=1.f,beta=0.f;
 auto call=[&](const half**A,const half**B,half**C,int m,int k,int lda,int ldb,int ldc){TORCH_CUDABLAS_CHECK(cublasGemmBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,1,k,&alpha,(const void**)A,CUDA_R_16F,lda,(const void**)B,CUDA_R_16F,ldb,&beta,(void**)C,CUDA_R_16F,ldc,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));};
 call((const half**)(p+0*N),(const half**)(p+1*N),(half**)(p+4*N),H,D,D,D,H);
 call((const half**)(p+2*N),(const half**)(p+1*N),(half**)(p+5*N),H,D,D,D,H);
 silu_mul<<<(N*H+255)/256,256,0,stream>>>((half*)gate.data_ptr(),(half*)up.data_ptr(),(half*)hid.data_ptr(),N*H);
 setup_down<<<1,32,0,stream>>>(p,(half*)hid.data_ptr(),(half*)ye.data_ptr(),N);
 call((const half**)(p+3*N),(const half**)(p+1*N),(half**)(p+4*N),D,H,H,H,D);
 reduce<<<dim3((D+255)/256,B),256,0,stream>>>((half*)ye.data_ptr(),ew.data_ptr<float>(),out.data_ptr<float>(),B);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
