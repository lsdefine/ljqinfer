#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "/mnt/data/kw/llama.cpp/ggml/src/ggml-common.h"

__device__ __forceinline__ float h2f(ggml_half v) { return __half2float(v); }

__global__ void dq_q3(const block_q3_K *x, float *yy) {
 int64_t i=blockIdx.x,r=threadIdx.x/4,tid=r/2,is0=r%2,l0=16*is0+4*(threadIdx.x%4),n=tid/4,j=tid-4*n;
 uint8_t mask=1<<(4*n+j); int64_t is=8*n+2*j+is0; int shift=2*j;
 int8_t us=is<4?(x[i].scales[is]&15)|(((x[i].scales[is+8])&3)<<4):is<8?(x[i].scales[is]&15)|(((x[i].scales[is+4]>>2)&3)<<4):is<12?(x[i].scales[is-8]>>4)|(((x[i].scales[is]>>4)&3)<<4):(x[i].scales[is-8]>>4)|(((x[i].scales[is-4]>>6)&3)<<4);
 float dl=h2f(x[i].d)*(us-32); float *y=yy+i*QK_K+128*n+32*j; const uint8_t*q=x[i].qs+32*n,*hm=x[i].hmask;
 #pragma unroll
 for(int l=l0;l<l0+4;l++) y[l]=dl*((int8_t)((q[l]>>shift)&3)-((hm[l]&mask)?0:4));
}
__device__ __forceinline__ void scale_min(int j,const uint8_t*q,uint8_t&d,uint8_t&m){if(j<4){d=q[j]&63;m=q[j+4]&63;}else{d=(q[j+4]&15)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}}
__global__ void dq_q4(const block_q4_K*x,float*yy){int64_t i=blockIdx.x,tid=threadIdx.x,il=tid/8,ir=tid%8,is=2*il;float*y=yy+i*256+64*il+4*ir;float da=__low2float(x[i].dm),dm=__high2float(x[i].dm);const uint8_t*q=x[i].qs+32*il+4*ir;uint8_t sc,m;scale_min(is,x[i].scales,sc,m);float d1=da*sc,m1=dm*m;scale_min(is+1,x[i].scales,sc,m);float d2=da*sc,m2=dm*m;for(int l=0;l<4;l++){y[l]=d1*(q[l]&15)-m1;y[l+32]=d2*(q[l]>>4)-m2;}}
__global__ void dq_q5(const block_q5_K*x,float*yy){int64_t i=blockIdx.x,tid=threadIdx.x,il=tid/16,ir=tid%16,is=2*il;float*y=yy+i*256+64*il+2*ir;float da=__low2float(x[i].dm),dm=__high2float(x[i].dm);const uint8_t*ql=x[i].qs+32*il+2*ir,*qh=x[i].qh+2*ir;uint8_t sc,m;scale_min(is,x[i].scales,sc,m);float d1=da*sc,m1=dm*m;scale_min(is+1,x[i].scales,sc,m);float d2=da*sc,m2=dm*m;uint8_t hm=1<<(2*il);y[0]=d1*((ql[0]&15)+((qh[0]&hm)?16:0))-m1;y[1]=d1*((ql[1]&15)+((qh[1]&hm)?16:0))-m1;hm<<=1;y[32]=d2*((ql[0]>>4)+((qh[0]&hm)?16:0))-m2;y[33]=d2*((ql[1]>>4)+((qh[1]&hm)?16:0))-m2;}
__global__ void dq_q6(const block_q6_K*x,float*yy){int64_t i=blockIdx.x,tid=threadIdx.x,ip=tid/32,il=tid-32*ip,is=8*ip+il/16;float*y=yy+i*256+128*ip+il;float d=h2f(x[i].d);const uint8_t*ql=x[i].ql+64*ip+il;uint8_t qh=x[i].qh[32*ip+il];const int8_t*sc=x[i].scales+is;y[0]=d*sc[0]*((int8_t)((ql[0]&15)|(((qh>>0)&3)<<4))-32);y[32]=d*sc[2]*((int8_t)((ql[32]&15)|(((qh>>2)&3)<<4))-32);y[64]=d*sc[4]*((int8_t)((ql[0]>>4)|(((qh>>4)&3)<<4))-32);y[96]=d*sc[6]*((int8_t)((ql[32]>>4)|(((qh>>6)&3)<<4))-32);}
torch::Tensor dequant_k_cuda(torch::Tensor p,int64_t K,int64_t qt){TORCH_CHECK(p.is_cuda()&&p.scalar_type()==torch::kUInt8&&p.is_contiguous());TORCH_CHECK(K%256==0);int64_t M=p.numel()/(qt==3?110:qt==4?144:qt==5?176:qt==6?210:1)*256/K;TORCH_CHECK(qt>=3&&qt<=6&&M*K/256*(qt==3?110:qt==4?144:qt==5?176:210)==p.numel());auto y=torch::empty({M,K},p.options().dtype(torch::kFloat32));int64_t nb=M*K/256;auto s=at::cuda::getCurrentCUDAStream();if(qt==3)dq_q3<<<nb,64,0,s>>>((block_q3_K*)p.data_ptr(),y.data_ptr<float>());else if(qt==4)dq_q4<<<nb,32,0,s>>>((block_q4_K*)p.data_ptr(),y.data_ptr<float>());else if(qt==5)dq_q5<<<nb,64,0,s>>>((block_q5_K*)p.data_ptr(),y.data_ptr<float>());else dq_q6<<<nb,64,0,s>>>((block_q6_K*)p.data_ptr(),y.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
