#include <cstdint>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "/mnt/data/kw/llama.cpp/ggml/src/ggml-common.h"

__device__ __forceinline__ float h2f(ggml_half v) { return __half2float(v); }

// One block = 1 IQ3 block (256 weights). 32 threads, each writes half2 x4 = 8 halfs.
// Multi-block per CUDA block: each thread-block processes BPK IQ3 blocks.
template<int BPK>
__global__ void iq3_kernel_mp(const uint8_t *vx, half *yy, int64_t nblocks){
  int64_t base=(int64_t)blockIdx.x*BPK;
  int tid=threadIdx.x, il=tid/8, ib=tid%8;
  #pragma unroll
  for(int k=0;k<BPK;k++){
    int64_t i=base+k; if(i>=nblocks) return;
    const block_iq3_xxs *x=(const block_iq3_xxs*)vx;
    half *y=yy+i*256+32*ib+8*il;
    const uint8_t *q3=x[i].qs+8*ib;
    const uint16_t *gas=(const uint16_t*)(x[i].qs+256/4)+2*ib;
    const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);
    const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);
    uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);
    float d=h2f(x[i].d)*(0.5f+(aux>>28))*0.5f;
    uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];
    half2 o0,o1,o2,o3;
    o0.x=__float2half(d*g1[0]*((signs&1u)?-1.f:1.f));
    o0.y=__float2half(d*g1[1]*((signs&2u)?-1.f:1.f));
    o1.x=__float2half(d*g1[2]*((signs&4u)?-1.f:1.f));
    o1.y=__float2half(d*g1[3]*((signs&8u)?-1.f:1.f));
    o2.x=__float2half(d*g2[0]*((signs&16u)?-1.f:1.f));
    o2.y=__float2half(d*g2[1]*((signs&32u)?-1.f:1.f));
    o3.x=__float2half(d*g2[2]*((signs&64u)?-1.f:1.f));
    o3.y=__float2half(d*g2[3]*((signs&128u)?-1.f:1.f));
    ((half2*)y)[0]=o0; ((half2*)y)[1]=o1; ((half2*)y)[2]=o2; ((half2*)y)[3]=o3;
  }
}

template<int BPK>
__global__ void iq4_kernel_mp(const uint8_t *vx, half *yy, int64_t nblocks){
  int64_t base=(int64_t)blockIdx.x*BPK;
  int tid=threadIdx.x, il=tid/8, ib=tid%8;
  #pragma unroll
  for(int k=0;k<BPK;k++){
    int64_t i=base+k; if(i>=nblocks) return;
    const block_iq4_xs *x=(const block_iq4_xs*)vx;
    half *y=yy+i*256+32*ib+4*il;
    const uint8_t *q4=x[i].qs+16*ib+4*il;
    int sc=((x[i].scales_l[ib/2]>>(4*(ib%2)))&15)|(((x[i].scales_h>>(2*ib))&3)<<4);
    float d=h2f(x[i].d)*(sc-32);
    half2 o0,o1,o2,o3;
    o0.x=__float2half(d*kvalues_iq4nl[q4[0]&15]);
    o0.y=__float2half(d*kvalues_iq4nl[q4[1]&15]);
    o1.x=__float2half(d*kvalues_iq4nl[q4[2]&15]);
    o1.y=__float2half(d*kvalues_iq4nl[q4[3]&15]);
    o2.x=__float2half(d*kvalues_iq4nl[q4[0]>>4]);
    o2.y=__float2half(d*kvalues_iq4nl[q4[1]>>4]);
    o3.x=__float2half(d*kvalues_iq4nl[q4[2]>>4]);
    o3.y=__float2half(d*kvalues_iq4nl[q4[3]>>4]);
    ((half2*)y)[0]=o0; ((half2*)y)[1]=o1;
    ((half2*)(y+16))[0]=o2; ((half2*)(y+16))[1]=o3;
  }
}


template<class T> __global__ void iq3_kernel(const T *vx, half *yy, int64_t nblocks) {
    int64_t i=blockIdx.x; if (i>=nblocks) return;
    const block_iq3_xxs *x=(const block_iq3_xxs*)vx;
    int tid=threadIdx.x, il=tid/8, ib=tid%8;
    half *y=yy+i*256+32*ib+8*il;
    const uint8_t *q3=x[i].qs+8*ib;
    const uint16_t *gas=(const uint16_t*)(x[i].qs+256/4)+2*ib;
    const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);
    const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);
    uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);
    float d=h2f(x[i].d)*(0.5f+(aux>>28))*0.5f;
    uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];
    #pragma unroll
    for(int j=0;j<4;j++) {
        y[j]=__float2half(d*g1[j]*((signs&(1u<<j))?-1.f:1.f));
        y[j+4]=__float2half(d*g2[j]*((signs&(1u<<(j+4)))?-1.f:1.f));
    }
}

template<class T> __global__ void iq4_kernel(const T *vx, half *yy, int64_t nblocks) {
    int64_t i=blockIdx.x; if (i>=nblocks) return;
    const block_iq4_xs *x=(const block_iq4_xs*)vx;
    int tid=threadIdx.x, il=tid/8, ib=tid%8;
    half *y=yy+i*256+32*ib+4*il;
    const uint8_t *q4=x[i].qs+16*ib+4*il;
    int sc=((x[i].scales_l[ib/2]>>(4*(ib%2)))&15)|(((x[i].scales_h>>(2*ib))&3)<<4);
    float d=h2f(x[i].d)*(sc-32);
    #pragma unroll
    for(int j=0;j<4;j++) {
        y[j]=__float2half(d*kvalues_iq4nl[q4[j]&15]);
        y[j+16]=__float2half(d*kvalues_iq4nl[q4[j]>>4]);
    }
}


// ==== selected-expert dequant: fused index_select + dequant ====
// ==== selected dequant: p[E,nout,rb] act[na] -> [na*nout,K]
// Each row packs (K/256) IQ blocks contiguously (rb bytes).
template<int BPK>
__global__ void iq3_sel_mp(const uint8_t *vx,const int64_t *act_ids,half *yy,
                           int64_t nb,int64_t nout,int64_t bpr){
  int64_t base=(int64_t)blockIdx.x*BPK;
  int tid=threadIdx.x,il=tid/8,ib=tid%8;
  #pragma unroll
  for(int k=0;k<BPK;k++){
    int64_t oi=base+k; if(oi>=nb) return;
    int64_t row=oi/bpr;              // 0..na*nout-1
    int64_t blk=oi-row*bpr;          // block within row
    int64_t ae=row/nout;
    int64_t local=row-ae*nout;
    int64_t src=act_ids[ae]*(nout*bpr)+local*bpr+blk;
    const block_iq3_xxs *x=(const block_iq3_xxs*)vx;
    half *y=yy+oi*256+32*ib+8*il;
    const uint8_t *q3=x[src].qs+8*ib;
    const uint16_t *gas=(const uint16_t*)(x[src].qs+256/4)+2*ib;
    const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);
    const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);
    uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);
    float d=h2f(x[src].d)*(0.5f+(aux>>28))*0.5f;
    uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];
    half2 o0,o1,o2,o3;
    o0.x=__float2half(d*g1[0]*((signs&1u)?-1.f:1.f));   o0.y=__float2half(d*g1[1]*((signs&2u)?-1.f:1.f));
    o1.x=__float2half(d*g1[2]*((signs&4u)?-1.f:1.f));   o1.y=__float2half(d*g1[3]*((signs&8u)?-1.f:1.f));
    o2.x=__float2half(d*g2[0]*((signs&16u)?-1.f:1.f));  o2.y=__float2half(d*g2[1]*((signs&32u)?-1.f:1.f));
    o3.x=__float2half(d*g2[2]*((signs&64u)?-1.f:1.f));  o3.y=__float2half(d*g2[3]*((signs&128u)?-1.f:1.f));
    ((half2*)y)[0]=o0;((half2*)y)[1]=o1;((half2*)y)[2]=o2;((half2*)y)[3]=o3;
  }
}
template<int BPK>
__global__ void iq4_sel_mp(const uint8_t *vx,const int64_t *act_ids,half *yy,
                           int64_t nb,int64_t nout,int64_t bpr){
  int64_t base=(int64_t)blockIdx.x*BPK;
  int tid=threadIdx.x,il=tid/8,ib=tid%8;
  #pragma unroll
  for(int k=0;k<BPK;k++){
    int64_t oi=base+k; if(oi>=nb) return;
    int64_t row=oi/bpr;
    int64_t blk=oi-row*bpr;
    int64_t ae=row/nout;
    int64_t local=row-ae*nout;
    int64_t src=act_ids[ae]*(nout*bpr)+local*bpr+blk;
    const block_iq4_xs *x=(const block_iq4_xs*)vx;
    half *y=yy+oi*256+32*ib+4*il;
    const uint8_t *q4=x[src].qs+16*ib+4*il;
    int sc=((x[src].scales_l[ib/2]>>(4*(ib%2)))&15)|(((x[src].scales_h>>(2*ib))&3)<<4);
    float d=h2f(x[src].d)*(sc-32);
    half2 o0,o1,o2,o3;
    o0.x=__float2half(d*kvalues_iq4nl[q4[0]&15]); o0.y=__float2half(d*kvalues_iq4nl[q4[1]&15]);
    o1.x=__float2half(d*kvalues_iq4nl[q4[2]&15]); o1.y=__float2half(d*kvalues_iq4nl[q4[3]&15]);
    o2.x=__float2half(d*kvalues_iq4nl[q4[0]>>4]); o2.y=__float2half(d*kvalues_iq4nl[q4[1]>>4]);
    o3.x=__float2half(d*kvalues_iq4nl[q4[2]>>4]); o3.y=__float2half(d*kvalues_iq4nl[q4[3]>>4]);
    ((half2*)y)[0]=o0;((half2*)y)[1]=o1;((half2*)(y+16))[0]=o2;((half2*)(y+16))[1]=o3;
  }
}
torch::Tensor dequant_iq3_selected_cuda(torch::Tensor p,torch::Tensor act,int64_t K){
  TORCH_CHECK(p.dim()==3&&act.scalar_type()==torch::kInt64,"p[E,nout,rb] act int64");
  TORCH_CHECK(K%256==0,"K multiple of 256");
  int64_t na=act.numel(), nout=p.size(1), bpr=K/256;
  int64_t nb=na*nout*bpr;
  auto y=torch::empty({na*nout,K},p.options().dtype(torch::kFloat16));
  if(nb==0) return y;
  iq3_sel_mp<8><<<(nb+7)/8,32,0,at::cuda::getCurrentCUDAStream()>>>(
    p.data_ptr<uint8_t>(),act.data_ptr<int64_t>(),(half*)y.data_ptr<at::Half>(),nb,nout,bpr);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
__global__ void iq4_selected_to_q8_cache_kernel(const uint8_t* p,const int64_t* act,uint8_t* out,int64_t nout,int K,int rb){
  int64_t row=(int64_t)blockIdx.x;
  int lane=threadIdx.x&31, il=lane/8, ib=lane%8;
  int64_t ae=row/nout, local=row-ae*nout;
  int e=(int)act[ae];
  const uint8_t* rp=p+((int64_t)e*nout+local)*rb;
  int q8rb=K+(K>>5)*(int)sizeof(float);
  uint8_t* dst=out+row*q8rb;
  int8_t* qw=(int8_t*)dst; float* qs=(float*)(dst+K);
  for(int kb=0;kb<K/256;++kb){
    const block_iq4_xs& q=((const block_iq4_xs*)rp)[kb];
    const uint8_t* q4=q.qs+16*ib+4*il;
    uint32_t q4w=*reinterpret_cast<const uint32_t*>(q4);
    const uint32_t KV0=0xBFAD9881u, KV1=0xF6EADDCFu, KV2=0x26190D01u, KV3=0x71594535u;
    auto lut4=[&](uint32_t idx)->uint32_t{
      uint32_t t=(idx|(idx>>4))&0x00FF00FFu;
      uint32_t s=(t&0xFFu)|((t>>8)&0xFF00u);
      uint32_t a2=__byte_perm(KV0,KV1,s&0x7777u);
      uint32_t b2=__byte_perm(KV2,KV3,s&0x7777u);
      uint32_t m=((idx>>3)&0x01010101u)*0xFFu;
      return (a2&~m)|(b2&m);
    };
    uint32_t vlo=lut4(q4w&0x0F0F0F0Fu), vhi=lut4((q4w>>4)&0x0F0F0F0Fu);
    int off=kb*256+32*ib+4*il;
    *reinterpret_cast<uint32_t*>(qw+off)=vlo;
    *reinterpret_cast<uint32_t*>(qw+off+16)=vhi;
    if(il==0){
      int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);
      qs[kb*8+ib]=h2f(q.d)*(sc-32);
    }
  }
}
torch::Tensor materialize_q8_selected_cuda(torch::Tensor p,torch::Tensor act,int64_t K){
  TORCH_CHECK(p.dim()==3&&act.scalar_type()==torch::kInt64,"p[E,nout,rb] act int64");
  TORCH_CHECK(K%256==0,"K multiple of 256");
  int64_t na=act.numel(),nout=p.size(1); int rb=p.size(2),q8rb=K+(K>>5)*(int)sizeof(float);
  auto y=torch::empty({na*nout,q8rb},p.options().dtype(torch::kUInt8));
  iq4_selected_to_q8_cache_kernel<<<na*nout,32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),act.data_ptr<int64_t>(),y.data_ptr<uint8_t>(),nout,K,rb);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
torch::Tensor dequant_iq4_selected_cuda(torch::Tensor p,torch::Tensor act,int64_t K){
  TORCH_CHECK(p.dim()==3&&act.scalar_type()==torch::kInt64,"p[E,nout,rb] act int64");
  TORCH_CHECK(K%256==0,"K multiple of 256");
  int64_t na=act.numel(), nout=p.size(1), bpr=K/256;
  int64_t nb=na*nout*bpr;
  auto y=torch::empty({na*nout,K},p.options().dtype(torch::kFloat16));
  if(nb==0) return y;
  iq4_sel_mp<8><<<(nb+7)/8,32,0,at::cuda::getCurrentCUDAStream()>>>(
    p.data_ptr<uint8_t>(),act.data_ptr<int64_t>(),(half*)y.data_ptr<at::Half>(),nb,nout,bpr);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

torch::Tensor dequant_iq3_xxs_cuda(torch::Tensor p,int64_t K) {
    auto y=torch::empty({p.size(0),K},p.options().dtype(torch::kFloat16));
    int64_t nb=p.size(0)*(K/256);
    iq3_kernel_mp<8><<<((nb+7)/8),32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),(half*)y.data_ptr<at::Half>(),nb);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
torch::Tensor dequant_iq4_xs_cuda(torch::Tensor p,int64_t K) {
    auto y=torch::empty({p.size(0),K},p.options().dtype(torch::kFloat16));
    int64_t nb=p.size(0)*(K/256);
    iq4_kernel_mp<8><<<((nb+7)/8),32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),(half*)y.data_ptr<at::Half>(),nb);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int d=16; d; d>>=1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

__global__ void mmv_iq3_kernel(const uint8_t *vp, const float *xv, float *out, int K, int row_bytes) {
    int row=blockIdx.x, tid=threadIdx.x, il=tid/8, ib=tid%8;
    const uint8_t *rp=vp+(int64_t)row*row_bytes;
    float sum=0.f;
    for (int kb=0; kb<K/256; ++kb) {
        const block_iq3_xxs &q=((const block_iq3_xxs*)rp)[kb];
        const uint8_t *q3=q.qs+8*ib;
        const uint16_t *gas=(const uint16_t*)(q.qs+256/4)+2*ib;
        const uint8_t *g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);
        const uint8_t *g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);
        uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);
        float d=h2f(q.d)*(0.5f+(aux>>28))*0.5f;
        uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];
        int base=kb*256+32*ib+8*il;
        #pragma unroll
        for(int j=0;j<4;j++) {
            sum += d*g1[j]*((signs&(1u<<j))?-1.f:1.f)*xv[base+j];
            sum += d*g2[j]*((signs&(1u<<(j+4)))?-1.f:1.f)*xv[base+j+4];
        }
    }
    sum=warp_sum(sum); if(tid==0) out[row]=sum;
}

__global__ void mmv_iq4_kernel(const uint8_t *vp, const float *xv, float *out, int K, int row_bytes) {
    int row=blockIdx.x, tid=threadIdx.x, il=tid/8, ib=tid%8;
    const uint8_t *rp=vp+(int64_t)row*row_bytes;
    float sum=0.f;
    for (int kb=0; kb<K/256; ++kb) {
        const block_iq4_xs &q=((const block_iq4_xs*)rp)[kb];
        const uint8_t *q4=q.qs+16*ib+4*il;
        int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);
        float d=h2f(q.d)*(sc-32); int base=kb*256+32*ib+4*il;
        #pragma unroll
        for(int j=0;j<4;j++) {
            sum += d*kvalues_iq4nl[q4[j]&15]*xv[base+j];
            sum += d*kvalues_iq4nl[q4[j]>>4]*xv[base+j+16];
        }
    }
    sum=warp_sum(sum); if(tid==0) out[row]=sum;
}

torch::Tensor mmv_iq3_xxs_cuda(torch::Tensor p,torch::Tensor x) {
    auto y=torch::empty({p.size(0)},x.options()); int K=x.numel();
    mmv_iq3_kernel<<<p.size(0),32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),K,p.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
torch::Tensor mmv_iq4_xs_cuda(torch::Tensor p,torch::Tensor x) {
    auto y=torch::empty({p.size(0)},x.options()); int K=x.numel();
    mmv_iq4_kernel<<<p.size(0),32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),K,p.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

__global__ void mmq_iq3_kernel(const uint8_t *vp, const float *xv, float *out, int K, int M, int row_bytes) {
    int row=blockIdx.x, tok=blockIdx.y, tid=threadIdx.x, il=tid/8, ib=tid%8;
    const uint8_t *rp=vp+(int64_t)row*row_bytes; const float *xx=xv+(int64_t)tok*K; float sum=0.f;
    for(int kb=0;kb<K/256;++kb){const block_iq3_xxs&q=((const block_iq3_xxs*)rp)[kb];const uint8_t*q3=q.qs+8*ib;const uint16_t*gas=(const uint16_t*)(q.qs+64)+2*ib;const uint8_t*g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);const uint8_t*g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);float d=h2f(q.d)*(0.5f+(aux>>28))*0.5f;uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];int base=kb*256+32*ib+8*il;
        #pragma unroll
        for(int j=0;j<4;j++){sum+=d*g1[j]*((signs&(1u<<j))?-1.f:1.f)*xx[base+j];sum+=d*g2[j]*((signs&(1u<<(j+4)))?-1.f:1.f)*xx[base+j+4];}}
    sum=warp_sum(sum);if(tid==0)out[(int64_t)tok*M+row]=sum;
}
__global__ void mmq_iq4_kernel(const uint8_t *vp, const float *xv, float *out, int K, int M, int row_bytes) {
    int row=blockIdx.x,tok=blockIdx.y,tid=threadIdx.x,il=tid/8,ib=tid%8;const uint8_t*rp=vp+(int64_t)row*row_bytes;const float*xx=xv+(int64_t)tok*K;float sum=0.f;
    for(int kb=0;kb<K/256;++kb){const block_iq4_xs&q=((const block_iq4_xs*)rp)[kb];const uint8_t*q4=q.qs+16*ib+4*il;int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);float d=h2f(q.d)*(sc-32);int base=kb*256+32*ib+4*il;
        #pragma unroll
        for(int j=0;j<4;j++){sum+=d*kvalues_iq4nl[q4[j]&15]*xx[base+j];sum+=d*kvalues_iq4nl[q4[j]>>4]*xx[base+j+16];}}
    sum=warp_sum(sum);if(tid==0)out[(int64_t)tok*M+row]=sum;
}

// Decode Q=1..3 MMVQ-style: smem-cached x/h, multi-row CTAs (llama.cpp-inspired).
// v2 gu: one CTA per selected expert-task; all L rows share one load of x[D].
// Requires L<=1024, D multiple of 256. Typical: B=1,K=8,L=256,D=6144.
// v3: one warp per (task,row); grid=N*L, block=32. High occupancy, x from global (mmv style).
// gu t1 smem x cache
// v2 down: CTA tile of output rows; load all topk hiddens to smem once.
__global__ void moe_decode_down_iq4_kernel(const uint8_t* dp,const float* h,const int64_t* ei,
 const float* ew,float* out,int D,int L,int topk,int B,int rb){
 extern __shared__ float smem[];
 // layout: sh[topk * L] then se[topk] as int reinterpret — use float buffer for h only
 float* sh=smem; // topk * L  (for this batch item)
 int b=blockIdx.y; if(b>=B) return;
 int nwarps=blockDim.x>>5; int lane=threadIdx.x&31; int warp=threadIdx.x>>5;
 int il=lane/8, ib=lane%8;
 // coop load hiddens for this token's topk experts
 int hn=topk*L;
 for(int i=threadIdx.x;i<hn;i+=blockDim.x) sh[i]=h[(int64_t)b*topk*L+i];
 __syncthreads();
 int row_tile=blockIdx.x; // each CTA handles nwarps consecutive rows? use grid.x = ceil(D/nwarps)
 for(int r=0;r<nwarps;r++){
  int row=row_tile*nwarps+r; if(row>=D) continue;
  // only warp r computes row (or all warps different rows)
 }
 // rewrite: warp w computes row = blockIdx.x * nwarps + w
 int row=blockIdx.x*nwarps+warp;
 if(row<D){
  float total=0.f;
  for(int k=0;k<topk;k++){
   int task=b*topk+k; int e=(int)ei[task];
   const uint8_t* rp=dp+((int64_t)e*D+row)*rb;
   const float* xx=sh+(int64_t)k*L; float sum=0.f;
   for(int kb=0;kb<L/256;++kb){
    const block_iq4_xs&q=((const block_iq4_xs*)rp)[kb]; const uint8_t*q4=q.qs+16*ib+4*il;
    int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);
    float d=h2f(q.d)*(sc-32); int off=kb*256+32*ib+4*il;
    uint32_t q4w=*reinterpret_cast<const uint32_t*>(q4);
    #pragma unroll
    for(int j=0;j<4;j++){uint32_t qv=(q4w>>(8*j))&255u;sum+=d*kvalues_iq4nl[qv&15]*xx[off+j];sum+=d*kvalues_iq4nl[qv>>4]*xx[off+j+16];}
   }
   sum=warp_sum(sum); if(lane==0) total+=sum*ew[task];
  }
  if(lane==0) out[(int64_t)b*D+row]=total;
 }
}

__global__ void moe_decode_down_iq4_cache_kernel(const uint8_t* __restrict__ dp,const float* __restrict__ h,const int8_t* __restrict__ hq,const float* __restrict__ hds,const int64_t* __restrict__ ei,
 const float* __restrict__ ew,float* __restrict__ out,const uint8_t* __restrict__ q8cache,const int* __restrict__ cache_map,int D,int L,int topk,int B,int rb){
 extern __shared__ float smem[];
 float* sh=smem; // topk * L (fp32 h, for dcache branch)
 float* shds=smem+topk*L;          // topk * L/32 scales
 int8_t* shq=(int8_t*)(shds+topk*(L>>5)); // topk * L int8
 int b=blockIdx.y; if(b>=B) return;
 int nwarps=blockDim.x>>5; int lane=threadIdx.x&31; int warp=threadIdx.x>>5;
 int il=lane/8, ib=lane%8;
 int hn=topk*L;
 for(int i2=threadIdx.x;i2<hn;i2+=blockDim.x){ sh[i2]=h[(int64_t)b*topk*L+i2]; shq[i2]=hq[(int64_t)b*topk*L+i2]; }
 for(int i2=threadIdx.x;i2<(hn>>5);i2+=blockDim.x) shds[i2]=hds[((int64_t)b*topk*L>>5)+i2];
 __syncthreads();
 // dual-row: each warp computes two consecutive D-rows
 int row = (blockIdx.x*nwarps + warp)*2;
 if(row>=D) return;
 const bool has1 = (row+1)<D;
 float total0=0.f, total1=0.f;
 for(int k=0;k<topk;k++){
  int task=b*topk+k; int e=(int)ei[task];
  const float* xx=sh+(int64_t)k*L; const int8_t* xq8=shq+k*L; const float* xds8=shds+k*(L>>5); float sum0=0.f,sum1=0.f;
  int slot=cache_map[e];
  if(slot>=0){
   const int q8rb=L+(L>>5)*(int)sizeof(float);
   const uint8_t* cp0=q8cache+((int64_t)slot*D+row)*q8rb;
   const uint8_t* cp1=has1? q8cache+((int64_t)slot*D+row+1)*q8rb : cp0;
   auto q8_acc=[&](const uint8_t* cp,float& sum){
     const int8_t* qw=(const int8_t*)cp; const float* qs=(const float*)(cp+L);
     #pragma unroll
     for(int kb=0;kb<L/256;++kb){
       int off=kb*256+32*ib+4*il; int ds=off>>5;
       int hlo=*reinterpret_cast<const int*>(xq8+off);
       int hhi=*reinterpret_cast<const int*>(xq8+off+16);
       int wlo=*reinterpret_cast<const int*>(qw+off);
       int whi=*reinterpret_cast<const int*>(qw+off+16);
       sum += qs[ds]*xds8[ds]*__dp4a(wlo,hlo,__dp4a(whi,hhi,0));
     }
   };
   q8_acc(cp0,sum0); if(has1) q8_acc(cp1,sum1);
  } else {
   const uint8_t* rp0=dp+((int64_t)e*D+row)*rb;
   const uint8_t* rp1=has1? dp+((int64_t)e*D+row+1)*rb : rp0;
   for(int kb=0;kb<L/256;++kb){
     auto iq4_acc=[&](const uint8_t* rp,float& sum){
       const block_iq4_xs&q=((const block_iq4_xs*)rp)[kb]; const uint8_t*q4=q.qs+16*ib+4*il;
       int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);
       float d=h2f(q.d)*(sc-32); int off=kb*256+32*ib+4*il;
       uint32_t q4w=*reinterpret_cast<const uint32_t*>(q4);
       // prmt LUT: kvalues_iq4nl packed into 4 words; entries 0-7 in lo pool, 8-15 in hi pool
       const uint32_t KV0=0xBFAD9881u, KV1=0xF6EADDCFu, KV2=0x26190D01u, KV3=0x71594535u;
       uint32_t lo_idx = q4w & 0x0F0F0F0Fu;
       uint32_t hi_idx = (q4w>>4) & 0x0F0F0F0Fu;
       auto lut4=[&](uint32_t idx)->uint32_t{
         uint32_t t=(idx|(idx>>4))&0x00FF00FFu;
         uint32_t s=(t&0xFFu)|((t>>8)&0xFF00u);
         uint32_t a2=__byte_perm(KV0,KV1,s&0x7777u);
         uint32_t b2=__byte_perm(KV2,KV3,s&0x7777u);
         uint32_t m=((idx>>3)&0x01010101u)*0xFFu;
         return (a2&~m)|(b2&m);
       };
       uint32_t vlo=lut4(lo_idx), vhi=lut4(hi_idx);
       int hlo=*reinterpret_cast<const int*>(xq8+off);
       int hhi=*reinterpret_cast<const int*>(xq8+off+16);
       int isum=__dp4a((int)vlo,hlo,__dp4a((int)vhi,hhi,0));
       sum += d*xds8[off>>5]*isum;
     };
     iq4_acc(rp0,sum0);
     if(has1) iq4_acc(rp1,sum1);
   }
  }
  sum0=warp_sum(sum0); if(lane==0) total0+=sum0*ew[task];
  if(has1){ sum1=warp_sum(sum1); if(lane==0) total1+=sum1*ew[task]; }
 }
 if(lane==0) out[(int64_t)b*D+row]=total0;
 if(has1 && lane==0) out[(int64_t)b*D+row+1]=total1;
}

// Multi-warp gu: each block has NWARPS warps, each warp one output row.
// Greatly reduces tiny-block launch count vs <<<N*L,32>>>.
// fast v1: dual-row per warp + float4 x loads (same numerics as production)


// ---- dp4a gu: quantize x to q8 (per-32 scale), integer dp4a vs iq3_xxs grid ----
__global__ void down_cache_moe_x_q8_kernel(const float* x,int8_t* xq,float* xds,int64_t n){
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  int lane=threadIdx.x&31;
  float v=x[i];
  float a=fabsf(v);
  #pragma unroll
  for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,o));
  float scale=a>0.f?a/127.f:1.f;
  xq[i]=(int8_t)__float2int_rn(v/scale);
  if(lane==0) xds[i/32]=scale;
}

__global__ void moe_x_q8_half_kernel(const half* x,int8_t* xq,float* xds,int64_t n){
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  int lane=threadIdx.x&31;
  float v=__half2float(x[i]);
  float a=fabsf(v);
  #pragma unroll
  for(int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,o));
  float scale=a>0.f?a/127.f:1.f;
  xq[i]=(int8_t)__float2int_rn(v/scale);
  if(lane==0) xds[i/32]=scale;
}

template<int NWARPS>
__global__ void down_cache_moe_decode_gu_iq3_dp4a_kernel(const uint8_t* __restrict__ gp,const uint8_t* __restrict__ up,
 const int8_t* __restrict__ xq,const float* __restrict__ xds,
 const int64_t* __restrict__ ei,float* __restrict__ hidden,int D,int L,int topk,int N,int rb){
  extern __shared__ uint8_t smem_bw[];
  int task = blockIdx.x;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int rowi = warp >> 1;      // 4 rows per CTA
  int half = warp & 1;       // 2 warps per row
  int row = blockIdx.y * 4 + rowi;
  bool alive = (task < N) && (row < L);
  int hb = rb >> 1;                       // 1176
  int padh = (hb + 15) & ~15;             // 1184
  uint8_t* smg = smem_bw + (size_t)warp * 2 * padh;
  uint8_t* smu = smg + padh;
  float sg = 0.f, su = 0.f;
  if(alive){
    int e = (int)ei[task];
    int b = task / topk;
    const int8_t* xb = xq + (int64_t)b * D;
    const float* xs = xds + (int64_t)b * (D/32);
    const uint8_t* rg = gp + ((int64_t)e * L + row) * rb + (int64_t)half * hb;
    const uint8_t* ru = up + ((int64_t)e * L + row) * rb + (int64_t)half * hb;
    { // cooperative 8B staging of this half-row (hb=1176=147*8)
      const uint2* g2 = (const uint2*)rg; const uint2* u2 = (const uint2*)ru;
      uint2* s2 = (uint2*)smg; uint2* t2 = (uint2*)smu;
      int n2 = hb >> 3;
      for(int t = lane; t < n2; t += 32){ s2[t] = __ldg(g2 + t); t2[t] = __ldg(u2 + t); }
      __syncwarp();
    }
    int il = lane / 8, ib = lane % 8;
    int kb0 = half * (D/512);
    #pragma unroll 2
    for(int k = 0; k < D/512; ++k){
      int kb = kb0 + k;
      const block_iq3_xxs& qg = ((const block_iq3_xxs*)smg)[k];
      const block_iq3_xxs& qu = ((const block_iq3_xxs*)smu)[k];
      const uint8_t* q3g = qg.qs + 8*ib; const uint16_t* gasg = (const uint16_t*)(qg.qs + 64) + 2*ib;
      const uint8_t* q3u = qu.qs + 8*ib; const uint16_t* gasu = (const uint16_t*)(qu.qs + 64) + 2*ib;
      uint16_t q3gp=*reinterpret_cast<const uint16_t*>(q3g+2*il);
      uint16_t q3up=*reinterpret_cast<const uint16_t*>(q3u+2*il);
      uint32_t ag = (uint32_t)gasg[0] | ((uint32_t)gasg[1] << 16);
      uint32_t au = (uint32_t)gasu[0] | ((uint32_t)gasu[1] << 16);
      const uint32_t* sgn_g = (const uint32_t*)(ksigns64 + ((ag >> (7*il)) & 127));
      const uint32_t* sgn_u = (const uint32_t*)(ksigns64 + ((au >> (7*il)) & 127));
      int g1 = __vsub4(iq3xxs_grid[q3gp&255u] ^ sgn_g[0], sgn_g[0]);
      int g2 = __vsub4(iq3xxs_grid[q3gp>>8]   ^ sgn_g[1], sgn_g[1]);
      int u1 = __vsub4(iq3xxs_grid[q3up&255u] ^ sgn_u[0], sgn_u[0]);
      int u2 = __vsub4(iq3xxs_grid[q3up>>8]   ^ sgn_u[1], sgn_u[1]);
      int off = kb*256 + 32*ib + 8*il;
      int x1 = *(const int*)(xb + off);
      int x2 = *(const int*)(xb + off + 4);
      int dig = __dp4a(g1, x1, __dp4a(g2, x2, 0));
      int diu = __dp4a(u1, x1, __dp4a(u2, x2, 0));
      float xsc = xs[kb*8 + ib];
      sg += h2f(qg.d) * (0.5f + (ag >> 28)) * 0.5f * xsc * (float)dig;
      su += h2f(qu.d) * (0.5f + (au >> 28)) * 0.5f * xsc * (float)diu;
    }
    sg = warp_sum(sg); su = warp_sum(su);
  }
  __shared__ float2 part_bw[8];
  if(lane == 0) part_bw[warp] = make_float2(sg, su);
  __syncthreads();
  if(alive && half == 0 && lane == 0){
    float g = part_bw[warp].x + part_bw[warp+1].x;
    float u = part_bw[warp].y + part_bw[warp+1].y;
    hidden[(int64_t)task * L + row] = (g / (1.f + expf(-g))) * u;
  }
}

__global__ void add_f32_f16_to_f16_kernel(const float* a, const half* b, half* y, int64_t n) {
 int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
 if (i < n) y[i] = __float2half_rn(a[i] + __half2float(b[i]));
}

torch::Tensor add_f32_f16_to_f16_cuda(torch::Tensor a, torch::Tensor b) {
 TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.scalar_type()==torch::kFloat32 &&
             b.scalar_type()==torch::kFloat16 && a.sizes()==b.sizes() &&
             a.is_contiguous() && b.is_contiguous() && a.device()==b.device(),
             "add_f32_f16_to_f16 expects contiguous same-device fp32/fp16 tensors");
 auto y=torch::empty(a.sizes(),a.options().dtype(torch::kFloat16));
 int64_t n=a.numel();
 add_f32_f16_to_f16_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
     a.data_ptr<float>(),reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
     reinterpret_cast<half*>(y.data_ptr<at::Half>()),n);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return y;
}

#ifndef LJQ_SPECIAL_EXT_NO_IQ_FUSED
__attribute__((visibility("hidden"))) torch::Tensor moe_decode_iq_fused_out_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x,torch::Tensor ei,torch::Tensor ew,torch::Tensor h,torch::Tensor y){
 int B=x.size(0),D=x.size(1),L=g.size(1),K=ei.size(1),N=B*K;
 TORCH_CHECK(D%256==0 && L%256==0 && L<=4096, "decode fused shape");
 auto st=at::cuda::getCurrentCUDAStream();
 const int NW=8;
 dim3 grid(N, (L+NW-1)/NW); // one row per warp (fixed: was dual-row grid with single-row kernel)
 dim3 block(NW*32);
 auto xq=at::empty({B,D},x.options().dtype(torch::kChar));
 auto xds=at::empty({B,D/32},x.options());
 down_cache_moe_x_q8_kernel<<<((int64_t)B*D+255)/256,256,0,st>>>(x.data_ptr<float>(),(int8_t*)xq.data_ptr(),xds.data_ptr<float>(),(int64_t)B*D);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 down_cache_moe_decode_gu_iq3_dp4a_kernel<8><<<dim3(N,(L+3)/4),256,(size_t)8*2*((((int)g.size(2)>>1)+15)&~15),st>>>(g.data_ptr<uint8_t>(),u.data_ptr<uint8_t>(),(const int8_t*)xq.data_ptr(),xds.data_ptr<float>(),ei.data_ptr<int64_t>(),h.data_ptr<float>(),D,L,K,N,(int)g.size(2));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 {
  int nwarps=8; int threads=nwarps*32; int tiles=(D+nwarps-1)/nwarps;
  size_t shmem=(size_t)K*L*sizeof(float);
  moe_decode_down_iq4_kernel<<<dim3(tiles,B),threads,shmem,st>>>(d.data_ptr<uint8_t>(),h.data_ptr<float>(),ei.data_ptr<int64_t>(),ew.data_ptr<float>(),y.data_ptr<float>(),D,L,K,B,d.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return y;
}


torch::Tensor moe_decode_iq_down_cache_out_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x,torch::Tensor ei,torch::Tensor ew,torch::Tensor h,torch::Tensor y,torch::Tensor dcache,torch::Tensor cache_map){
 int B=x.size(0),D=x.size(1),L=g.size(1),K=ei.size(1),N=B*K;
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat16 && x.is_contiguous(), "decode x must be contiguous fp16 CUDA");
 TORCH_CHECK(h.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32, "decode workspaces must be fp32");
 TORCH_CHECK(D%256==0 && L%256==0 && L<=4096, "decode fused shape");
 auto st=at::cuda::getCurrentCUDAStream();
 const int NW=8;
 dim3 grid(N, (L+NW-1)/NW); // one row per warp (fixed: was dual-row grid with single-row kernel)
 dim3 block(NW*32);
 auto xq=at::empty({B,D},x.options().dtype(torch::kChar));
 auto xds=at::empty({B,D/32},x.options().dtype(torch::kFloat));
 moe_x_q8_half_kernel<<<((int64_t)B*D+255)/256,256,0,st>>>((const half*)x.data_ptr<at::Half>(),(int8_t*)xq.data_ptr(),xds.data_ptr<float>(),(int64_t)B*D);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 down_cache_moe_decode_gu_iq3_dp4a_kernel<8><<<dim3(N,(L+3)/4),256,(size_t)8*2*((((int)g.size(2)>>1)+15)&~15),st>>>(g.data_ptr<uint8_t>(),u.data_ptr<uint8_t>(),(const int8_t*)xq.data_ptr(),xds.data_ptr<float>(),ei.data_ptr<int64_t>(),h.data_ptr<float>(),D,L,K,N,(int)g.size(2));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 {
  int nwarps=8; int threads=nwarps*32; int tiles=(D+nwarps*2-1)/(nwarps*2); // dual-row down, dp4a h
  auto hqt=at::empty({(int64_t)B*K*L},x.options().dtype(torch::kChar));
  auto hdst=at::empty({(int64_t)B*K*L/32},x.options().dtype(torch::kFloat));
  down_cache_moe_x_q8_kernel<<<((int64_t)B*K*L+255)/256,256,0,st>>>(h.data_ptr<float>(),(int8_t*)hqt.data_ptr(),hdst.data_ptr<float>(),(int64_t)B*K*L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  size_t shmem=(size_t)K*L*sizeof(float)+(size_t)K*(L/32)*sizeof(float)+(size_t)K*L;
  moe_decode_down_iq4_cache_kernel<<<dim3(tiles,B),threads,shmem,st>>>(d.data_ptr<uint8_t>(),h.data_ptr<float>(),(const int8_t*)hqt.data_ptr(),hdst.data_ptr<float>(),ei.data_ptr<int64_t>(),ew.data_ptr<float>(),y.data_ptr<float>(),(const uint8_t*)dcache.data_ptr(),cache_map.data_ptr<int>(),D,L,K,B,d.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return y;
}


torch::Tensor moe_decode_iq_fused_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x,torch::Tensor ei,torch::Tensor ew){
 int B=x.size(0),D=x.size(1),L=g.size(1),K=ei.size(1);
 auto h=torch::empty({B*K,L},x.options()); auto y=torch::empty({B,D},x.options());
 return moe_decode_iq_fused_out_cuda(g,u,d,x,ei,ew,h,y);
}
#endif  // LJQ_SPECIAL_EXT_NO_IQ_FUSED


torch::Tensor mmq_iq3_xxs_cuda(torch::Tensor p,torch::Tensor x){auto y=torch::empty({x.size(0),p.size(0)},x.options());dim3 g(p.size(0),x.size(0));mmq_iq3_kernel<<<g,32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),x.size(1),p.size(0),p.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor mmq_iq4_xs_cuda(torch::Tensor p,torch::Tensor x){auto y=torch::empty({x.size(0),p.size(0)},x.options());dim3 g(p.size(0),x.size(0));mmq_iq4_kernel<<<g,32,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),x.size(1),p.size(0),p.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}


// v2: one CTA shares each decoded 256-weight tile across 8 token warps.
__global__ void mmq_iq3_tile_kernel(const uint8_t *vp,const float *xv,float *out,int K,int M,int T,int row_bytes){
    __shared__ half sw[256]; int row=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5, tok=blockIdx.y*8+warp;
    const uint8_t *rp=vp+(int64_t)row*row_bytes; float sum=0.f;
    for(int kb=0;kb<K/256;++kb){
        if(threadIdx.x<32){int il=threadIdx.x/8,ib=threadIdx.x%8;const block_iq3_xxs&q=((const block_iq3_xxs*)rp)[kb];const uint8_t*q3=q.qs+8*ib;const uint16_t*gas=(const uint16_t*)(q.qs+64)+2*ib;const uint8_t*g1=(const uint8_t*)(iq3xxs_grid+q3[2*il]);const uint8_t*g2=(const uint8_t*)(iq3xxs_grid+q3[2*il+1]);uint32_t aux=(uint32_t)gas[0]|((uint32_t)gas[1]<<16);float d=h2f(q.d)*(0.5f+(aux>>28))*0.5f;uint8_t signs=ksigns_iq2xs[(aux>>(7*il))&127];int b=32*ib+8*il;
            #pragma unroll
            for(int j=0;j<4;j++){sw[b+j]=__float2half(d*g1[j]*((signs&(1u<<j))?-1.f:1.f));sw[b+j+4]=__float2half(d*g2[j]*((signs&(1u<<(j+4)))?-1.f:1.f));}}
        __syncthreads(); if(tok<T){const float*xx=xv+(int64_t)tok*K+kb*256;
            #pragma unroll
            for(int j=lane;j<256;j+=32)sum+=__half2float(sw[j])*xx[j];}
        __syncthreads();}
    sum=warp_sum(sum);if(tok<T&&lane==0)out[(int64_t)tok*M+row]=sum;
}
__global__ void mmq_iq4_tile_kernel(const uint8_t *vp,const float *xv,float *out,int K,int M,int T,int row_bytes){
    __shared__ half sw[256]; int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5,tok=blockIdx.y*8+warp;
    const uint8_t *rp=vp+(int64_t)row*row_bytes;float sum=0.f;
    for(int kb=0;kb<K/256;++kb){
        if(threadIdx.x<32){int il=threadIdx.x/8,ib=threadIdx.x%8;const block_iq4_xs&q=((const block_iq4_xs*)rp)[kb];const uint8_t*q4=q.qs+16*ib+4*il;int sc=((q.scales_l[ib/2]>>(4*(ib%2)))&15)|(((q.scales_h>>(2*ib))&3)<<4);float d=h2f(q.d)*(sc-32);int b=32*ib+4*il;
            #pragma unroll
            for(int j=0;j<4;j++){sw[b+j]=__float2half(d*kvalues_iq4nl[q4[j]&15]);sw[b+j+16]=__float2half(d*kvalues_iq4nl[q4[j]>>4]);}}
        __syncthreads();if(tok<T){const float*xx=xv+(int64_t)tok*K+kb*256;
            #pragma unroll
            for(int j=lane;j<256;j+=32)sum+=__half2float(sw[j])*xx[j];}
        __syncthreads();}
    sum=warp_sum(sum);if(tok<T&&lane==0)out[(int64_t)tok*M+row]=sum;
}
torch::Tensor mmq_iq3_tile_cuda(torch::Tensor p,torch::Tensor x){auto y=torch::empty({x.size(0),p.size(0)},x.options());dim3 g(p.size(0),(x.size(0)+7)/8);mmq_iq3_tile_kernel<<<g,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),x.size(1),p.size(0),x.size(0),p.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
torch::Tensor mmq_iq4_tile_cuda(torch::Tensor p,torch::Tensor x){auto y=torch::empty({x.size(0),p.size(0)},x.options());dim3 g(p.size(0),(x.size(0)+7)/8);mmq_iq4_tile_kernel<<<g,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<uint8_t>(),x.data_ptr<float>(),y.data_ptr<float>(),x.size(1),p.size(0),x.size(0),p.size(1));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}


// ===== v3: tensor-core fused down GEMM (IQ4_XS), no FP16 weight buffer =====
// out[n,M] = x[n,K] @ Wd[M,K]^T ;  K=256 (1 IQ4 block per output row), Wd packed [M,136]
// CTA = [16 tokens x 64 output cols]; 4 WMMA m16n16k16 tiles along N, 16 K-steps, fp32 accum.
#include <mma.h>
using namespace nvcuda::wmma;
__global__ void down_iq4_tc_kernel(
    const uint8_t* __restrict__ wp, const __half* __restrict__ xv,
    float* __restrict__ out, int n, int M, int K, int row_bytes)
{
    int tok0=blockIdx.x*16, col0=blockIdx.y*64, tid=threadIdx.x;
    int warp=tid>>5, lane=tid&31;
    __shared__ __half Bs[256*64];   // [K=256][N=64] row-major (k-stride 64)
    __shared__ __half As[16*256];   // [M=16][K=256] row-major (k-stride 256)
    // decode 64 rows of IQ4_XS into Bs[k][n]; 8 warps x 8 rows, each warp serial over 8 rows
    {
        int rr0=warp*8; const uint8_t* rp=wp+(int64_t)(col0+rr0)*row_bytes;
        int il=lane/8, ib=lane%8;
        for(int rr=0; rr<8; ++rr, rp+=row_bytes){
            const block_iq4_xs* q=(const block_iq4_xs*)rp;   // K=256 -> exactly 1 block/row
            const uint8_t* q4=q->qs+16*ib+4*il;
            int sc=((q->scales_l[ib/2]>>(4*(ib%2)))&15)|(((q->scales_h>>(2*ib))&3)<<4);
            float d=__half2float(q->d)*(sc-32);
            int b=32*ib+4*il, ncol=rr0+rr;
            #pragma unroll
            for(int j=0;j<4;j++){
                Bs[(b+j)*64+ncol]=__float2half(d*kvalues_iq4nl[q4[j]&15]);
                Bs[(b+16+j)*64+ncol]=__float2half(d*kvalues_iq4nl[q4[j]>>4]);
            }
        }
    }
    // load A tile [16,256] from x (guard tok<n)
    for(int i=tid;i<16*256;i+=256){
        int m=i/256,k=i%256,tok=tok0+m;
        As[i]=(tok<n)?xv[tok*K+k]:__float2half(0.f);
    }
    __syncthreads();
    // WMMA: 4 warps x 1 [16,16] tile, 16 K-steps
    if(warp<4){
        fragment<matrix_a,16,16,16,__half,nvcuda::wmma::row_major> a_frag;
        fragment<matrix_b,16,16,16,__half,nvcuda::wmma::row_major> b_frag;
        fragment<accumulator,16,16,16,float> c_frag;
        fill_fragment(c_frag,0.f);
        int noff=warp*16;
        #pragma unroll
        for(int kk=0;kk<16;kk++){
            load_matrix_sync(a_frag,As+kk*16,256);
            load_matrix_sync(b_frag,Bs+noff+kk*16*64,64);
            mma_sync(c_frag,a_frag,b_frag,c_frag);
        }
        store_matrix_sync(out+(int64_t)tok0*M+col0+noff, c_frag, M, nvcuda::wmma::mem_row_major);
    }
}
torch::Tensor down_iq4_tc_cuda(torch::Tensor p,torch::Tensor x){
    TORCH_CHECK(p.is_cuda()&&p.scalar_type()==torch::kUInt8&&p.is_contiguous(),"down: p must be uint8 contiguous cuda");
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kHalf&&x.is_contiguous(),"down: x must be fp16 contiguous cuda");
    int n=x.size(0),K=x.size(1),M=p.size(0);
    TORCH_CHECK(K==256,"down TC: K must be 256 (1 IQ4 block/row)");
    TORCH_CHECK(n%16==0,"down TC: n must be multiple of 16");
    TORCH_CHECK(M%64==0,"down TC: M must be multiple of 64");
    auto out=torch::empty({n,M},x.options().dtype(torch::kFloat32));
    dim3 g(n/16,M/64);
    down_iq4_tc_kernel<<<g,256,0,at::cuda::getCurrentCUDAStream()>>>(
        p.data_ptr<uint8_t>(),(const __half*)x.data_ptr<at::Half>(),
        out.data_ptr<float>(),n,M,K,p.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

// ===== decode Q1-3 shared expert: direct Q8_0, two launches =====
__device__ __forceinline__ float q8_warp_dot_decode(const half *x,const unsigned char *row,int K,int lane){
  float a=0.f;
  for(int b=0;b<(K>>5);++b){
    const unsigned char *p=row+(size_t)b*34;
    float scale=__half2float(*reinterpret_cast<const half*>(p));
    float w=scale*(float)reinterpret_cast<const signed char*>(p+2)[lane];
    a=fmaf(__half2float(x[(b<<5)+lane]),w,a);
  }
  for(int d=16;d;d>>=1) a+=__shfl_down_sync(0xffffffff,a,d);
  return a;
}
// Forward decl: defined later in this TU (T=1 smem path used by production Special).
void down_cache_shared_decode_q8_inplace_cuda(torch::Tensor g, torch::Tensor u, torch::Tensor d,
                                      torch::Tensor x, torch::Tensor h, torch::Tensor y);

// Forward decl: defined later in this TU (T=1 smem path used by production Special).
void down_cache_shared_decode_q8_inplace_cuda(torch::Tensor g, torch::Tensor u, torch::Tensor d,
                                      torch::Tensor x, torch::Tensor h, torch::Tensor y);

torch::Tensor shared_decode_q8_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x){
  // 2026-07-22: multi-Q single-shot smem path (weight reused across Q). Returns fp16 [Q,N].
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16&&x.dim()==2,"x fp16 CUDA [Q,D]");
  TORCH_CHECK(g.scalar_type()==torch::kUInt8&&u.scalar_type()==torch::kUInt8&&d.scalar_type()==torch::kUInt8,"Q8 packed uint8");
  TORCH_CHECK(x.is_contiguous(),"shared_decode_q8 x must be contiguous");
  int Q=x.size(0),K=x.size(1),H=g.size(0),N=d.size(0);
  TORCH_CHECK(Q>=1&&Q<128&&K%32==0&&H%32==0,"small shared expects T=1..127");
  auto h=torch::empty({Q,H},x.options());
  auto yf=torch::empty({Q,N},x.options().dtype(torch::kFloat32));
  down_cache_shared_decode_q8_inplace_cuda(g,u,d,x,h,yf);
  return yf.to(torch::kFloat16);
}

// ===== T=1 route v4: at::mm logits + compact select (no torch.topk) =====
__global__ void moe_route_select_kernel(const float* __restrict__ logits,
                                        const float* __restrict__ bias,
                                        int64_t* __restrict__ ei,
                                        float* __restrict__ ew,
                                        int E, int n_group, int group_size, int top_g, int top_k, float scale){
  __shared__ float probs[256];
  __shared__ float sel[256];
  int tid=threadIdx.x;
  for(int e=tid;e<E;e+=blockDim.x){
    float p=1.f/(1.f+expf(-logits[e]));
    probs[e]=p;
    sel[e]=p+(bias?bias[e]:0.f);
  }
  __syncthreads();
  if(tid==0){
    float gscore[8]; int gpick[8];
    for(int g=0;g<n_group;g++){
      float b1=-1e30f,b2=-1e30f; int base=g*group_size;
      for(int j=0;j<group_size;j++){
        float v=sel[base+j];
        if(v>b1){b2=b1;b1=v;} else if(v>b2){b2=v;}
      }
      gscore[g]=b1+b2;
    }
    for(int t=0;t<top_g;t++){
      float bv=-1e30f; int bi=-1;
      for(int g=0;g<n_group;g++){
        bool used=false; for(int k=0;k<t;k++) if(gpick[k]==g){used=true;break;}
        if(!used && gscore[g]>bv){bv=gscore[g]; bi=g;}
      }
      gpick[t]=bi;
    }
    float cand_s[128]; int cand_i[128]; int nc=top_g*group_size;
    for(int t=0;t<nc;t++){
      int g=gpick[t/group_size]; int j=t%group_size;
      cand_s[t]=sel[g*group_size+j]; cand_i[t]=g*group_size+j;
    }
    int top_idx[16];
    for(int t=0;t<top_k;t++){
      float bv=-1e30f; int bi=-1;
      for(int i=0;i<nc;i++){
        bool used=false; for(int k=0;k<t;k++) if(top_idx[k]==i){used=true;break;}
        if(!used && cand_s[i]>bv){bv=cand_s[i]; bi=i;}
      }
      top_idx[t]=bi;
    }
    float sum=0.f; float tmpw[16]; int tmpe[16];
    for(int t=0;t<top_k;t++){
      int e=cand_i[top_idx[t]]; tmpe[t]=e; tmpw[t]=probs[e]; sum+=tmpw[t];
    }
    float inv=(sum>6.103515625e-5f)? (scale/sum):0.f;
    for(int t=0;t<top_k;t++){ ei[t]=(int64_t)tmpe[t]; ew[t]=tmpw[t]*inv; }
  }
}

// logits_workspace: optional [E] float prealloc for CUDA graph
void moe_route_decode_t1_inplace_cuda(torch::Tensor x, torch::Tensor router, torch::Tensor bias,
                                      torch::Tensor ei, torch::Tensor ew,
                                      int64_t n_group, int64_t top_g, int64_t top_k, double scale){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.dim()==2 && x.size(0)==1, "x");
  TORCH_CHECK(router.is_cuda() && router.scalar_type()==torch::kFloat32 && router.dim()==2, "W");
  TORCH_CHECK(ei.is_cuda() && ew.is_cuda(), "ei/ew");
  TORCH_CHECK(ei.numel()>=top_k && ew.numel()>=top_k, "size");
  int D=(int)x.size(1), E=(int)router.size(0);
  TORCH_CHECK(router.size(1)==D, "D");
  int group_size=E/(int)n_group;
  int dev=x.device().index();
  auto st=at::cuda::getCurrentCUDAStream(dev);
  // cublas-backed GEMV via at::mm: [1,D]x[D,E] = [1,E]
  auto logits = at::mm(x, router.transpose(0,1)); // [1,E]
  const float* bp=(bias.defined()&&bias.numel()>0)?bias.data_ptr<float>():nullptr;
  moe_route_select_kernel<<<1,256,0,st>>>(logits.data_ptr<float>(), bp,
    ei.data_ptr<int64_t>(), ew.data_ptr<float>(),
    E,(int)n_group,group_size,(int)top_g,(int)top_k,(float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// select-only for measuring / when logits already computed
void moe_route_select_only_cuda(torch::Tensor logits, torch::Tensor bias,
                                torch::Tensor ei, torch::Tensor ew,
                                int64_t n_group, int64_t top_g, int64_t top_k, double scale){
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type()==torch::kFloat32, "logits");
  auto flat=logits.reshape({-1});
  int E=(int)flat.size(0);
  int group_size=E/(int)n_group;
  auto st=at::cuda::getCurrentCUDAStream(logits.device().index());
  const float* bp=(bias.defined()&&bias.numel()>0)?bias.data_ptr<float>():nullptr;
  moe_route_select_kernel<<<1,256,0,st>>>(flat.data_ptr<float>(), bp,
    ei.data_ptr<int64_t>(), ew.data_ptr<float>(),
    E,(int)n_group,group_size,(int)top_g,(int)top_k,(float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> moe_route_decode_t1_cuda(torch::Tensor x, torch::Tensor router, torch::Tensor bias,
                                                    int64_t n_group, int64_t top_g, int64_t top_k, double scale){
  auto ei=torch::empty({1,(long)top_k}, x.options().dtype(torch::kLong));
  auto ew=torch::empty({1,(long)top_k}, x.options());
  moe_route_decode_t1_inplace_cuda(x,router,bias,ei,ew,n_group,top_g,top_k,scale);
  return {ei,ew};
}



// ==== rms_norm T=1 fused ====
__global__ void rms_norm_t1_f2h_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                       float* __restrict__ yf, __half* __restrict__ yh,
                                       int D, float eps){
  __shared__ float red[32];
  int tid=threadIdx.x;
  float sum=0.f;
  for(int i=tid;i<D;i+=blockDim.x){ float v=x[i]; sum=fmaf(v,v,sum); }
  for(int off=16;off>0;off>>=1) sum+=__shfl_down_sync(0xffffffff,sum,off);
  if((tid&31)==0) red[tid>>5]=sum;
  __syncthreads();
  if(tid<32){
    float v=(tid<(blockDim.x>>5))?red[tid]:0.f;
    for(int off=16;off>0;off>>=1) v+=__shfl_down_sync(0xffffffff,v,off);
    if(tid==0) red[0]=v;
  }
  __syncthreads();
  float inv=rsqrtf(red[0]/(float)D + eps);
  for(int i=tid;i<D;i+=blockDim.x){
    float o=x[i]*inv*w[i];
    yf[i]=o;
    yh[i]=__float2half(o);
  }
}

void rms_norm_t1_inplace_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor yf, torch::Tensor yh, double eps){
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && yf.is_cuda() && yh.is_cuda());
  TORCH_CHECK(x.scalar_type()==torch::kFloat);
  TORCH_CHECK(w.scalar_type()==torch::kFloat);
  TORCH_CHECK(yf.scalar_type()==torch::kFloat);
  TORCH_CHECK(yh.scalar_type()==torch::kHalf);
  int D=(int)x.numel();
  auto st=at::cuda::getCurrentCUDAStream(x.device().index());
  rms_norm_t1_f2h_kernel<<<1,256,0,st>>>(x.data_ptr<float>(), w.data_ptr<float>(),
    yf.data_ptr<float>(), reinterpret_cast<__half*>(yh.data_ptr<at::Half>()), D, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ===== shared_q8 multi-Q / TN (legacy t1 path deleted) =====
// Runtime-T shared-expert matrix path. TOKEN_TILE is an internal M tile: the
// same two kernels cover every 2 <= T <= 32 without T-specific launchers.
// A warp owns one Q8 weight row and reuses each weight value across the token
// tile. grid.y adds matrix-row tiles instead of recursively launching Q<=4.
constexpr int SHARED_TN_TOKEN_TILE = 4;
constexpr int SHARED_TN_GU_ROWS = 8;

template<int TOKEN_TILE, int ROWS>
__attribute__((visibility("hidden"))) __global__ void shared_q8_gu_tn_kernel(
    const half* __restrict__ x,
    const unsigned char* __restrict__ g,
    const unsigned char* __restrict__ u,
    half* __restrict__ h,
    int T, int K, int H, int rb) {
  extern __shared__ unsigned char smem_raw[];
  half* xs = reinterpret_cast<half*>(smem_raw);
  const int t0 = (int)blockIdx.y * TOKEN_TILE;
  const int nt = min(TOKEN_TILE, T - t0);
  const int xelems = nt * K;
  for (int i = threadIdx.x; i < xelems; i += blockDim.x)
    xs[i] = x[(size_t)t0 * K + i];
  float* vals = reinterpret_cast<float*>(xs + xelems);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row_local = warp >> 1;
  const int kind = warp & 1;
  const int row = (int)blockIdx.x * ROWS + row_local;
  float acc[TOKEN_TILE];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) acc[t] = 0.f;
  if (row < H) {
    const unsigned char* wp = (kind == 0 ? g : u) + (size_t)row * rb;
    for (int b = 0; b < (K >> 5); ++b) {
      const unsigned char* p = wp + (size_t)b * 34;
      const float w = __half2float(*reinterpret_cast<const half*>(p)) *
          (float)reinterpret_cast<const signed char*>(p + 2)[lane];
      const int k = (b << 5) + lane;
#pragma unroll
      for (int t = 0; t < TOKEN_TILE; ++t)
        if (t < nt) acc[t] = fmaf(__half2float(xs[(size_t)t * K + k]), w, acc[t]);
    }
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t)
      if (t < nt)
        for (int delta = 16; delta; delta >>= 1)
          acc[t] += __shfl_down_sync(0xffffffff, acc[t], delta);
  }
  if (lane == 0 && row < H) {
    const int o = (row_local * 2 + kind) * TOKEN_TILE;
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t)
      if (t < nt) vals[o + t] = acc[t];
  }
  __syncthreads();
  if (kind == 0 && lane == 0 && row < H) {
    const int go = row_local * 2 * TOKEN_TILE;
    const int uo = go + TOKEN_TILE;
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t) {
      if (t < nt) {
        const float gv = vals[go + t], uv = vals[uo + t];
        const float sig = 1.f / (1.f + expf(-gv));
        h[(size_t)(t0 + t) * H + row] = __float2half_rn((gv * sig) * uv);
      }
    }
  }
}

template<int TOKEN_TILE>
__attribute__((visibility("hidden"))) __global__ void shared_q8_down_tn_kernel(
    const half* __restrict__ hact,
    const unsigned char* __restrict__ d,
    float* __restrict__ y,
    int T, int K, int N, int rb) {
  extern __shared__ half hs[];
  const int t0 = (int)blockIdx.y * TOKEN_TILE;
  const int nt = min(TOKEN_TILE, T - t0);
  const int helems = nt * K;
  for (int i = threadIdx.x; i < helems; i += blockDim.x)
    hs[i] = hact[(size_t)t0 * K + i];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int row = (int)blockIdx.x * warps + warp;
  if (row >= N) return;
  float acc[TOKEN_TILE];
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t) acc[t] = 0.f;
  const unsigned char* wp = d + (size_t)row * rb;
  for (int b = 0; b < (K >> 5); ++b) {
    const unsigned char* p = wp + (size_t)b * 34;
    const float w = __half2float(*reinterpret_cast<const half*>(p)) *
        (float)reinterpret_cast<const signed char*>(p + 2)[lane];
    const int k = (b << 5) + lane;
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t)
      if (t < nt) acc[t] = fmaf(__half2float(hs[(size_t)t * K + k]), w, acc[t]);
  }
#pragma unroll
  for (int t = 0; t < TOKEN_TILE; ++t)
    if (t < nt)
      for (int delta = 16; delta; delta >>= 1)
        acc[t] += __shfl_down_sync(0xffffffff, acc[t], delta);
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TOKEN_TILE; ++t)
      if (t < nt) y[(size_t)(t0 + t) * N + row] = acc[t];
  }
}

void down_cache_shared_decode_q8_inplace_cuda(torch::Tensor g, torch::Tensor u, torch::Tensor d,
                                   torch::Tensor x, torch::Tensor h, torch::Tensor y){
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kHalf, "x half");
  TORCH_CHECK(h.is_cuda() && h.scalar_type()==torch::kHalf, "h half");
  TORCH_CHECK(y.is_cuda() && y.scalar_type()==torch::kFloat, "y float");
  int K = x.dim()==2 ? (int)x.size(1) : (int)x.numel();
  int Q = x.dim()==2 ? (int)x.size(0) : 1;
  TORCH_CHECK(Q>=1 && Q<128, "small shared expects T=1..127");
  int H=(int)g.size(0), N=(int)d.size(0);
  if(h.dim()==1){
    TORCH_CHECK(Q==1 && (int)h.numel()>=H, "h buf");
  } else {
    TORCH_CHECK(h.size(0)>=Q && h.size(1)>=H, "h [Q,H]");
  }
  if(y.dim()==1){
    TORCH_CHECK(Q==1 && (int)y.numel()>=N, "y buf");
  } else {
    TORCH_CHECK(y.size(0)>=Q && y.size(1)>=N, "y [Q,N]");
  }
  TORCH_CHECK(Q<=32, "shared decode supports T<=32");
  int rbg=(int)g.size(1), rbd=(int)d.size(1);
  auto st=at::cuda::getCurrentCUDAStream(x.device().index());
  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  half* hp = reinterpret_cast<half*>(h.data_ptr<at::Half>());
  float* yp = y.data_ptr<float>();
  constexpr int TT=SHARED_TN_TOKEN_TILE, ROWS=SHARED_TN_GU_ROWS;
  dim3 gu_grid((H+ROWS-1)/ROWS, (Q+TT-1)/TT);
  constexpr int gu_threads=ROWS*2*32;
  const int alloc_t=std::min(Q,TT);
  const size_t gu_smem=(size_t)alloc_t*K*sizeof(half)
      +(size_t)ROWS*2*alloc_t*sizeof(float);
  if(alloc_t>=4) C10_CUDA_CHECK(cudaFuncSetAttribute(
      shared_q8_gu_tn_kernel<TT,ROWS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,(int)gu_smem));
  shared_q8_gu_tn_kernel<TT,ROWS><<<gu_grid,gu_threads,gu_smem,st>>>(
      xp,g.data_ptr<unsigned char>(),u.data_ptr<unsigned char>(),hp,
      Q,K,H,rbg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int dn_threads=256, dn_warps=dn_threads/32;
  dim3 dn_grid((N+dn_warps-1)/dn_warps, (Q+TT-1)/TT);
  const size_t dn_smem=(size_t)alloc_t*H*sizeof(half);
  shared_q8_down_tn_kernel<TT><<<dn_grid,dn_threads,dn_smem,st>>>(
      hp,d.data_ptr<unsigned char>(),yp,Q,H,N,rbd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

