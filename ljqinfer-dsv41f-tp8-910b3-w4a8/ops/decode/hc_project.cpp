// Decode-only fused HC prepare/projection. No device allocation.
// BF16 x[B*6,20480], FP32 weight[24,20480] -> z[B*6,24], invRMS[B*6].
// Two output columns per AIV, FP32 products and reductions. No weight conversion.
#include "kernel_operator.h"
#include <cstdint>
#include <cfloat>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
namespace {
constexpr int K=20480, N=24, T=5120;
bool aligned(const void* p){return p && !(uintptr_t(p)&31);}
bool overlaps(const void* a,uint64_t an,const void* b,uint64_t bn){
 auto x=uintptr_t(a),y=uintptr_t(b);return x<=y ? y-x<an : x-y<bn;
}
}
extern "C" __global__ __aicore__ void hc_project_kernel(
 GM_ADDR X,GM_ADDR W,GM_ADDR Z,GM_ADDR ST,uint32_t rows,float eps){
 const uint32_t jobs=rows*12, stride=GetBlockNum();
 if(GetBlockIdx()>=jobs)return;
 TPipe p;TBuf<TPosition::VECCALC> bh,bx,bw,bv,bt,br,bo;
 p.InitBuffer(bh,T*2);p.InitBuffer(bx,T*4);p.InitBuffer(bw,2*T*4);
 p.InitBuffer(bv,T*4);p.InitBuffer(bt,T*4);p.InitBuffer(br,128);p.InitBuffer(bo,32);
 auto h=bh.Get<bfloat16_t>();auto x=bx.Get<float>(),w=bw.Get<float>();
 auto v=bv.Get<float>(),tmp=bt.Get<float>(),red=br.Get<float>(),out=bo.Get<float>();
 GlobalTensor<bfloat16_t> gx;GlobalTensor<float> gw,gz,gs;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gw.SetGlobalBuffer((__gm__ float*)W);
 gz.SetGlobalBuffer((__gm__ float*)Z);gs.SetGlobalBuffer((__gm__ float*)ST);
 for(uint32_t job=GetBlockIdx();job<jobs;job+=stride){
  uint32_t row=job/12,col=(job%12)*2;float a=0.f,b=0.f,total=0.f;
  for(int tile=0;tile<4;++tile){
   DataCopy(h,gx[uint64_t(row)*K+tile*T],T);
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
   DataCopy(w,gw[uint64_t(col)*K+tile*T],
            DataCopyParams{2,uint16_t(T*4/32),uint16_t((K-T)*4/32),0});
   SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
   WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   Cast(x,h,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
   if(col==0){
    Mul(v,x,x,T);PipeBarrier<PIPE_V>();ReduceSum(red,v,tmp,T);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    total+=red.GetValue(0);PipeBarrier<PIPE_ALL>();
   }
   WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
   for(int c=0;c<2;++c){
    Mul(v,x,w[c*T],T);PipeBarrier<PIPE_V>();ReduceSum(red,v,tmp,T);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float val=red.GetValue(0);if(c==0)a+=val;else b+=val;
    PipeBarrier<PIPE_ALL>();
   }
  }
  out.SetValue(0,a);out.SetValue(1,b);
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopyPad(gz[uint64_t(row)*N+col],out,DataCopyExtParams{1,8,0,0,0});
  PipeBarrier<PIPE_ALL>();
  if(col==0){
   // Match the existing prepare kernel's four 5120-element reductions.
   Duplicate(red,total,int32_t(32));PipeBarrier<PIPE_V>();
   Muls(red,red,1.f/20480.f,int32_t(32));PipeBarrier<PIPE_V>();
   Adds(red,red,eps,int32_t(32));PipeBarrier<PIPE_V>();
   Sqrt(red,red,int32_t(32));Duplicate(tmp,1.f,int32_t(32));PipeBarrier<PIPE_V>();
   Div(red,tmp,red,int32_t(32));PipeBarrier<PIPE_ALL>();
   DataCopyPad(gs[row],red,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
  }
 }
}
extern "C" int dec_hc_project(void* stream,const void* x,const void* w,
 void* z,void* stats,uint32_t batch,float eps){
 if(!stream||batch<1||batch>4||!(eps>0&&eps<=FLT_MAX))return -1;
 const void* ptrs[]={x,w,z,stats};
 uint64_t rows=batch*6,sizes[]={rows*K*2,uint64_t(N)*K*4,rows*N*4,rows*4};
 for(int i=0;i<4;++i){if(!aligned(ptrs[i]))return -1;
  for(int j=0;j<i;++j)if(overlaps(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;}
 hc_project_kernel<<<80,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,(uint8_t*)z,
  (uint8_t*)stats,rows,eps);return 0;
}
