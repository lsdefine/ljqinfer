// Full-vocabulary inverse-CDF sampling, fixed p=.95. See sampling.py.
#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
__aicore__ inline uint32_t mix(uint32_t k){k*=2654435761u;k^=k>>15;k*=2246822519u;k^=k>>13;k*=3266489917u;k^=k>>16;return k;}
__aicore__ inline float getf(GlobalTensor<float>& g,LocalTensor<float> b,uint32_t i){
 DataCopy(b,g[i&~7u],8);SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 float v=b.GetValue(i&7u);SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);return v;}
__aicore__ inline int64_t geti(GlobalTensor<int64_t>& g,LocalTensor<int64_t> b,uint32_t i){
 DataCopy(b,g[i&~3u],4);SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 int64_t v=b.GetValue(i&3u);SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);return v;}
__aicore__ inline uint32_t upper(GlobalTensor<float>& g,LocalTensor<float> b,uint32_t row,uint32_t n,float x){
 uint32_t lo=0,hi=n;while(lo<hi){uint32_t m=lo+(hi-lo)/2;if(getf(g,b,row*n+m)<=x)lo=m+1;else hi=m;}return lo;}
extern "C" __global__ __aicore__ void finish_kernel(GM_ADDR C,GM_ADDR I,GM_ADDR T,GM_ADDR S,GM_ADDR O,GM_ADDR G,uint32_t rows,uint32_t n){
 const uint32_t first=GetBlockIdx()*4;
 if(first>=rows)return;
 const uint32_t count=rows-first<4?rows-first:4;
 TPipe pipe;TBuf<TPosition::VECCALC> bf,bi,bt,bs,bo;
 pipe.InitBuffer(bf,32);pipe.InitBuffer(bi,32);pipe.InitBuffer(bt,32);pipe.InitBuffer(bs,32);pipe.InitBuffer(bo,rows*8+32);
 auto fb=bf.Get<float>();auto ib=bi.Get<int64_t>();auto tb=bt.Get<float>();auto sb=bs.Get<int32_t>();auto ob=bo.Get<int64_t>();
 GlobalTensor<float> cg,tg;GlobalTensor<int64_t> ig,og,gg;GlobalTensor<int32_t> sg;
 gg.SetGlobalBuffer((__gm__ int64_t*)G);cg.SetGlobalBuffer((__gm__ float*)C);ig.SetGlobalBuffer((__gm__ int64_t*)I);tg.SetGlobalBuffer((__gm__ float*)T);sg.SetGlobalBuffer((__gm__ int32_t*)S);og.SetGlobalBuffer((__gm__ int64_t*)O);
 DataCopyPad(sb,sg,DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
 DataCopyPad(tb,tg,DataCopyExtParams{1,rows/6*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);uint32_t seed=uint32_t(sb.GetValue(0));
 for(uint32_t row=first;row<first+count;++row){
  uint32_t pick=n-1;
  if(tb.GetValue(row/6)>0){
   float total=getf(cg,fb,row*n+n-1);uint32_t k=upper(cg,fb,row,n,0.05f);if(k>=n)k=n-1;
   float tail=k?getf(cg,fb,row*n+k-1):0.f;
   uint32_t h=mix(mix(row^(seed*2654435761u+0x9E3779B9u))+0x85EBCA6Bu);
   float u=float(int32_t(h>>8))*5.9604645e-8f+2.9802322e-8f;
   float target=tail+(total-tail)*u;pick=upper(cg,fb,row,n,target);if(pick>=n)pick=n-1;
  }
  ob.SetValue(row-first,tb.GetValue(row/6)>0?geti(ig,ib,row*n+pick):geti(gg,ib,row));
 }
 SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
 DataCopyPad(og[first],ob,DataCopyExtParams{1,count*8,0,0,0});
}
extern "C" __global__ __aicore__ void advance_seed(GM_ADDR S){
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,32);
 auto b=buf.Get<int32_t>();GlobalTensor<int32_t> g;g.SetGlobalBuffer((__gm__ int32_t*)S);
 DataCopyPad(b,g,DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});
 SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
 b.SetValue(0,int32_t((uint32_t(b.GetValue(0))+1)&0x7fffffffu));
 SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
 DataCopyPad(g,b,DataCopyExtParams{1,4,0,0,0});
}
extern "C" int dec_top_p_finish(void* stream,void* cdf,void* ids,void* temp,void* seed,void* out,void* greedy,uint32_t rows,uint32_t n){
 if(!stream||!cdf||!ids||!temp||!seed||!out||!greedy||rows<6||rows>24||rows%6||n!=129280)return -1;
 finish_kernel<<<(rows+3)/4,nullptr,stream>>>((uint8_t*)cdf,(uint8_t*)ids,(uint8_t*)temp,(uint8_t*)seed,(uint8_t*)out,(uint8_t*)greedy,rows,n);
 advance_seed<<<1,nullptr,stream>>>((uint8_t*)seed);return 0;
}
