// Fixed Q6 KV preparation: RMSNorm -> RoPE -> simulated FP8 QDQ.
#include "kernel_operator.h"
#include <cfloat>
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
namespace {
constexpr uint32_t Q=6, MAX_BATCH=4, MAX_N=512, MAX_BLOCKS=40;
inline bool aligned(const void* p) { return p && !(uintptr_t(p)&31); }
inline bool overlap(const void* a, uint64_t an, const void* b, uint64_t bn) {
    uintptr_t x=uintptr_t(a), y=uintptr_t(b);
    return x<=y ? y-x<an : x-y<bn;
}
inline uint32_t blocks_for(uint32_t n) { return n<MAX_BLOCKS?n:MAX_BLOCKS; }
template<HardEvent E> __aicore__ inline void fence(){ SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
}
__aicore__ inline float kv_powceil(float a) {
    union { float f; uint32_t u; } x; x.f=a;
    x.u=(x.u&0x7f800000u)+((x.u&0x7fffffu)?0x800000u:0u);return x.f;
}

__global__ __aicore__ void kv_prepare_kernel(GM_ADDR X,GM_ADDR W,GM_ADDR F,GM_ADDR Y,uint32_t rows,float eps){
    if(GetBlockIdx()>=rows)return;
    const uint32_t row=GetBlockIdx();constexpr int32_t n=512;constexpr float inv_n=1.f/512.f;
    TPipe p;TBuf<TPosition::VECCALC> bh;
    p.InitBuffer(bh,1024);auto h=bh.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<float> gw,gf;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gw.SetGlobalBuffer((__gm__ float*)W);gf.SetGlobalBuffer((__gm__ float*)F);
    DataCopy(h,gx[uint64_t(row)*512],512);PipeBarrier<PIPE_ALL>();
    {
        TBuf<TPosition::VECCALC> bx,bw,bs,bt,br;
        p.InitBuffer(bx,2048);p.InitBuffer(bw,2048);p.InitBuffer(bs,2048);p.InitBuffer(bt,2048);p.InitBuffer(br,128);
        auto x=bx.Get<float>(),w=bw.Get<float>(),s=bs.Get<float>(),tmp=bt.Get<float>(),red=br.Get<float>();
        DataCopy(w,gw,512);PipeBarrier<PIPE_ALL>();
        Cast(x,h,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
        Mul(s,x,x,n);Duplicate(red,0.0f,32);PipeBarrier<PIPE_V>();
        ReduceSum(red,s,tmp,n);PipeBarrier<PIPE_V>();
        Muls(red,red,inv_n,32);PipeBarrier<PIPE_V>();
        Adds(red,red,eps,32);PipeBarrier<PIPE_V>();
        Sqrt(red,red,32);Duplicate(tmp,1.0f,32);PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float inv=red.GetValue(0);
        Muls(x,x,inv,n);PipeBarrier<PIPE_V>();
        Mul(x,x,w,n);PipeBarrier<PIPE_V>();
        Cast(h,x,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_ALL>();
    }
    {
        TBuf<TPosition::VECCALC> bz,bf,bi,ba,bb,bc,bd,bu,bv,bt;
        p.InitBuffer(bz,256);p.InitBuffer(bf,256);p.InitBuffer(bi,512);
        p.InitBuffer(ba,128);p.InitBuffer(bb,128);p.InitBuffer(bc,128);p.InitBuffer(bd,128);
        p.InitBuffer(bu,256);p.InitBuffer(bv,256);p.InitBuffer(bt,128);
        auto z=bz.Get<float>(),f=bf.Get<float>(),a=ba.Get<float>(),b=bb.Get<float>();
        auto c=bc.Get<float>(),d=bd.Get<float>(),u=bu.Get<float>(),v=bv.Get<float>(),t=bt.Get<float>();
        auto idx=bi.Get<uint32_t>();
        for(uint32_t j=0;j<32;j++){
            idx.SetValue(j,8*j);idx.SetValue(32+j,8*j+4);
            idx.SetValue(64+2*j,4*j);idx.SetValue(65+2*j,4*(32+j));
        }
        fence<HardEvent::S_V>();DataCopy(f,gf[uint64_t(row)*64],64);PipeBarrier<PIPE_ALL>();
        Cast(z,h[448],RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Gather(a,z,idx,0,32);Gather(b,z,idx[32],0,32);
        Gather(c,f,idx,0,32);Gather(d,f,idx[32],0,32);PipeBarrier<PIPE_V>();
        Mul(u,a,c,int32_t(32));Mul(t,b,d,int32_t(32));PipeBarrier<PIPE_V>();
        Sub(u,u,t,int32_t(32));PipeBarrier<PIPE_V>();
        Mul(u[32],a,d,int32_t(32));Mul(t,b,c,int32_t(32));PipeBarrier<PIPE_V>();
        Add(u[32],u[32],t,int32_t(32));PipeBarrier<PIPE_V>();
        Gather(v,u,idx[64],0,64);PipeBarrier<PIPE_V>();
        Cast(h[448],v,RoundMode::CAST_RINT,64);PipeBarrier<PIPE_ALL>();
    }
    {
        TBuf<TPosition::VECCALC> fb,ib;p.InitBuffer(fb,7*512*4);p.InitBuffer(ib,512*4);
        auto x=fb.Get<float>(),a=x[512],scale=x[1024],u=x[1536],step=x[2048],y=x[2560],tmp=x[3072];
        auto ints=ib.Get<int32_t>();
        Cast(x,h,RoundMode::CAST_NONE,int32_t(n));PipeBarrier<PIPE_V>();
        Abs(a,x,int32_t(n));PipeBarrier<PIPE_V>();
        for(int j=0;j<n;j+=32){
            ReduceMax(tmp,a[j],y,int32_t(32),false);fence<HardEvent::V_S>();
            float s=kv_powceil((tmp.GetValue(0)>1e-4f?tmp.GetValue(0):1e-4f)/448.f);
            fence<HardEvent::S_V>();
            Duplicate(scale[j],s,int32_t(32));PipeBarrier<PIPE_V>();
        }
        Div(u,x,scale,int32_t(n));PipeBarrier<PIPE_V>();
        Mins(u,u,448.f,int32_t(n));PipeBarrier<PIPE_V>();
        Maxs(u,u,-448.f,int32_t(n));PipeBarrier<PIPE_V>();
        Abs(a,u,int32_t(n));PipeBarrier<PIPE_V>();
        Maxs(a,a,0.015625f,int32_t(n));PipeBarrier<PIPE_V>();
        ShiftRight(step.ReinterpretCast<uint32_t>(),a.ReinterpretCast<uint32_t>(),uint32_t(23),int32_t(n));PipeBarrier<PIPE_V>();
        ShiftLeft(step.ReinterpretCast<uint32_t>(),step.ReinterpretCast<uint32_t>(),uint32_t(23),int32_t(n));PipeBarrier<PIPE_V>();
        Muls(step,step,0.125f,int32_t(n));PipeBarrier<PIPE_V>();
        Div(y,u,step,int32_t(n));PipeBarrier<PIPE_V>();
        Cast(ints,y,RoundMode::CAST_RINT,int32_t(n));PipeBarrier<PIPE_V>();
        Cast(y,ints,RoundMode::CAST_NONE,int32_t(n));PipeBarrier<PIPE_V>();
        Mul(y,y,step,int32_t(n));PipeBarrier<PIPE_V>();
        Mul(y,y,scale,int32_t(n));PipeBarrier<PIPE_V>();
        // Integer rounding erases negative zero; restore input sign bit exactly.
        ShiftRight(tmp.ReinterpretCast<uint32_t>(),x.ReinterpretCast<uint32_t>(),uint32_t(31),int32_t(n));PipeBarrier<PIPE_V>();
        ShiftLeft(tmp.ReinterpretCast<uint32_t>(),tmp.ReinterpretCast<uint32_t>(),uint32_t(31),int32_t(n));PipeBarrier<PIPE_V>();
        Or(y.ReinterpretCast<uint16_t>(),y.ReinterpretCast<uint16_t>(),tmp.ReinterpretCast<uint16_t>(),int32_t(2*n));PipeBarrier<PIPE_V>();
        Cast(h,y,RoundMode::CAST_RINT,int32_t(n));fence<HardEvent::V_MTE3>();
        DataCopy(gy[uint64_t(row)*512],h,512);fence<HardEvent::MTE3_MTE2>();
    }
}
extern "C" int dec_kv_prepare(void* stream,void* x,void* w,void* freqs,void* y,uint32_t batch,float eps){
    if(!stream||!aligned(x)||!aligned(w)||!aligned(freqs)||!aligned(y)||!batch||batch>MAX_BATCH||!(eps>0.f&&eps<=FLT_MAX))return -1;
    uint64_t bytes=uint64_t(batch)*6*512*2;
    if((x!=y&&overlap(x,bytes,y,bytes))||overlap(w,2048,y,bytes)||overlap(freqs,uint64_t(batch)*6*256,y,bytes))return -1;
    kv_prepare_kernel<<<batch*6,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,(uint8_t*)freqs,(uint8_t*)y,batch*6,eps);
    return 0;
}
