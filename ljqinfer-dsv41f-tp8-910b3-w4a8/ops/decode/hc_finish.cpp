// TP reduction epilogue + HC residual expansion.
#include "kernel_operator.h"
#include <cfloat>
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
namespace {
constexpr uint32_t Q=6, MAX_BATCH=4, MAX_N=5120, MAX_BLOCKS=40;
inline bool aligned(const void* p) { return p && !(uintptr_t(p)&31); }
inline bool overlap(const void* a, uint64_t an, const void* b, uint64_t bn) {
    uintptr_t x=uintptr_t(a), y=uintptr_t(b);
    return x<=y ? y-x<an : x-y<bn;
}
inline uint32_t blocks_for(uint32_t n) { return n<MAX_BLOCKS?n:MAX_BLOCKS; }
template<HardEvent E> __aicore__ inline void fence(){ SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
}
template<bool MOE> __global__ __aicore__ void hc_finish_kernel(
    GM_ADDR X,GM_ADDR S,GM_ADDR R,GM_ADDR POST,GM_ADDR COMB,GM_ADDR Y,
    uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120,T=2560;
    TPipe p;TBuf<TPosition::VECCALC> bb,bf,ba,bt,bc,bo,bx,bs;
    p.InitBuffer(bb,5*T*2);p.InitBuffer(bf,5*T*4);
    p.InitBuffer(ba,T*4);p.InitBuffer(bt,T*4);p.InitBuffer(bc,96);
    p.InitBuffer(bo,T*2);p.InitBuffer(bx,T*4);p.InitBuffer(bs,T*4);
    auto x=bx.Get<float>(),shared=bs.Get<float>();
    auto rb=bb.Get<bfloat16_t>();auto rf=bf.Get<float>();
    auto acc=ba.Get<float>(),tmp=bt.Get<float>(),coeff=bc.Get<float>();
    auto out=bo.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gr,gy;GlobalTensor<float> gx,gs,gp,gc;
    gx.SetGlobalBuffer((__gm__ float*)X);
    if constexpr(MOE)gs.SetGlobalBuffer((__gm__ float*)S);gr.SetGlobalBuffer((__gm__ bfloat16_t*)R);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gp.SetGlobalBuffer((__gm__ float*)POST);gc.SetGlobalBuffer((__gm__ float*)COMB);
    for(uint32_t job=GetBlockIdx();job<rows*2;job+=blocks){
        uint32_t row=job/2,offset=(job%2)*T;
        DataCopyPad(coeff,gp[uint64_t(row)*4],DataCopyExtParams{1,16,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        DataCopy(coeff[8],gc[uint64_t(row)*16],int32_t(16));
        for(int s=0;s<4;++s)
            DataCopy(rb[s*T],gr[uint64_t(row)*4*D+s*D+offset],T);
        DataCopy(x,gx[uint64_t(row)*D+offset],T);
        if constexpr(MOE)DataCopy(shared,gs[uint64_t(row)*D+offset],T);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
        if constexpr(MOE){
            Cast(rb[4*T],shared,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
            Cast(shared,rb[4*T],RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
            Add(x,x,shared,T);PipeBarrier<PIPE_V>();
        }
        // Preserve both standalone BF16 rounding boundaries exactly.
        Cast(rb[4*T],x,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
        Cast(rf,rb,RoundMode::CAST_NONE,int32_t(5*T));PipeBarrier<PIPE_V>();
        for(int g=0;g<4;++g){
            Muls(acc,rf,coeff.GetValue(8+g),T);PipeBarrier<PIPE_V>();
            for(int s=1;s<4;++s){
                Muls(tmp,rf[s*T],coeff.GetValue(8+s*4+g),T);PipeBarrier<PIPE_V>();
                Add(acc,acc,tmp,T);PipeBarrier<PIPE_V>();
            }
            Muls(tmp,rf[4*T],coeff.GetValue(g),T);PipeBarrier<PIPE_V>();
            Add(acc,tmp,acc,T);PipeBarrier<PIPE_V>();
            Cast(out,acc,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_ALL>();
            DataCopy(gy[uint64_t(row)*4*D+g*D+offset],out,T);PipeBarrier<PIPE_ALL>();
        }
    }
}

// Caller-owned contiguous storage; B=1..4, Q=6, hidden=5120, HC=4.
// x/shared: FP32[B,Q,D], residual/output: BF16[B,Q,4,D],
// post: FP32[B,Q,4], comb: FP32[B,Q,4,4]. Output may alias residual.
// Attention: BF16(x); MoE: BF16(x + FP32(BF16(shared))).
// No allocation, synchronization, workspace or communication.
extern "C" int dec_hc_finish(void* stream,void* x,void* shared,void* residual,
    void* post,void* comb,void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH)return -1;
    uint64_t rows=uint64_t(batch)*Q;
    void* ptrs[]={x,residual,post,comb};
    uint64_t sizes[]={rows*20480,rows*40960,rows*16,rows*64};
    if(!aligned(out))return -1;
    for(int i=0;i<4;++i){
        if(!aligned(ptrs[i]))return -1;
        if(!(i==1 && out==residual) && overlap(out,rows*40960,ptrs[i],sizes[i]))return -1;
    }
    if(shared && (!aligned(shared)||overlap(out,rows*40960,shared,rows*20480)))return -1;
    uint32_t blocks=blocks_for(rows*2);
    if(shared)hc_finish_kernel<true><<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)shared,
        (uint8_t*)residual,(uint8_t*)post,(uint8_t*)comb,(uint8_t*)out,rows,blocks);
    else hc_finish_kernel<false><<<blocks,nullptr,stream>>>((uint8_t*)x,nullptr,
        (uint8_t*)residual,(uint8_t*)post,(uint8_t*)comb,(uint8_t*)out,rows,blocks);
    return 0;
}
