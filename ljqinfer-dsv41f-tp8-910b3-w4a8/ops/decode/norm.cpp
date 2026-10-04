// Decode-only leaf ABI (not compatible with opsbak dec_rms).
// Caller owns persistent, contiguous, 32-byte-aligned device buffers and passes
// the CURRENT stream on EVERY call. No allocation, sync, cache or lazy init.
// B=1..MAX_BATCH, Q=6. To extend to B8 change MAX_BATCH, rebuild, reallocate
// all caller buffers and recapture. No active mask: every supplied row is used.
// dec_rms_norm_f32(s,x,w,y,B,N,eps): BF16 x/y[B,6,N], N=128/512/1280/5120;
//   optional FP32 w[N] (nullptr => ones). FP32 sum/mean/sqrt/div/multiply,
//   one final BF16 RNE. eps finite >0; finite inputs with finite FP32 sum(x*x).
// dec_rope(s,x,f,y,B,H,D,inverse): BF16 x/y[B,6,H,D], H=1..128,
//   D=128/512; FP32 f[B,6,32,2] contains interleaved (cos,sin), NOT angles.
//   Rotate adjacent pairs in LAST 64 columns; prefix is bit-exact copied.
//   inverse=1 conjugates sin; phases/positions/YaRN are caller's responsibility.
// Both allow y==x; otherwise all output/input ranges must be disjoint.
// Weights/phases cannot overlap y. No caller workspace. Return -1 for invalid
// host geometry/pointers/alias, 0 for launch submitted (NOT async completion).
// Warm kernels before capture; caller owns error checks, lifetime and ordering.
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
extern "C" __global__ __aicore__ void decode_rms_norm_kernel(
    GM_ADDR X,GM_ADDR W,GM_ADDR Y,uint32_t rows,int32_t n,float eps,
    uint32_t weighted,uint32_t blocks,float inv_n) {
    // AIV id can run to 2*blockDim-1 on 910B3. Discard the extra lanes:
    // ids [0,blocks) alone own row=id+k*blocks, even for in-place operation.
    if(GetBlockIdx()>=blocks)return;
    TPipe p;TBuf<TPosition::VECCALC> bh,bx,bw,bs,bt,br;
    p.InitBuffer(bh,MAX_N*2);p.InitBuffer(bx,MAX_N*4);
    p.InitBuffer(bw,MAX_N*4);p.InitBuffer(bs,MAX_N*4);
    p.InitBuffer(bt,MAX_N*4);p.InitBuffer(br,128);
    auto h=bh.Get<bfloat16_t>();auto x=bx.Get<float>();auto w=bw.Get<float>();
    auto s=bs.Get<float>();auto tmp=bt.Get<float>();auto red=br.Get<float>();
    GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<float> gw;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    if(weighted){
        gw.SetGlobalBuffer((__gm__ float*)W);DataCopy(w,gw,n);
        PipeBarrier<PIPE_ALL>();
    }
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        uint64_t off=uint64_t(row)*n;
        DataCopy(h,gx[off],n);PipeBarrier<PIPE_ALL>();
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
        if(weighted){Mul(x,x,w,n);PipeBarrier<PIPE_V>();}
        Cast(h,x,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[off],h,n);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void decode_rope_kernel(
    GM_ADDR X,GM_ADDR F,GM_ADDR Y,uint32_t rows,uint32_t heads,
    uint32_t dim,uint32_t inverse,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    TPipe p;TBuf<TPosition::VECCALC> bh,bz,bf,bi,ba,bb,bc,bd,bu,bv,bt;
    p.InitBuffer(bh,1024);p.InitBuffer(bz,256);p.InitBuffer(bf,256);
    p.InitBuffer(bi,512);p.InitBuffer(ba,128);p.InitBuffer(bb,128);
    p.InitBuffer(bc,128);p.InitBuffer(bd,128);p.InitBuffer(bu,256);
    p.InitBuffer(bv,256);p.InitBuffer(bt,128);
    auto h=bh.Get<bfloat16_t>();auto z=bz.Get<float>();auto f=bf.Get<float>();
    auto idx=bi.Get<uint32_t>();auto a=ba.Get<float>();auto b=bb.Get<float>();
    auto c=bc.Get<float>();auto d=bd.Get<float>();auto u=bu.Get<float>();
    auto v=bv.Get<float>();auto t=bt.Get<float>();
    // Gather offsets are BYTES, including the final planar->interleaved gather.
    for(uint32_t j=0;j<32;j++){
        idx.SetValue(j,8*j);idx.SetValue(32+j,8*j+4);
        idx.SetValue(64+2*j,4*j);idx.SetValue(65+2*j,4*(32+j));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<float> gf;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gf.SetGlobalBuffer((__gm__ float*)F);
    for(uint32_t task=GetBlockIdx();task<rows*heads;task+=blocks){
        uint64_t off=uint64_t(task)*dim;
        DataCopy(h,gx[off],dim);DataCopy(f,gf[uint64_t(task/heads)*64],64);
        PipeBarrier<PIPE_ALL>();Cast(z,h[dim-64],RoundMode::CAST_NONE,64);
        PipeBarrier<PIPE_V>();
        Gather(a,z,idx,0,32);Gather(b,z,idx[32],0,32);
        Gather(c,f,idx,0,32);Gather(d,f,idx[32],0,32);PipeBarrier<PIPE_V>();
        if(inverse){Muls(d,d,-1.0f,32);PipeBarrier<PIPE_V>();}
        Mul(u,a,c,int32_t(32));Mul(t,b,d,int32_t(32));PipeBarrier<PIPE_V>();
        Sub(u,u,t,int32_t(32));PipeBarrier<PIPE_V>();
        Mul(u[32],a,d,int32_t(32));Mul(t,b,c,int32_t(32));PipeBarrier<PIPE_V>();
        Add(u[32],u[32],t,int32_t(32));PipeBarrier<PIPE_V>();
        Gather(v,u,idx[64],0,64);PipeBarrier<PIPE_V>();
        Cast(h[dim-64],v,RoundMode::CAST_RINT,64);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[off],h,dim);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_rms_norm_f32(void* stream,void* x,void* w,void* y,
    uint32_t batch,uint32_t n,float eps){
    if(!stream||!aligned(x)||!aligned(y)||(w&&!aligned(w))||!batch||batch>MAX_BATCH||
       !(n==128||n==512||n==1280||n==5120)||!(eps>0.0f&&eps<=FLT_MAX))return -1;
    uint32_t rows=batch*Q,blocks=blocks_for(rows);uint64_t bytes=uint64_t(rows)*n*2;
    if((x!=y&&overlap(x,bytes,y,bytes))||(w&&overlap(w,n*4,y,bytes)))return -1;
    decode_rms_norm_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,
        (uint8_t*)y,rows,int32_t(n),eps,w!=nullptr,blocks,1.0f/float(n));return 0;
}
extern "C" int dec_rope(void* stream,void* x,void* freqs,void* y,
    uint32_t batch,uint32_t heads,uint32_t dim,uint32_t inverse){
    if(!stream||!aligned(x)||!aligned(freqs)||!aligned(y)||!batch||batch>MAX_BATCH||
       !heads||heads>128||!(dim==128||dim==512)||inverse>1)return -1;
    uint32_t rows=batch*Q,blocks=blocks_for(rows*heads);
    uint64_t bytes=uint64_t(rows)*heads*dim*2;
    if((x!=y&&overlap(x,bytes,y,bytes))||overlap(freqs,uint64_t(rows)*256,y,bytes))return -1;
    decode_rope_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)freqs,
        (uint8_t*)y,rows,heads,dim,inverse,blocks);return 0;
}

// Q6 HC collapse + RMS. BF16 residual[B,6,4,5120], FP32 pre[B,6,4]
// and weight[5120], BF16 out[B,6,5120]. Collapse rounds to BF16 BEFORE RMS.
// Output is disjoint from all inputs. No workspace or Past mutation.
extern "C" __global__ __aicore__ void decode_hc_norm_kernel(
    GM_ADDR R,GM_ADDR P,GM_ADDR W,GM_ADDR Y,uint32_t rows,float eps,uint32_t blocks) {
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe pipe;TBuf<TPosition::VECCALC> bh,bx,ba,bt,bw,br,bc;
    pipe.InitBuffer(bh,D*2);pipe.InitBuffer(bx,D*4);pipe.InitBuffer(ba,D*4);
    pipe.InitBuffer(bt,D*4);pipe.InitBuffer(bw,D*4);
    pipe.InitBuffer(br,128);pipe.InitBuffer(bc,32);
    auto h=bh.Get<bfloat16_t>();auto x=bx.Get<float>();auto acc=ba.Get<float>();
    auto tmp=bt.Get<float>();auto w=bw.Get<float>();
    auto red=br.Get<float>();auto coeff=bc.Get<float>();
    GlobalTensor<bfloat16_t> gr,gy;GlobalTensor<float> gp,gw;
    gr.SetGlobalBuffer((__gm__ bfloat16_t*)R);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gp.SetGlobalBuffer((__gm__ float*)P);gw.SetGlobalBuffer((__gm__ float*)W);
    DataCopy(w,gw,D);PipeBarrier<PIPE_ALL>();
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks) {
        DataCopyPad(coeff,gp[uint64_t(row)*4],DataCopyExtParams{1,16,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float v[4];for(int c=0;c<4;++c)v[c]=coeff.GetValue(c);
        for(int c=0;c<4;++c) {
            DataCopy(h,gr[(uint64_t(row)*4+c)*D],D);PipeBarrier<PIPE_ALL>();
            Cast(x,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
            if(c==0){Muls(acc,x,v[c],D);}
            else {Muls(tmp,x,v[c],D);PipeBarrier<PIPE_V>();Add(acc,acc,tmp,D);}
            PipeBarrier<PIPE_ALL>();
        }
        Cast(h,acc,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(x,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Mul(acc,x,x,D);Duplicate(red,0.0f,32);PipeBarrier<PIPE_V>();
        ReduceSum(red,acc,tmp,D);PipeBarrier<PIPE_V>();
        Muls(red,red,1.0f/5120.0f,32);PipeBarrier<PIPE_V>();
        Adds(red,red,eps,32);PipeBarrier<PIPE_V>();
        Sqrt(red,red,32);Duplicate(tmp,1.0f,32);PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float inv=red.GetValue(0);
        Muls(x,x,inv,D);PipeBarrier<PIPE_V>();Mul(x,x,w,D);PipeBarrier<PIPE_V>();
        Cast(h,x,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(row)*D],h,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_hc_norm(void* stream,void* residual,void* pre,void* weight,
                           void* out,uint32_t batch,float eps) {
    if(!stream||!aligned(residual)||!aligned(pre)||!aligned(weight)||!aligned(out)||
       !batch||batch>MAX_BATCH||!(eps>0.0f&&eps<=FLT_MAX))return -1;
    uint64_t rows=uint64_t(batch)*Q,bytes=rows*5120*2;
    if(overlap(out,bytes,residual,bytes*4)||overlap(out,bytes,pre,rows*16)||
       overlap(out,bytes,weight,5120*4))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows));
    decode_hc_norm_kernel<<<blocks,nullptr,stream>>>((uint8_t*)residual,
        (uint8_t*)pre,(uint8_t*)weight,(uint8_t*)out,uint32_t(rows),eps,blocks);
    return 0;
}

// HC preparation: BF16 residual[B,6,20480] -> FP32 h and inv-RMS[B,6].
// The projection consumes unnormalized h; normalize its FP32 output afterwards.
extern "C" __global__ __aicore__ void decode_hc_prepare_kernel(
    GM_ADDR X,GM_ADDR H,GM_ADDR ST,uint32_t rows,float eps,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> bh,bx,bs,bt,br;
    p.InitBuffer(bh,D*2);p.InitBuffer(bx,D*4);p.InitBuffer(bs,D*4);
    p.InitBuffer(bt,D*4);p.InitBuffer(br,128);
    auto h=bh.Get<bfloat16_t>();auto x=bx.Get<float>(),s=bs.Get<float>();
    auto tmp=bt.Get<float>(),red=br.Get<float>();
    GlobalTensor<bfloat16_t> gx;GlobalTensor<float> gh,gs;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gh.SetGlobalBuffer((__gm__ float*)H);
    gs.SetGlobalBuffer((__gm__ float*)ST);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        float total=0.0f;
        for(int tile=0;tile<4;++tile){
            uint64_t off=uint64_t(row)*20480+tile*D;
            DataCopy(h,gx[off],D);PipeBarrier<PIPE_ALL>();
            Cast(x,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_ALL>();
            DataCopy(gh[off],x,D);
            Mul(s,x,x,D);PipeBarrier<PIPE_V>();
            ReduceSum(red,s,tmp,D);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            total+=red.GetValue(0);PipeBarrier<PIPE_ALL>();
        }
        Duplicate(red,total,int32_t(32));PipeBarrier<PIPE_V>();
        Muls(red,red,1.0f/20480.0f,int32_t(32));PipeBarrier<PIPE_V>();
        Adds(red,red,eps,int32_t(32));PipeBarrier<PIPE_V>();
        Sqrt(red,red,int32_t(32));Duplicate(tmp,1.0f,int32_t(32));PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32));PipeBarrier<PIPE_ALL>();
        DataCopyPad(gs[row],red,DataCopyExtParams{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_hc_prepare(void* stream,void* x,void* h,void* stats,uint32_t batch,float eps){
    if(!stream||!batch||batch>MAX_BATCH||!(eps>0&&eps<=FLT_MAX)
       ||!aligned(x)||!aligned(h)||!aligned(stats))return -1;
    uint64_t rows=uint64_t(batch)*Q;
    if(overlap(x,rows*40960,h,rows*81920)||overlap(x,rows*40960,stats,rows*4)
       ||overlap(h,rows*81920,stats,rows*4))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows));
    decode_hc_prepare_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)h,
        (uint8_t*)stats,uint32_t(rows),eps,blocks);return 0;
}
// BF16 x[B,6,5120], residual/out[B,6,4,5120]; FP32 post[B,6,4],
// comb[B,6,source,target]. Out must be disjoint from all inputs.
// Preserve sequential FP32 residual sum, add post*x, then round to BF16.
extern "C" __global__ __aicore__ void decode_hc_expand_kernel(
    GM_ADDR X,GM_ADDR R,GM_ADDR POST,GM_ADDR COMB,GM_ADDR Y,
    uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120,T=2560;
    TPipe p;TBuf<TPosition::VECCALC> bb,bf,ba,bt,bc,bo;
    p.InitBuffer(bb,5*T*2);p.InitBuffer(bf,5*T*4);
    p.InitBuffer(ba,T*4);p.InitBuffer(bt,T*4);p.InitBuffer(bc,96);
    p.InitBuffer(bo,T*2);
    auto rb=bb.Get<bfloat16_t>();auto rf=bf.Get<float>();
    auto acc=ba.Get<float>(),tmp=bt.Get<float>(),coeff=bc.Get<float>();
    auto out=bo.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> gx,gr,gy;GlobalTensor<float> gp,gc;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gr.SetGlobalBuffer((__gm__ bfloat16_t*)R);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gp.SetGlobalBuffer((__gm__ float*)POST);gc.SetGlobalBuffer((__gm__ float*)COMB);
    for(uint32_t job=GetBlockIdx();job<rows*2;job+=blocks){
        uint32_t row=job/2,offset=(job%2)*T;
        DataCopyPad(coeff,gp[uint64_t(row)*4],DataCopyExtParams{1,16,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        DataCopy(coeff[8],gc[uint64_t(row)*16],int32_t(16));
        for(int s=0;s<4;++s)
            DataCopy(rb[s*T],gr[uint64_t(row)*4*D+s*D+offset],T);
        DataCopy(rb[4*T],gx[uint64_t(row)*D+offset],T);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();
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
extern "C" int dec_hc_expand(void* stream,void* x,void* residual,void* post,
    void* comb,void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH)return -1;
    void* ptrs[]={x,residual,post,comb,out};
    uint64_t rows=uint64_t(batch)*Q,sizes[]={rows*10240,rows*40960,rows*16,rows*64,rows*40960};
    for(int i=0;i<5;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=0;i<4;++i)if(overlap(out,sizes[4],ptrs[i],sizes[i]))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows*2));
    decode_hc_expand_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)residual,
        (uint8_t*)post,(uint8_t*)comb,(uint8_t*)out,uint32_t(rows),blocks);return 0;
}

// Explicit TP boundary: FP32 all-reduced [B,6,5120] -> BF16.
extern "C" __global__ __aicore__ void decode_tp_cast_kernel(
    GM_ADDR X,GM_ADDR Y,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> bx,by;
    p.InitBuffer(bx,D*4);p.InitBuffer(by,D*2);
    auto x=bx.Get<float>();auto y=by.Get<bfloat16_t>();
    GlobalTensor<float> gx;GlobalTensor<bfloat16_t> gy;
    gx.SetGlobalBuffer((__gm__ float*)X);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        uint64_t off=uint64_t(row)*D;
        DataCopy(x,gx[off],D);PipeBarrier<PIPE_ALL>();
        Cast(y,x,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[off],y,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_tp_cast(void* stream,const void* x,void* y,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(x)||!aligned(y))return -1;
    uint32_t rows=batch*Q;uint64_t n=uint64_t(rows)*5120;
    if(overlap(x,n*4,y,n*2))return -1;
    uint32_t blocks=blocks_for(rows);
    decode_tp_cast_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)y,rows,blocks);
    return 0;
}

// Shared expert activation: BF16 inputs/output, clipped FP32 SwiGLU.
// Gate has only an upper clamp; up has symmetric bounds. limit=0 disables.
extern "C" __global__ __aicore__ void decode_swiglu_kernel(
    GM_ADDR G,GM_ADDR U,GM_ADDR Y,uint32_t rows,int32_t width,float limit,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    TPipe p;TBuf<TPosition::VECCALC> bb,bb2,bg,bu,bt;
    p.InitBuffer(bb,width*2);p.InitBuffer(bb2,width*2);p.InitBuffer(bg,width*4);
    p.InitBuffer(bu,width*4);p.InitBuffer(bt,width*4);
    auto b=bb.Get<bfloat16_t>();auto b2=bb2.Get<bfloat16_t>();auto g=bg.Get<float>();
    auto u=bu.Get<float>();auto t=bt.Get<float>();
    GlobalTensor<bfloat16_t> gg,gu,gy;
    gg.SetGlobalBuffer((__gm__ bfloat16_t*)G);gu.SetGlobalBuffer((__gm__ bfloat16_t*)U);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        uint64_t off=uint64_t(row)*width;
        DataCopy(b,gg[off],width);
        DataCopy(b2,gu[off],width);fence<HardEvent::MTE2_V>();
        Cast(g,b,RoundMode::CAST_NONE,width);
        Cast(u,b2,RoundMode::CAST_NONE,width);PipeBarrier<PIPE_V>();
        if(limit>0){
            Mins(g,g,limit,width);Mins(u,u,limit,width);PipeBarrier<PIPE_V>();
            Maxs(u,u,-limit,width);PipeBarrier<PIPE_V>();
        }
        Muls(t,g,-1.0f,width);PipeBarrier<PIPE_V>();
        Exp(t,t,width);PipeBarrier<PIPE_V>();
        Adds(t,t,1.0f,width);PipeBarrier<PIPE_V>();
        Div(g,g,t,width);PipeBarrier<PIPE_V>();
        Mul(g,g,u,width);PipeBarrier<PIPE_V>();
        Cast(b,g,RoundMode::CAST_RINT,width);fence<HardEvent::V_MTE3>();
        DataCopy(gy[off],b,width);fence<HardEvent::MTE3_MTE2>();
    }
}
extern "C" int dec_swiglu(void* stream,const void* gate,const void* up,void* out,
                           uint32_t batch,uint32_t width,float limit){
    if(!stream||!batch||batch>MAX_BATCH||!width||width>4096||width%32||
       !(limit>=0&&limit<=FLT_MAX)||!aligned(gate)||!aligned(up)||!aligned(out))return -1;
    uint32_t rows=batch*Q;uint64_t bytes=uint64_t(rows)*width*2;
    if(overlap(out,bytes,gate,bytes)||overlap(out,bytes,up,bytes))return -1;
    uint32_t blocks=blocks_for(rows);
    decode_swiglu_kernel<<<blocks,nullptr,stream>>>((uint8_t*)gate,(uint8_t*)up,
        (uint8_t*)out,rows,int32_t(width),limit,blocks);return 0;
}

// FP32 routed + BF16-rounded FP32 shared -> BF16 output.
// Both inputs must already include their own TP sums; no communication here.
extern "C" __global__ __aicore__ void decode_moe_finish_kernel(
    GM_ADDR R,GM_ADDR S,GM_ADDR Y,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> br,bs,bb;
    p.InitBuffer(br,D*4);p.InitBuffer(bs,D*4);p.InitBuffer(bb,D*2);
    auto r=br.Get<float>();auto s=bs.Get<float>();auto b=bb.Get<bfloat16_t>();
    GlobalTensor<float> gr,gs;GlobalTensor<bfloat16_t> gy;
    gr.SetGlobalBuffer((__gm__ float*)R);gs.SetGlobalBuffer((__gm__ float*)S);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        uint64_t off=uint64_t(row)*D;
        DataCopy(r,gr[off],D);DataCopy(s,gs[off],D);PipeBarrier<PIPE_ALL>();
        Cast(b,s,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(s,b,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Add(r,r,s,D);PipeBarrier<PIPE_V>();
        Cast(b,r,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[off],b,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_moe_finish(void* stream,const void* routed,const void* shared,
                               void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(routed)||!aligned(shared)||!aligned(out))return -1;
    uint32_t rows=batch*Q;uint64_t n=uint64_t(rows)*5120;
    if(overlap(out,n*2,routed,n*4)||overlap(out,n*2,shared,n*4))return -1;
    uint32_t blocks=blocks_for(rows);
    decode_moe_finish_kernel<<<blocks,nullptr,stream>>>((uint8_t*)routed,(uint8_t*)shared,
        (uint8_t*)out,rows,blocks);return 0;
}

// Target MoE gate: FP32 logits[B*6,384], bias[384] -> IDs/prob[B*6,6].
// Correction bias affects selection only. Positive temperature/scale; finite
// device inputs. This target-only ABI does not stand in for the draft gate.
extern "C" __global__ __aicore__ void decode_route_kernel(
    GM_ADDR L,GM_ADDR B,GM_ADDR I,GM_ADDR P,uint32_t rows,float inv_temp,
    float scale,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t E=384,N=512,K=6;
    TPipe pipe;TBuf<TPosition::VECCALC> bl,bb,bs,bt,bi,ba,bw,bo,bp;
    pipe.InitBuffer(bl,N*4);pipe.InitBuffer(bb,N*4);pipe.InitBuffer(bs,N*4);
    pipe.InitBuffer(bt,N*4);pipe.InitBuffer(bi,N*4);pipe.InitBuffer(ba,N*8);
    pipe.InitBuffer(bw,N*8);pipe.InitBuffer(bo,64);pipe.InitBuffer(bp,32);
    auto l=bl.Get<float>();auto bias=bb.Get<float>();auto score=bs.Get<float>();
    auto tmp=bt.Get<float>();auto ids=bi.Get<uint32_t>();auto sorted=ba.Get<float>();
    auto work=bw.Get<float>();auto out=bo.Get<int64_t>();auto prob=bp.Get<float>();
    GlobalTensor<float> gl,gb,gp;GlobalTensor<int64_t> gi;
    gl.SetGlobalBuffer((__gm__ float*)L);gb.SetGlobalBuffer((__gm__ float*)B);
    gp.SetGlobalBuffer((__gm__ float*)P);gi.SetGlobalBuffer((__gm__ int64_t*)I);
    DataCopy(bias,gb,E);PipeBarrier<PIPE_ALL>();
    CreateVecIndex(ids.ReinterpretCast<int32_t>(),int32_t(0),uint32_t(N));
    PipeBarrier<PIPE_ALL>();
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        DataCopy(l,gl[uint64_t(row)*E],E);PipeBarrier<PIPE_ALL>();
        Muls(l,l,inv_temp,E);PipeBarrier<PIPE_V>();
        // Stable softplus, avoiding overflow in exp for large positive logits.
        Abs(tmp,l,E);PipeBarrier<PIPE_V>();Muls(tmp,tmp,-1.0f,E);PipeBarrier<PIPE_V>();
        Exp(tmp,tmp,E);Maxs(score,l,0.0f,E);PipeBarrier<PIPE_V>();
        // Compensate log(1+t): when t is tiny, FP32 1+t rounds to 1.
        Adds(l,tmp,1.0f,E);PipeBarrier<PIPE_V>();
        Adds(work,l,-1.0f,E);PipeBarrier<PIPE_V>();
        Sub(work,tmp,work,E);PipeBarrier<PIPE_V>();
        Div(work,work,l,E);Ln(tmp,l,E);PipeBarrier<PIPE_V>();
        Add(tmp,tmp,work,E);PipeBarrier<PIPE_V>();
        Add(score,score,tmp,E);PipeBarrier<PIPE_V>();Sqrt(score,score,E);PipeBarrier<PIPE_V>();
        Duplicate(l,-FLT_MAX,N);PipeBarrier<PIPE_V>();Add(l,score,bias,E);PipeBarrier<PIPE_V>();
        Sort<float,true>(sorted,l,ids,work,int32_t(N/32));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float total=0.0f;
        for(int j=0;j<K;++j){
            uint32_t id=sorted.ReinterpretCast<uint32_t>().GetValue(2*j+1);
            out.SetValue(j,int64_t(id));float v=score.GetValue(id);prob.SetValue(j,v);total+=v;
        }
        for(int j=0;j<K;++j)prob.SetValue(j,prob.GetValue(j)/(total+1e-20f)*scale);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gi[uint64_t(row)*K],out,DataCopyExtParams{1,K*8,0,0,0});
        DataCopyPad(gp[uint64_t(row)*K],prob,DataCopyExtParams{1,K*4,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_route(void* stream,const void* logits,const void* bias,
                          void* ids,void* probabilities,uint32_t batch,
                          float temperature,float scale){
    if(!stream||!batch||batch>MAX_BATCH||!(temperature>0&&temperature<=FLT_MAX)
       ||!(1.0f/temperature<=FLT_MAX)||!(scale>0&&scale<=FLT_MAX))return -1;
    const void* ptrs[]={logits,bias,ids,probabilities};uint64_t rows=uint64_t(batch)*Q;
    uint64_t sizes[]={rows*384*4,384*4,rows*6*8,rows*6*4};
    for(int i=0;i<4;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=2;i<4;++i)for(int j=0;j<i;++j)
        if(overlap(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows));
    decode_route_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)bias,
        (uint8_t*)ids,(uint8_t*)probabilities,uint32_t(rows),1.0f/temperature,scale,blocks);
    return 0;
}

// Router projection is FP32: explicitly widen BF16 hidden states first.
extern "C" __global__ __aicore__ void decode_gate_cast_kernel(
    GM_ADDR X,GM_ADDR Y,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> bx,by;
    p.InitBuffer(bx,D*2);p.InitBuffer(by,D*4);
    auto x=bx.Get<bfloat16_t>();auto y=by.Get<float>();
    GlobalTensor<bfloat16_t> gx;GlobalTensor<float> gy;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gy.SetGlobalBuffer((__gm__ float*)Y);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        uint64_t off=uint64_t(row)*D;
        DataCopy(x,gx[off],D);PipeBarrier<PIPE_ALL>();
        Cast(y,x,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[off],y,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_gate_cast(void* stream,const void* x,void* y,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(x)||!aligned(y))return -1;
    uint32_t rows=batch*Q;uint64_t n=uint64_t(rows)*5120;
    if(overlap(x,n*2,y,n*4))return -1;
    uint32_t blocks=blocks_for(rows);
    decode_gate_cast_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)y,rows,blocks);
    return 0;
}

// Stable expert grouping for B*6 tokens and six choices per token.
// IDs must be in [0,384); equal IDs retain original token/choice order.
// Outputs: grouped BF16 rows, FP32 probabilities, original->grouped INT64
// inverse, and 384 INT64 cumulative ends consumed by grouped GEMM.
extern "C" __global__ __aicore__ void decode_dispatch_kernel(
    GM_ADDR X,GM_ADDR I,GM_ADDR P,GM_ADDR Y,GM_ADDR W,GM_ADDR V,GM_ADDR E,
    uint32_t pairs,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120,EX=384,CAP=MAX_BATCH*Q*6;
    TPipe pipe;TBuf<TPosition::VECCALC> bi,bp,bx,be,bv,bw,bc;
    pipe.InitBuffer(bi,((CAP*8+31)/32)*32);
    pipe.InitBuffer(bp,((CAP*4+31)/32)*32);
    pipe.InitBuffer(bx,D*2);pipe.InitBuffer(be,EX*8);
    pipe.InitBuffer(bv,((CAP*8+31)/32)*32);
    pipe.InitBuffer(bw,((CAP*4+31)/32)*32);
    auto inverse=bv.Get<int64_t>();auto weights=bw.Get<float>();
    pipe.InitBuffer(bc,EX*8);auto cursor=bc.Get<int64_t>();
    auto ids=bi.Get<int64_t>();auto prob=bp.Get<float>();
    auto x=bx.Get<bfloat16_t>();auto ends=be.Get<int64_t>();
    GlobalTensor<int64_t> gi,gv,ge;GlobalTensor<float> gp,gw;
    GlobalTensor<bfloat16_t> gx,gy;
    gi.SetGlobalBuffer((__gm__ int64_t*)I);gv.SetGlobalBuffer((__gm__ int64_t*)V);
    ge.SetGlobalBuffer((__gm__ int64_t*)E);gp.SetGlobalBuffer((__gm__ float*)P);
    gw.SetGlobalBuffer((__gm__ float*)W);gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    DataCopyPad(ids,gi,DataCopyExtParams{1,pairs*8,0,0,0},
                DataCopyPadExtParams<int64_t>{false,0,0,0});
    // Only block 0 publishes metadata; other blocks need IDs only.
    if(GetBlockIdx()==0){
        DataCopyPad(prob,gp,DataCopyExtParams{1,pairs*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
    }
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    // Build cumulative ends, stable inverse and probability exactly once.
    if(GetBlockIdx()==0){
        for(int32_t e=0;e<EX;++e)ends.SetValue(e,0);
        for(uint32_t j=0;j<pairs;++j){
            int32_t e=int32_t(ids.GetValue(j));
            ends.SetValue(e,ends.GetValue(e)+1);
        }
        int64_t total=0;
        for(int32_t e=0;e<EX;++e){
            cursor.SetValue(e,total);total+=ends.GetValue(e);ends.SetValue(e,total);
        }
        for(uint32_t j=0;j<pairs;++j){
            int32_t e=int32_t(ids.GetValue(j));int64_t dst=cursor.GetValue(e);
            cursor.SetValue(e,dst+1);inverse.SetValue(j,dst);
            weights.SetValue(dst,prob.GetValue(j));
        }
        // Single-owner DMA avoids cross-core scalar cache-line collisions.
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(ge,ends,EX);
        DataCopyPad(gv,inverse,DataCopyExtParams{1,pairs*8,0,0,0});
        DataCopyPad(gw,weights,DataCopyExtParams{1,pairs*4,0,0,0});
    }
    for(uint32_t j=GetBlockIdx();j<pairs;j+=blocks){
        uint32_t dst;
        if(GetBlockIdx()==0){
            dst=uint32_t(inverse.GetValue(j));
        }else{
            // Stable rank: smaller expert IDs, then earlier equal-ID pairs.
            // Local IDs only: no cross-core dependency on the metadata DMA.
            int32_t e=int32_t(ids.GetValue(j));dst=0;
            for(uint32_t k=0;k<pairs;++k){
                int32_t other=int32_t(ids.GetValue(k));
                dst+=uint32_t(other<e||(other==e&&k<j));
            }
        }
        DataCopy(x,gx[uint64_t(j/6)*D],D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(dst)*D],x,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_dispatch(void* stream,const void* x,const void* ids,
    const void* probability,void* rows,void* sorted_probability,void* inverse,
    void* ends,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH)return -1;
    const void* ptrs[]={x,ids,probability,rows,sorted_probability,inverse,ends};
    uint64_t tokens=batch*Q,pairs=tokens*6;
    uint64_t sizes[]={tokens*5120*2,pairs*8,pairs*4,pairs*5120*2,pairs*4,pairs*8,384*8};
    for(int i=0;i<7;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=3;i<7;++i)for(int j=0;j<i;++j)
        if(overlap(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;
    uint32_t blocks=blocks_for(uint32_t(pairs));
    decode_dispatch_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)ids,
        (uint8_t*)probability,(uint8_t*)rows,(uint8_t*)sorted_probability,
        (uint8_t*)inverse,(uint8_t*)ends,uint32_t(pairs),blocks);return 0;
}

// Grouped expert activation: BF16 [B*36,576], FP32 probability[B*36]
// -> BF16 [B*36,288]. Probability is applied BEFORE the only BF16 round.
extern "C" __global__ __aicore__ void decode_routed_act_kernel(
    GM_ADDR H,GM_ADDR P,GM_ADDR Y,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t K=288,R=8,N=K*R;
    TPipe pipe;TBuf<TPosition::VECCALC> bb,bg,bu,bt,bp;
    pipe.InitBuffer(bb,N*2);pipe.InitBuffer(bg,N*4);
    pipe.InitBuffer(bu,N*4);pipe.InitBuffer(bt,N*4);pipe.InitBuffer(bp,R*4);
    auto b=bb.Get<bfloat16_t>();auto g=bg.Get<float>();
    auto u=bu.Get<float>();auto t=bt.Get<float>();auto prob=bp.Get<float>();
    GlobalTensor<bfloat16_t> gh,gy;GlobalTensor<float> gp;
    gh.SetGlobalBuffer((__gm__ bfloat16_t*)H);gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    gp.SetGlobalBuffer((__gm__ float*)P);
    for(uint32_t tile=GetBlockIdx();tile<(rows+R-1)/R;tile+=blocks){
        uint32_t row=tile*R;uint16_t active=min(uint32_t(R),rows-row);
        int32_t count=int32_t(active)*K;
        DataCopyParams cp{active,K/16,K/16,0};
        DataCopy(b,gh[uint64_t(row)*2*K],cp);PipeBarrier<PIPE_ALL>();
        Cast(g,b,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_ALL>();
        DataCopy(b,gh[uint64_t(row)*2*K+K],cp);
        DataCopyPad(prob,gp[row],DataCopyExtParams{1,uint32_t(active)*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        PipeBarrier<PIPE_ALL>();Cast(u,b,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Mins(g,g,10.0f,count);Mins(u,u,10.0f,count);PipeBarrier<PIPE_V>();
        Maxs(u,u,-10.0f,count);Muls(t,g,-1.0f,count);PipeBarrier<PIPE_V>();
        Exp(t,t,count);PipeBarrier<PIPE_V>();Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();
        Div(g,g,t,count);PipeBarrier<PIPE_V>();Mul(g,g,u,count);PipeBarrier<PIPE_V>();
        for(uint16_t j=0;j<active;++j)Muls(g[j*K],g[j*K],prob.GetValue(j),K);
        PipeBarrier<PIPE_V>();Cast(b,g,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(row)*K],b,count);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_routed_act(void* stream,const void* hidden,const void* probability,
                               void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(hidden)||!aligned(probability)||!aligned(out))return -1;
    uint64_t rows=uint64_t(batch)*Q*6;
    if(overlap(out,rows*288*2,hidden,rows*576*2)||
       overlap(out,rows*288*2,probability,rows*4))return -1;
    uint32_t blocks=blocks_for(uint32_t((rows+7)/8));
    decode_routed_act_kernel<<<blocks,nullptr,stream>>>((uint8_t*)hidden,
        (uint8_t*)probability,(uint8_t*)out,uint32_t(rows),blocks);return 0;
}

// BF16 grouped down-projection -> FP32 local sum; TP reduction is external.
// inverse[B*6,6] maps original choice to grouped row, in [0,B*36).
// Sum in ascending GROUPED row order, not choice order (matches reference).
extern "C" __global__ __aicore__ void decode_routed_combine_kernel(
    GM_ADDR X,GM_ADDR I,GM_ADDR Y,uint32_t tokens,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t K=6,D=5120,T=2560;
    TPipe pipe;TBuf<TPosition::VECCALC> bi,bb,bf,ba;
    pipe.InitBuffer(bi,64);pipe.InitBuffer(bb,K*T*2);
    pipe.InitBuffer(bf,K*T*4);pipe.InitBuffer(ba,T*4);
    auto idx=bi.Get<int64_t>();auto b=bb.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto acc=ba.Get<float>();
    GlobalTensor<bfloat16_t> gx;GlobalTensor<int64_t> gi;GlobalTensor<float> gy;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gi.SetGlobalBuffer((__gm__ int64_t*)I);
    gy.SetGlobalBuffer((__gm__ float*)Y);
    for(uint32_t tile=GetBlockIdx();tile<tokens*2;tile+=blocks){
        uint32_t token=tile/2,col=(tile%2)*T;
        DataCopyPad(idx,gi[token*K],DataCopyExtParams{1,K*8,0,0,0},
                    DataCopyPadExtParams<int64_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t rows[K];for(int j=0;j<K;++j)rows[j]=idx.GetValue(j);
        for(int end=K-1;end>0;--end)for(int j=0;j<end;++j){
            int64_t lo=min(rows[j],rows[j+1]),hi=max(rows[j],rows[j+1]);
            rows[j]=lo;rows[j+1]=hi;
        }
        for(int j=0;j<K;++j)DataCopy(b[j*T],gx[uint64_t(rows[j])*D+col],T);
        PipeBarrier<PIPE_ALL>();Cast(f,b,RoundMode::CAST_NONE,K*T);PipeBarrier<PIPE_V>();
        Adds(acc,f,0.0f,T);PipeBarrier<PIPE_V>();
        for(int j=1;j<K;++j){Add(acc,acc,f[j*T],T);PipeBarrier<PIPE_V>();}
        PipeBarrier<PIPE_ALL>();DataCopy(gy[uint64_t(token)*D+col],acc,T);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_routed_combine(void* stream,const void* values,const void* inverse,
                                   void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(values)||!aligned(inverse)||!aligned(out))return -1;
    uint64_t tokens=uint64_t(batch)*Q;
    if(overlap(out,tokens*5120*4,values,tokens*6*5120*2)||
       overlap(out,tokens*5120*4,inverse,tokens*6*8))return -1;
    uint32_t blocks=blocks_for(uint32_t(tokens*2));
    decode_routed_combine_kernel<<<blocks,nullptr,stream>>>((uint8_t*)values,
        (uint8_t*)inverse,(uint8_t*)out,uint32_t(tokens),blocks);return 0;
}

// Engram host rows: INT8[B*18,256], FP32[B*18,8] block-32 scales.
// Output BF16[B*6,768]; rank owns three consecutive hash columns.
extern "C" __global__ __aicore__ void decode_engram_rows_kernel(
    GM_ADDR X,GM_ADDR S,GM_ADDR Y,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=256;
    TPipe pipe;TBuf<TPosition::VECCALC> bx,bh,bf,bs,by;
    pipe.InitBuffer(bx,D);pipe.InitBuffer(bh,D*2);pipe.InitBuffer(bf,D*4);
    pipe.InitBuffer(bs,32);pipe.InitBuffer(by,D*2);
    auto x=bx.Get<int8_t>();auto h=bh.Get<half>();auto f=bf.Get<float>();
    auto s=bs.Get<float>();auto y=by.Get<bfloat16_t>();
    GlobalTensor<int8_t> gx;GlobalTensor<float> gs;GlobalTensor<bfloat16_t> gy;
    gx.SetGlobalBuffer((__gm__ int8_t*)X);gs.SetGlobalBuffer((__gm__ float*)S);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        DataCopy(x,gx[uint64_t(row)*D],D);DataCopy(s,gs[uint64_t(row)*8],8);
        PipeBarrier<PIPE_ALL>();
        Cast(h,x,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Cast(f,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(int j=0;j<8;++j)Muls(f[j*32],f[j*32],s.GetValue(j),int32_t(32));
        PipeBarrier<PIPE_V>();Cast(y,f,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(row)*D],y,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_engram_rows_i8(void* stream,const void* values,
                                 const void* scales,void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(values)||!aligned(scales)||!aligned(out))return -1;
    uint32_t rows=batch*18,blocks=blocks_for(rows);
    if(overlap(out,uint64_t(rows)*512,values,uint64_t(rows)*256)||
       overlap(out,uint64_t(rows)*512,scales,uint64_t(rows)*32))return -1;
    decode_engram_rows_kernel<<<blocks,nullptr,stream>>>((uint8_t*)values,
        (uint8_t*)scales,(uint8_t*)out,rows,blocks);return 0;
}

// Local projection rounds to BF16 before FP32 TP sum. The gate rounds that
// sum to BF16 again, matching canonical Engram. Rotation is a separate FP32
// fixed Matmul plan; this preparation only widens hidden/projection storage.
extern "C" __global__ __aicore__ void decode_engram_widen_kernel(
    GM_ADDR X,GM_ADDR Y,uint32_t tiles,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t N=256;
    TPipe p;TBuf<TPosition::VECCALC> bh,bf;
    p.InitBuffer(bh,N*2);p.InitBuffer(bf,N*4);
    auto h=bh.Get<bfloat16_t>();auto f=bf.Get<float>();
    GlobalTensor<bfloat16_t> gx;GlobalTensor<float> gy;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gy.SetGlobalBuffer((__gm__ float*)Y);
    for(uint32_t i=GetBlockIdx();i<tiles;i+=blocks){
        DataCopy(h,gx[uint64_t(i)*N],N);PipeBarrier<PIPE_ALL>();
        Cast(f,h,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(i)*N],f,N);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_engram_widen(void* stream,void* x,void* y,uint32_t batch,uint32_t width){
    if(!stream||!batch||batch>MAX_BATCH||(width!=20480&&width!=25600)||
       !aligned(x)||!aligned(y))return -1;
    uint64_t n=uint64_t(batch)*Q*width;
    if(overlap(x,n*2,y,n*4))return -1;
    uint32_t tiles=uint32_t(n/256),blocks=blocks_for(tiles);
    decode_engram_widen_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)y,tiles,blocks);
    return 0;
}
// rot FP32[B*6,4,5120] in original basis; kv FP32[B*6,25600] after TP SUM;
// weight FP32[4,5120] = q_weight*k_weight. output may equal hidden exactly.
extern "C" __global__ __aicore__ void decode_engram_gate_kernel(
    GM_ADDR X,GM_ADDR ROT,GM_ADDR KV,GM_ADDR W,GM_ADDR OUT,
    uint32_t rows,float eps,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> ba,bk,bw,bs,bt,bh,br,bg;
    p.InitBuffer(ba,D*4);p.InitBuffer(bk,D*4);p.InitBuffer(bw,D*4);
    p.InitBuffer(bs,D*4);p.InitBuffer(bt,D*4);p.InitBuffer(bh,D*2);
    p.InitBuffer(br,128);p.InitBuffer(bg,128);
    auto a=ba.Get<float>();auto k=bk.Get<float>();auto w=bw.Get<float>();
    auto sq=bs.Get<float>();auto tmp=bt.Get<float>();auto h=bh.Get<bfloat16_t>();
    auto red=br.Get<float>();auto g=bg.Get<float>();
    GlobalTensor<float> gr,gkv,gw;GlobalTensor<bfloat16_t> gx,go;
    gr.SetGlobalBuffer((__gm__ float*)ROT);gkv.SetGlobalBuffer((__gm__ float*)KV);
    gw.SetGlobalBuffer((__gm__ float*)W);gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)OUT);
    for(uint32_t row=GetBlockIdx();row<rows*4;row+=blocks){
        uint64_t off=uint64_t(row)*D,ko=uint64_t(row/4)*5*D+(row%4)*D;
        DataCopy(a,gr[off],D);DataCopy(k,gkv[ko],D);DataCopy(w,gw[(row%4)*D],D);
        PipeBarrier<PIPE_ALL>();
        Cast(h,k,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(k,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Mul(sq,a,a,D);PipeBarrier<PIPE_V>();ReduceSum(red,sq,tmp,D);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float hs=red.GetValue(0)/float(D)+eps;
        Mul(sq,k,k,D);PipeBarrier<PIPE_V>();ReduceSum(red,sq,tmp,D);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float ks=red.GetValue(0)/float(D)+eps;
        Mul(a,a,w,D);PipeBarrier<PIPE_V>();Mul(a,a,k,D);PipeBarrier<PIPE_V>();
        ReduceSum(red,a,tmp,D);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float dot=red.GetValue(0);
        Duplicate(g,hs,int32_t(8));Duplicate(g[8],ks,int32_t(8));PipeBarrier<PIPE_V>();
        Sqrt(g,g,int32_t(16));Duplicate(red,1.0f,int32_t(16));PipeBarrier<PIPE_V>();
        Div(g,red,g,int32_t(16));PipeBarrier<PIPE_V>();
        Mul(g,g,g[8],int32_t(8));PipeBarrier<PIPE_V>();
        Muls(g,g,dot,int32_t(8));PipeBarrier<PIPE_V>();
        Muls(g,g,0.013975424859373685f,int32_t(8));PipeBarrier<PIPE_V>();
        Abs(g,g,int32_t(8));PipeBarrier<PIPE_V>();
        Maxs(g,g,1.e-6f,int32_t(8));PipeBarrier<PIPE_V>();
        Sqrt(g,g,int32_t(8));PipeBarrier<PIPE_V>();
        Muls(g,g,dot<0.0f?1.0f:-1.0f,int32_t(8));PipeBarrier<PIPE_V>();
        Exp(g,g,int32_t(8));PipeBarrier<PIPE_V>();Adds(g,g,1.0f,int32_t(8));PipeBarrier<PIPE_V>();
        Div(g,red,g,int32_t(8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float gate=g.GetValue(0);
        DataCopy(k,gkv[uint64_t(row/4)*5*D+4*D],D);PipeBarrier<PIPE_ALL>();
        Cast(h,k,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(k,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        // Wait for vector reads before MTE2 reuses the BF16 staging buffer.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(h,gx[off],D);PipeBarrier<PIPE_ALL>();
        Cast(a,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Muls(k,k,gate,D);PipeBarrier<PIPE_V>();Add(a,a,k,D);PipeBarrier<PIPE_V>();
        Cast(h,a,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(go[off],h,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_engram_gate(void* stream,void* hidden,void* rot,void* kv,
    void* weight,void* out,uint32_t batch,float eps){
    if(!stream||!batch||batch>MAX_BATCH||!(eps>0&&eps<=FLT_MAX))return -1;
    void* ptrs[]={hidden,rot,kv,weight,out};
    uint64_t rows=uint64_t(batch)*Q,sizes[]={rows*40960,rows*81920,rows*102400,81920,rows*40960};
    for(int i=0;i<5;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=0;i<4;++i)if((i!=0||out!=hidden)&&overlap(out,sizes[4],ptrs[i],sizes[i]))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows*4));
    decode_engram_gate_kernel<<<blocks,nullptr,stream>>>((uint8_t*)hidden,(uint8_t*)rot,
        (uint8_t*)kv,(uint8_t*)weight,(uint8_t*)out,uint32_t(rows),eps,blocks);
    return 0;
}


// DSpark-only fixed Q6 storage; row five is padding, not a sixth proposal.

extern "C" __global__ __aicore__ void ds_route_kernel(
    GM_ADDR L,GM_ADDR B,GM_ADDR I,GM_ADDR P,uint32_t rows,float inv_temp,
    float scale,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t E=128,N=128,K=3;
    TPipe pipe;TBuf<TPosition::VECCALC> bl,bb,bs,bt,bi,ba,bw,bo,bp;
    pipe.InitBuffer(bl,N*4);pipe.InitBuffer(bb,N*4);pipe.InitBuffer(bs,N*4);
    pipe.InitBuffer(bt,N*4);pipe.InitBuffer(bi,N*4);pipe.InitBuffer(ba,N*8);
    pipe.InitBuffer(bw,N*8);pipe.InitBuffer(bo,64);pipe.InitBuffer(bp,32);
    auto l=bl.Get<float>();auto bias=bb.Get<float>();auto score=bs.Get<float>();
    auto tmp=bt.Get<float>();auto ids=bi.Get<uint32_t>();auto sorted=ba.Get<float>();
    auto work=bw.Get<float>();auto out=bo.Get<int64_t>();auto prob=bp.Get<float>();
    GlobalTensor<float> gl,gb,gp;GlobalTensor<int64_t> gi;
    gl.SetGlobalBuffer((__gm__ float*)L);gb.SetGlobalBuffer((__gm__ float*)B);
    gp.SetGlobalBuffer((__gm__ float*)P);gi.SetGlobalBuffer((__gm__ int64_t*)I);
    DataCopy(bias,gb,E);PipeBarrier<PIPE_ALL>();
    CreateVecIndex(ids.ReinterpretCast<int32_t>(),int32_t(0),uint32_t(N));
    PipeBarrier<PIPE_ALL>();
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        DataCopy(l,gl[uint64_t(row)*E],E);PipeBarrier<PIPE_ALL>();
        Muls(l,l,inv_temp,E);PipeBarrier<PIPE_V>();
        // Stable softplus, avoiding overflow in exp for large positive logits.
        Abs(tmp,l,E);PipeBarrier<PIPE_V>();Muls(tmp,tmp,-1.0f,E);PipeBarrier<PIPE_V>();
        Exp(tmp,tmp,E);Maxs(score,l,0.0f,E);PipeBarrier<PIPE_V>();
        // Compensate log(1+t): when t is tiny, FP32 1+t rounds to 1.
        Adds(l,tmp,1.0f,E);PipeBarrier<PIPE_V>();
        Adds(work,l,-1.0f,E);PipeBarrier<PIPE_V>();
        Sub(work,tmp,work,E);PipeBarrier<PIPE_V>();
        Div(work,work,l,E);Ln(tmp,l,E);PipeBarrier<PIPE_V>();
        Add(tmp,tmp,work,E);PipeBarrier<PIPE_V>();
        Add(score,score,tmp,E);PipeBarrier<PIPE_V>();Sqrt(score,score,E);PipeBarrier<PIPE_V>();
        Duplicate(l,-FLT_MAX,N);PipeBarrier<PIPE_V>();Add(l,score,bias,E);PipeBarrier<PIPE_V>();
        Sort<float,true>(sorted,l,ids,work,int32_t(N/32));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float total=0.0f;
        for(int j=0;j<K;++j){
            uint32_t id=sorted.ReinterpretCast<uint32_t>().GetValue(2*j+1);
            out.SetValue(j,int64_t(id));float v=score.GetValue(id);prob.SetValue(j,v);total+=v;
        }
        for(int j=0;j<K;++j)prob.SetValue(j,prob.GetValue(j)/(total+1e-20f)*scale);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gi[uint64_t(row)*K],out,DataCopyExtParams{1,K*8,0,0,0});
        DataCopyPad(gp[uint64_t(row)*K],prob,DataCopyExtParams{1,K*4,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_route(void* stream,const void* logits,const void* bias,
                          void* ids,void* probabilities,uint32_t batch,
                          float temperature,float scale){
    if(!stream||!batch||batch>MAX_BATCH||!(temperature>0&&temperature<=FLT_MAX)
       ||!(1.0f/temperature<=FLT_MAX)||!(scale>0&&scale<=FLT_MAX))return -1;
    const void* ptrs[]={logits,bias,ids,probabilities};uint64_t rows=uint64_t(batch)*Q;
    uint64_t sizes[]={rows*128*4,128*4,rows*3*8,rows*3*4};
    for(int i=0;i<4;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=2;i<4;++i)for(int j=0;j<i;++j)
        if(overlap(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows));
    ds_route_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)bias,
        (uint8_t*)ids,(uint8_t*)probabilities,uint32_t(rows),1.0f/temperature,scale,blocks);
    return 0;
}


extern "C" __global__ __aicore__ void ds_dispatch_kernel(
    GM_ADDR X,GM_ADDR I,GM_ADDR P,GM_ADDR Y,GM_ADDR W,GM_ADDR V,GM_ADDR E,
    uint32_t pairs,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120,EX=128,CAP=MAX_BATCH*Q*3;
    TPipe pipe;TBuf<TPosition::VECCALC> bi,bp,bx,be,bv,bw,bc;
    pipe.InitBuffer(bi,((CAP*8+31)/32)*32);
    pipe.InitBuffer(bp,((CAP*4+31)/32)*32);
    pipe.InitBuffer(bx,D*2);pipe.InitBuffer(be,EX*8);
    pipe.InitBuffer(bv,((CAP*8+31)/32)*32);
    pipe.InitBuffer(bw,((CAP*4+31)/32)*32);
    auto inverse=bv.Get<int64_t>();auto weights=bw.Get<float>();
    pipe.InitBuffer(bc,EX*8);auto cursor=bc.Get<int64_t>();
    auto ids=bi.Get<int64_t>();auto prob=bp.Get<float>();
    auto x=bx.Get<bfloat16_t>();auto ends=be.Get<int64_t>();
    GlobalTensor<int64_t> gi,gv,ge;GlobalTensor<float> gp,gw;
    GlobalTensor<bfloat16_t> gx,gy;
    gi.SetGlobalBuffer((__gm__ int64_t*)I);gv.SetGlobalBuffer((__gm__ int64_t*)V);
    ge.SetGlobalBuffer((__gm__ int64_t*)E);gp.SetGlobalBuffer((__gm__ float*)P);
    gw.SetGlobalBuffer((__gm__ float*)W);gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    DataCopyPad(ids,gi,DataCopyExtParams{1,pairs*8,0,0,0},
                DataCopyPadExtParams<int64_t>{false,0,0,0});
    // Only block 0 publishes metadata; other blocks need IDs only.
    if(GetBlockIdx()==0){
        DataCopyPad(prob,gp,DataCopyExtParams{1,pairs*4,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
    }
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    // Build cumulative ends, stable inverse and probability exactly once.
    if(GetBlockIdx()==0){
        for(int32_t e=0;e<EX;++e)ends.SetValue(e,0);
        for(uint32_t j=0;j<pairs;++j){
            int32_t e=int32_t(ids.GetValue(j));
            ends.SetValue(e,ends.GetValue(e)+1);
        }
        int64_t total=0;
        for(int32_t e=0;e<EX;++e){
            cursor.SetValue(e,total);total+=ends.GetValue(e);ends.SetValue(e,total);
        }
        for(uint32_t j=0;j<pairs;++j){
            int32_t e=int32_t(ids.GetValue(j));int64_t dst=cursor.GetValue(e);
            cursor.SetValue(e,dst+1);inverse.SetValue(j,dst);
            weights.SetValue(dst,prob.GetValue(j));
        }
        // Single-owner DMA avoids cross-core scalar cache-line collisions.
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(ge,ends,EX);
        DataCopyPad(gv,inverse,DataCopyExtParams{1,pairs*8,0,0,0});
        DataCopyPad(gw,weights,DataCopyExtParams{1,pairs*4,0,0,0});
    }
    for(uint32_t j=GetBlockIdx();j<pairs;j+=blocks){
        uint32_t dst;
        if(GetBlockIdx()==0){
            dst=uint32_t(inverse.GetValue(j));
        }else{
            // Stable rank: smaller expert IDs, then earlier equal-ID pairs.
            // Local IDs only: no cross-core dependency on the metadata DMA.
            int32_t e=int32_t(ids.GetValue(j));dst=0;
            for(uint32_t k=0;k<pairs;++k){
                int32_t other=int32_t(ids.GetValue(k));
                dst+=uint32_t(other<e||(other==e&&k<j));
            }
        }
        DataCopy(x,gx[uint64_t(j/3)*D],D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(dst)*D],x,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_dispatch(void* stream,const void* x,const void* ids,
    const void* probability,void* rows,void* sorted_probability,void* inverse,
    void* ends,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH)return -1;
    const void* ptrs[]={x,ids,probability,rows,sorted_probability,inverse,ends};
    uint64_t tokens=batch*Q,pairs=tokens*3;
    uint64_t sizes[]={tokens*5120*2,pairs*8,pairs*4,pairs*5120*2,pairs*4,pairs*8,128*8};
    for(int i=0;i<7;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=3;i<7;++i)for(int j=0;j<i;++j)
        if(overlap(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;
    uint32_t blocks=blocks_for(uint32_t(pairs));
    ds_dispatch_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)ids,
        (uint8_t*)probability,(uint8_t*)rows,(uint8_t*)sorted_probability,
        (uint8_t*)inverse,(uint8_t*)ends,uint32_t(pairs),blocks);return 0;
}


extern "C" int dec_ds_routed_act(void* stream,const void* hidden,const void* probability,
                               void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(hidden)||!aligned(probability)||!aligned(out))return -1;
    uint64_t rows=uint64_t(batch)*Q*3;
    if(overlap(out,rows*288*2,hidden,rows*576*2)||
       overlap(out,rows*288*2,probability,rows*4))return -1;
    uint32_t blocks=blocks_for(uint32_t((rows+7)/8));
    decode_routed_act_kernel<<<blocks,nullptr,stream>>>((uint8_t*)hidden,
        (uint8_t*)probability,(uint8_t*)out,uint32_t(rows),blocks);return 0;
}


extern "C" __global__ __aicore__ void ds_combine_kernel(
    GM_ADDR X,GM_ADDR I,GM_ADDR Y,uint32_t tokens,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t K=3,D=5120,T=2560;
    TPipe pipe;TBuf<TPosition::VECCALC> bi,bb,bf,ba;
    pipe.InitBuffer(bi,64);pipe.InitBuffer(bb,K*T*2);
    pipe.InitBuffer(bf,K*T*4);pipe.InitBuffer(ba,T*4);
    auto idx=bi.Get<int64_t>();auto b=bb.Get<bfloat16_t>();
    auto f=bf.Get<float>();auto acc=ba.Get<float>();
    GlobalTensor<bfloat16_t> gx;GlobalTensor<int64_t> gi;GlobalTensor<float> gy;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);gi.SetGlobalBuffer((__gm__ int64_t*)I);
    gy.SetGlobalBuffer((__gm__ float*)Y);
    for(uint32_t tile=GetBlockIdx();tile<tokens*2;tile+=blocks){
        uint32_t token=tile/2,col=(tile%2)*T;
        DataCopyPad(idx,gi[token*K],DataCopyExtParams{1,K*8,0,0,0},
                    DataCopyPadExtParams<int64_t>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t rows[K];for(int j=0;j<K;++j)rows[j]=idx.GetValue(j);
        for(int end=K-1;end>0;--end)for(int j=0;j<end;++j){
            int64_t lo=min(rows[j],rows[j+1]),hi=max(rows[j],rows[j+1]);
            rows[j]=lo;rows[j+1]=hi;
        }
        for(int j=0;j<K;++j)DataCopy(b[j*T],gx[uint64_t(rows[j])*D+col],T);
        PipeBarrier<PIPE_ALL>();Cast(f,b,RoundMode::CAST_NONE,K*T);PipeBarrier<PIPE_V>();
        Adds(acc,f,0.0f,T);PipeBarrier<PIPE_V>();
        for(int j=1;j<K;++j){Add(acc,acc,f[j*T],T);PipeBarrier<PIPE_V>();}
        PipeBarrier<PIPE_ALL>();DataCopy(gy[uint64_t(token)*D+col],acc,T);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_combine(void* stream,const void* values,const void* inverse,
                                   void* out,uint32_t batch){
    if(!stream||!batch||batch>MAX_BATCH||!aligned(values)||!aligned(inverse)||!aligned(out))return -1;
    uint64_t tokens=uint64_t(batch)*Q;
    if(overlap(out,tokens*5120*4,values,tokens*3*5120*2)||
       overlap(out,tokens*5120*4,inverse,tokens*3*8))return -1;
    uint32_t blocks=blocks_for(uint32_t(tokens*2));
    ds_combine_kernel<<<blocks,nullptr,stream>>>((uint8_t*)values,
        (uint8_t*)inverse,(uint8_t*)out,uint32_t(tokens),blocks);return 0;
}


constexpr uint32_t DS_MAXK=15360;
extern "C" __global__ __aicore__ void ds_w8_expand_kernel(GM_ADDR w, GM_ADDR s, GM_ADDR y,
                                                 uint32_t rows, uint32_t k) {
    const uint32_t cores = GetBlockNum();
    const uint32_t me    = GetBlockIdx();
    if (me >= cores) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bw, bh, bf, bs;
    pipe.InitBuffer(bw, DS_MAXK * sizeof(int8_t));
    pipe.InitBuffer(bh, DS_MAXK * sizeof(half));
    pipe.InitBuffer(bf, DS_MAXK * sizeof(float));
    pipe.InitBuffer(bs, 32);

    LocalTensor<int8_t> lw = bw.Get<int8_t>();
    LocalTensor<half>   lh = bh.Get<half>();
    LocalTensor<float>  lf = bf.Get<float>();
    LocalTensor<float>  ls = bs.Get<float>();
    LocalTensor<bfloat16_t> lb = lh.ReinterpretCast<bfloat16_t>();

    GlobalTensor<int8_t>     gw; gw.SetGlobalBuffer((__gm__ int8_t*)w);
    GlobalTensor<float>      gs; gs.SetGlobalBuffer((__gm__ float*)s);
    GlobalTensor<bfloat16_t> gy; gy.SetGlobalBuffer((__gm__ bfloat16_t*)y);

    DataCopyExtParams sp; sp.blockCount = 1; sp.blockLen = sizeof(float);
    sp.srcStride = 0; sp.dstStride = 0;
    DataCopyPadExtParams<float> pp; pp.isPad = false; pp.leftPadding = 0;
    pp.rightPadding = 0; pp.paddingValue = 0;

    for (uint32_t row = me; row < rows; row += cores) {
        DataCopy(lw, gw[(uint64_t)row * k], k);
        DataCopyPad(ls, gs[row], sp, pp);
        PipeBarrier<PIPE_ALL>();
        Cast(lh, lw, RoundMode::CAST_NONE, k);
        PipeBarrier<PIPE_ALL>();
        Cast(lf, lh, RoundMode::CAST_NONE, k);
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        Muls(lf, lf, ls.GetValue(0), int32_t(k));
        PipeBarrier<PIPE_ALL>();
        Cast(lb, lf, RoundMode::CAST_RINT, k);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gy[(uint64_t)row * k], lb, k);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int dec_ds_w8_expand(void* stream,void* w,void* s,void* y,uint32_t n,uint32_t k) {
    if(!stream||!w||!s||!y||!n||!k||k>DS_MAXK||k%32) return -1;
    if((uintptr_t(w)|uintptr_t(s)|uintptr_t(y))&31) return -1;
    uint64_t wn=uint64_t(n)*k, sn=uint64_t(n)*4, yn=wn*2;
    uintptr_t a=uintptr_t(y), b=uintptr_t(w), c=uintptr_t(s);
    if((a<=b ? b-a<yn : a-b<wn)||(a<=c ? c-a<yn : a-c<sn)) return -1;
    uint32_t blocks=n<40?n:40;
    ds_w8_expand_kernel<<<blocks,nullptr,stream>>>((uint8_t*)w,(uint8_t*)s,(uint8_t*)y,n,k);
    return 0;
}


// result[B,8] = slot, verify_start, accepted, bonus_token, active, error, 0, 0.
// Features correspond to predictions: row q seeds at verify_start+q+1.
// Only accepted rows seed canonical history. Proposal KV never enters Past.
__aicore__ inline bool ds_live(GlobalTensor<int64_t>& r,uint32_t b,uint32_t B,uint32_t slots){
    int64_t s=r.GetValue(b*8),p=r.GetValue(b*8+1),a=r.GetValue(b*8+2);
    if(!r.GetValue(b*8+4)||r.GetValue(b*8+5)||s<0||s>=slots||p<0||p>INT64_MAX-16||a<1||a>6)return false;
    for(uint32_t j=0;j<B;++j)if(j!=b&&r.GetValue(j*8+4)&&r.GetValue(j*8)==s)return false;
    return true;
}
extern "C" __global__ __aicore__ void ds_prepare_kernel(GM_ADDR R,GM_ADDR INV,GM_ADDR SF,GM_ADDR DF,GM_ADDR PRE,GM_ADDR IDS,uint32_t B,uint32_t slots){
    if(GetBlockIdx()>=GetBlockNum())return;
    GlobalTensor<int64_t> r,ids;r.SetGlobalBuffer((__gm__ int64_t*)R);ids.SetGlobalBuffer((__gm__ int64_t*)IDS);
    GlobalTensor<float> inv,sf,df,pre;inv.SetGlobalBuffer((__gm__ float*)INV);sf.SetGlobalBuffer((__gm__ float*)SF);df.SetGlobalBuffer((__gm__ float*)DF);pre.SetGlobalBuffer((__gm__ float*)PRE);
    TPipe p;TBuf<TPosition::VECCALC> bi,ba,bc,bs,bo,bp,bid;p.InitBuffer(bi,128);p.InitBuffer(ba,128);p.InitBuffer(bc,128);p.InitBuffer(bs,128);p.InitBuffer(bo,256);p.InitBuffer(bp,32);p.InitBuffer(bid,64);
    auto i=bi.Get<float>(),a=ba.Get<float>(),c=bc.Get<float>(),s=bs.Get<float>(),o=bo.Get<float>(),pr=bp.Get<float>();auto id=bid.Get<int64_t>();
    DataCopy(i,inv,32);PipeBarrier<PIPE_ALL>();
    for(uint32_t row=GetBlockIdx();row<B*6;row+=GetBlockNum()){
        uint32_t b=row/6,q=row%6;bool live=ds_live(r,b,B,slots);
        int64_t start=live?r.GetValue(b*8+1):0,acc=live?r.GetValue(b*8+2):0;
        for(int k=0;k<2;++k){
            float pos=live?float(start+(k?acc+q+1:q+1)):0.0f;
            Muls(a,i,pos,int32_t(32));PipeBarrier<PIPE_V>();Cos(c,a,32);Sin(s,a,32);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for(int j=0;j<32;++j){o.SetValue(j*2,c.GetValue(j));o.SetValue(j*2+1,s.GetValue(j));}
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            if(k)DataCopy(df[row*64],o,64);else DataCopy(sf[row*64],o,64);PipeBarrier<PIPE_ALL>();
        }
        for(int j=0;j<4;++j)pr.SetValue(j,live&&q<5&&j==0?1.0f:0.0f);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(pre[row*4],pr,DataCopyExtParams{1,16,0,0,0});PipeBarrier<PIPE_ALL>();
        if(q==0){for(int j=0;j<6;++j)id.SetValue(j,live?(j?128799:r.GetValue(b*8+3)):-1);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(ids[b*6],id,DataCopyExtParams{1,48,0,0,0});PipeBarrier<PIPE_ALL>();}
    }
}
extern "C" int dec_ds_prepare(void* stream,void* result,void* inv,void* seed_freq,void* draft_freq,void* pre,void* ids,uint32_t B,uint32_t slots){
    if(!stream||!B||B>MAX_BATCH||slots<B)return -1;
    void* p[]={result,inv,seed_freq,draft_freq,pre,ids};for(auto x:p)if(!aligned(x))return -1;
    ds_prepare_kernel<<<blocks_for(B*6),nullptr,stream>>>((uint8_t*)result,(uint8_t*)inv,(uint8_t*)seed_freq,(uint8_t*)draft_freq,(uint8_t*)pre,(uint8_t*)ids,B,slots);return 0;
}
// A single owner per request writes all accepted rows; rejected/inactive bytes untouched.
extern "C" __global__ __aicore__ void ds_seed_kernel(GM_ADDR KV,GM_ADDR R,GM_ADDR H,uint32_t B,uint32_t slots,uint32_t ring,uint32_t pad){
    if(GetBlockIdx()>=GetBlockNum())return;
    GlobalTensor<int64_t> r;r.SetGlobalBuffer((__gm__ int64_t*)R);
    GlobalTensor<bfloat16_t> kv,h;kv.SetGlobalBuffer((__gm__ bfloat16_t*)KV);h.SetGlobalBuffer((__gm__ bfloat16_t*)H);
    TPipe p;TBuf<TPosition::VECCALC> bv;p.InitBuffer(bv,1024);auto v=bv.Get<bfloat16_t>();
    for(uint32_t b=GetBlockIdx();b<B;b+=GetBlockNum())if(ds_live(r,b,B,slots)){
        int64_t slot=r.GetValue(b*8),start=r.GetValue(b*8+1),n=r.GetValue(b*8+2);
        for(int64_t q=0;q<n;++q){DataCopy(v,kv[(b*6+q)*512],512);PipeBarrier<PIPE_ALL>();
            DataCopy(h[(slot*(ring+pad)+pad+(start+q+1)%ring)*512],v,512);PipeBarrier<PIPE_ALL>();}
    }
}
extern "C" int dec_ds_seed(void* stream,void* kv,void* result,void* ring_data,uint32_t B,uint32_t slots,uint32_t ring,uint32_t pad){
    if(!stream||!B||B>MAX_BATCH||slots<B||ring<134||!aligned(kv)||!aligned(result)||!aligned(ring_data))return -1;
    ds_seed_kernel<<<B,nullptr,stream>>>((uint8_t*)kv,(uint8_t*)result,(uint8_t*)ring_data,B,slots,ring,pad);return 0;
}
// Fixed 128-history + five provisional rows, all five queries see the same set.
// FP32 score and online softmax, sink contributes only to denominator.
// Draft attention: fixed 32-key UB tiles, no GM scratch or host-length branch.
// Keep all five proposal keys visible to each of the five live query rows;
// the sixth query row and invalid/duplicate receipt slots produce exact zeros.
namespace {
template<HardEvent E> __aicore__ inline void ds_fence() {
    SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0);
}
template<int G> struct DraftAttentionTile {
    static constexpr int K = 32;
    static constexpr int D = 512;
    static constexpr int NF = 2*G*D + 2*K*D + K*8 + 64 + K*8 + 32;
    TPipe pipe;
    TBuf<TPosition::VECCALC> fb, bb;
    LocalTensor<float> q, acc, kv, work, part, score, broad, stat;
    LocalTensor<bfloat16_t> bf;
    __aicore__ inline DraftAttentionTile() {
        pipe.InitBuffer(fb, NF*4);
        pipe.InitBuffer(bb, K*D*2);
        q = fb.Get<float>(); acc = q[G*D]; kv = acc[G*D];
        work = kv[K*D]; part = work[K*D]; score = part[K*8];
        broad = score[64]; stat = broad[K*8];
        bf = bb.Get<bfloat16_t>();
    }
    // Queries of one (b,t) head group are contiguous BF16[H,512] rows.
    __aicore__ inline void load_query(__gm__ bfloat16_t* p, int ge) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        DataCopy(bf, g, ge*D);
        ds_fence<HardEvent::MTE2_V>();
        Cast(q, bf, RoundMode::CAST_NONE, int32_t(ge*D));
        PipeBarrier<PIPE_V>();
        ds_fence<HardEvent::V_MTE2>(); // BF16 staging may now be overwritten.
    }
    __aicore__ inline void load_run(__gm__ bfloat16_t* p, int offset, int rows) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        DataCopy(bf[offset*D], g, int32_t(rows*D));
    }
    // One cast per tile, shared by every head in the group.
    __aicore__ inline void cast_tile(int n) {
        ds_fence<HardEvent::MTE2_V>();
        Cast(kv, bf, RoundMode::CAST_NONE, int32_t(n*D));
        PipeBarrier<PIPE_V>();
        ds_fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void consume(int n, int g, float& mx, float& den) {
        for(int c=0; c<D; c+=64)
            Mul(work[c], kv[c], q[g*D+c], uint64_t(64), uint8_t(n),
                BinaryRepeatParams(1,1,1,64,64,0));
        PipeBarrier<PIPE_V>();
        for(int s=0;s<n;s+=16){ int m=n-s; if(m>16)m=16;
            WholeReduceSum(part[s*8], work[s*512], int32_t(64), int32_t(m*8), 1, 1, 8); }
        PipeBarrier<PIPE_V>();
        Duplicate(score, -__builtin_inff(), int32_t(64));
        PipeBarrier<PIPE_V>();
        WholeReduceSum(score, part, int32_t(8), int32_t(n), 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Muls(score, score, 0.04419417382415922f, int32_t(K));
        PipeBarrier<PIPE_V>();
        WholeReduceMax(stat, score, int32_t(n), 1, 1, 1, 8,
                       ReduceOrder::ORDER_ONLY_VALUE);
        ds_fence<HardEvent::V_S>();
        float next = max(mx, stat.GetValue(0));
        ds_fence<HardEvent::S_V>();
        Adds(score, score, -next, int32_t(K));
        Duplicate(score[32], mx==next ? 0.f : mx-next, int32_t(8));
        PipeBarrier<PIPE_V>();
        Exp(score, score, int32_t(64));
        PipeBarrier<PIPE_V>();
        WholeReduceSum(stat[8], score, int32_t(n), 1, 1, 1, 8);
        Brcb(broad, score, uint8_t(K/8), BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        ds_fence<HardEvent::V_S>();
        float alpha = mx==next ? 1.f : score.GetValue(32);
        float tile_den = stat.GetValue(8);
        ds_fence<HardEvent::S_V>();
        if(n<K) Duplicate(work[n*D], 0.f, int32_t((K-n)*D));
        for(int c=0; c<D; c+=64)
            Mul(work[c], kv[c], broad, uint64_t(64), uint8_t(n),
                BinaryRepeatParams(1,1,0,64,64,1));
        PipeBarrier<PIPE_V>();
        for(int rows=K/2; rows>0; rows/=2) {
            Add(work, work, work[rows*D], int32_t(rows*D));
            PipeBarrier<PIPE_V>();
        }
        Muls(acc[g*D], acc[g*D], alpha, int32_t(D));
        PipeBarrier<PIPE_V>();
        Add(acc[g*D], acc[g*D], work, int32_t(D));
        PipeBarrier<PIPE_V>();
        den = den*alpha + tile_den;
        mx = next;
    }
    __aicore__ inline void store(__gm__ bfloat16_t* p, int ge) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        PipeBarrier<PIPE_ALL>();
        Cast(bf, acc, RoundMode::CAST_RINT, int32_t(ge*D));
        PipeBarrier<PIPE_ALL>(); DataCopy(g, bf, ge*D); PipeBarrier<PIPE_ALL>();
    }
};

}
extern "C" __global__ __aicore__ void ds_attention_kernel(GM_ADDR QRY,GM_ADDR KV,GM_ADDR H,GM_ADDR R,GM_ADDR SINK,GM_ADDR OUT,uint32_t B,uint32_t slots,uint32_t ring,uint32_t pad,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    GlobalTensor<int64_t> receipt; receipt.SetGlobalBuffer((__gm__ int64_t*)R);
    GlobalTensor<float> sink; sink.SetGlobalBuffer((__gm__ float*)SINK);
    DraftAttentionTile<1> v;
    for(uint32_t row=GetBlockIdx();row<B*48;row+=blocks){
        uint32_t b=row/48, qq=(row/8)%6, head=row%8;
        Duplicate(v.acc,0.f,int32_t(512)); PipeBarrier<PIPE_V>();
        if(ds_live(receipt,b,B,slots) && qq<5){
            int64_t slot=receipt.GetValue(b*8);
            int64_t end=receipt.GetValue(b*8+1)+receipt.GetValue(b*8+2)+1;
            v.load_query((__gm__ bfloat16_t*)QRY+uint64_t(row)*512,1);
            float mx=sink.GetValue(head), den=1.f;
            // Compact only the negative-position prefix; preserve chronological
            // ring ordering and the original five non-causal proposal entries.
            int first=end<128 ? int(128-end) : 0;
            for(int tile=first;tile<133;tile+=32){
                int n=0,run_rows=0,run_offset=0;
                __gm__ bfloat16_t* run=nullptr;
                int stop=tile+32; if(stop>133)stop=133;
                for(int j=tile;j<stop;++j){
                    int64_t pos=end-128+j;
                    __gm__ bfloat16_t* key=j<128
                        ? (__gm__ bfloat16_t*)H+(slot*(ring+pad)+pad+pos%ring)*512
                        : (__gm__ bfloat16_t*)KV+(b*6+j-128)*512;
                    if(run_rows && key!=run+run_rows*512){
                        v.load_run(run,run_offset,run_rows);run_rows=0;
                    }
                    if(!run_rows){run=key;run_offset=n;}
                    ++run_rows;++n;
                }
                if(run_rows)v.load_run(run,run_offset,run_rows);
                v.cast_tile(n);v.consume(n,0,mx,den);
            }
            Muls(v.acc,v.acc,1.f/den,int32_t(512));PipeBarrier<PIPE_V>();
        }
        v.store((__gm__ bfloat16_t*)OUT+uint64_t(row)*512,1);
    }
}
extern "C" int dec_ds_attention(void* stream,void* q,void* kv,void* history,void* result,void* sink,void* out,uint32_t B,uint32_t slots,uint32_t ring,uint32_t pad){
    if(!stream||!B||B>MAX_BATCH||slots<B||ring<134)return -1;void* p[]={q,kv,history,result,sink,out};for(auto x:p)if(!aligned(x))return -1;
    uint32_t blocks=blocks_for(B*48);ds_attention_kernel<<<blocks,nullptr,stream>>>((uint8_t*)q,(uint8_t*)kv,(uint8_t*)history,(uint8_t*)result,(uint8_t*)sink,(uint8_t*)out,B,slots,ring,pad,blocks);return 0;
}
// Sharded lookup. mode=0: target embedding -> four identical HC lanes,
// ids = bonus,noise,noise,noise,noise; sixth compute row zero.
// mode=1: previous Markov chain token -> replicated padded-six 256-wide rows.
extern "C" __global__ __aicore__ void ds_lookup_kernel(GM_ADDR IDS,GM_ADDR W,GM_ADDR Y,uint32_t B,uint32_t rank,uint32_t step,uint32_t mode){
    if(GetBlockIdx()>=GetBlockNum())return;
    GlobalTensor<int64_t> ids;GlobalTensor<bfloat16_t> w,y;ids.SetGlobalBuffer((__gm__ int64_t*)IDS);w.SetGlobalBuffer((__gm__ bfloat16_t*)W);y.SetGlobalBuffer((__gm__ bfloat16_t*)Y);
    TPipe p;TBuf<TPosition::VECCALC> bv;p.InitBuffer(bv,10240);auto v=bv.Get<bfloat16_t>();
    int32_t d=mode?256:5120;
    for(uint32_t row=GetBlockIdx();row<(mode?B:B*6);row+=GetBlockNum()){
        int64_t id=ids.GetValue((mode?row:row/6)*6+(mode?step:0));if(!mode&&row%6)id=id<0?-1:128799;
        id-=rank*16160;bool live=id>=0&&id<16160&&(mode||row%6<5);
        if(live)DataCopy(v,w[uint64_t(id)*d],d);else Duplicate(v,bfloat16_t(0.0f),d);
        PipeBarrier<PIPE_ALL>();for(int j=0;j<(mode?1:4);++j){DataCopy(y[(uint64_t(row)*(mode?1:4)+j)*d],v,d);PipeBarrier<PIPE_ALL>();}
    }
}
extern "C" int dec_ds_lookup(void* stream,void* ids,void* weight,void* out,uint32_t B,uint32_t rank,uint32_t step,uint32_t mode){
    if(!stream||!B||B>MAX_BATCH||rank>=8||step>=5||mode>1||!aligned(ids)||!aligned(weight)||!aligned(out))return -1;
    ds_lookup_kernel<<<blocks_for(mode?B:B*6),nullptr,stream>>>((uint8_t*)ids,(uint8_t*)weight,(uint8_t*)out,B,rank,step,mode);return 0;
}
// Preserve base logits. Only the chain step consumed by greedy is written.
extern "C" __global__ __aicore__ void ds_bias_kernel(GM_ADDR L,GM_ADDR MARKOV,GM_ADDR OUT,uint32_t rows,uint32_t step,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;GlobalTensor<float> l,m,o;l.SetGlobalBuffer((__gm__ float*)L);m.SetGlobalBuffer((__gm__ float*)MARKOV);o.SetGlobalBuffer((__gm__ float*)OUT);
    TPipe p;TBuf<TPosition::VECCALC> bl,bm;p.InitBuffer(bl,640);p.InitBuffer(bm,640);auto x=bl.Get<float>(),y=bm.Get<float>();
    for(uint32_t tile=GetBlockIdx();tile<rows*101;tile+=blocks){
        uint64_t compact=uint64_t(tile)*160;
        uint64_t off=(uint64_t(tile/101)*6+step)*16160+(tile%101)*160;
        DataCopy(x,l[off],160);DataCopy(y,m[compact],160);PipeBarrier<PIPE_ALL>();
        Add(y,x,y,int32_t(160));PipeBarrier<PIPE_ALL>();DataCopy(o[off],y,160);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_bias(void* stream,void* logits,void* markov,void* out,uint32_t B,uint32_t step){
    if(!stream||!B||B>MAX_BATCH||step>=5||!aligned(logits)||!aligned(markov)||!aligned(out))return -1;
    uint64_t full=uint64_t(B)*6*16160*4,compact=uint64_t(B)*16160*4;
    if(overlap(out,full,logits,full)||overlap(out,full,markov,compact))return -2;
    uint32_t blocks=blocks_for(B*101);ds_bias_kernel<<<blocks,nullptr,stream>>>((uint8_t*)logits,(uint8_t*)markov,(uint8_t*)out,B,step,blocks);return 0;
}
extern "C" __global__ __aicore__ void ds_chain_kernel(GM_ADDR GREEDY,GM_ADDR IDS,GM_ADDR RESULT,uint32_t B,uint32_t step,uint32_t slots){
    if(GetBlockIdx()!=0)return;GlobalTensor<int64_t> g,ids,r;g.SetGlobalBuffer((__gm__ int64_t*)GREEDY);ids.SetGlobalBuffer((__gm__ int64_t*)IDS);r.SetGlobalBuffer((__gm__ int64_t*)RESULT);
    TPipe p;TBuf<TPosition::VECCALC> bv;p.InitBuffer(bv,64);auto v=bv.Get<int64_t>();
    for(uint32_t b=0;b<B;++b){DataCopyPad(v,ids[b*6],DataCopyExtParams{1,48,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);v.SetValue(step+1,ds_live(r,b,B,slots)?g.GetValue(b*6+step):-1);SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);DataCopyPad(ids[b*6],v,DataCopyExtParams{1,48,0,0,0});PipeBarrier<PIPE_ALL>();}
}
extern "C" int dec_ds_chain(void* stream,void* greedy,void* ids,void* result,uint32_t B,uint32_t step,uint32_t slots){if(!stream||!B||B>MAX_BATCH||step>=5||slots<B||!aligned(greedy)||!aligned(ids)||!aligned(result))return -1;ds_chain_kernel<<<1,nullptr,stream>>>((uint8_t*)greedy,(uint8_t*)ids,(uint8_t*)result,B,step,slots);return 0;}

// Multihead RMS is the existing norm leaf with a rows*heads launch count.
extern "C" int dec_ds_rms(void* stream,void* x,void* w,void* y,uint32_t B,uint32_t heads,uint32_t n,float eps){
    if(!stream||!B||B>MAX_BATCH||!(heads==1||heads==8)||!(n==512||n==1280||n==5120)||!aligned(x)||!aligned(y)||(w&&!aligned(w))||!(eps>0&&eps<=FLT_MAX))return -1;
    uint32_t rows=B*6*heads,blocks=blocks_for(rows);uint64_t bytes=uint64_t(rows)*n*2;
    if((x!=y&&overlap(x,bytes,y,bytes))||(w&&overlap(w,n*4,y,bytes)))return -1;
    decode_rms_norm_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,(uint8_t*)y,rows,int32_t(n),eps,w!=nullptr,blocks,1.0f/float(n));return 0;
}


// DSpark KV: 14 groups of 32 non-RoPE coordinates; last 64 stay BF16.
// Scalar RNE is explicit (no dependence on the device's FP8 conversion mode).
__aicore__ inline float ds_pow2(int e){union{uint32_t u;float f;}v;v.u=uint32_t(e+127)<<23;return v.f;}
__aicore__ inline float ds_fp8_value(float x,float scale){
    float u=x/scale,a=u<0?-u:u;union{float f;uint32_t u;}v;v.f=a;
    int e=int((v.u>>23)&255)-127;if(e<-6)e=-6;
    float step=ds_pow2(e-3),z=a/step;
    int n=int(z);float rem=z-float(n);if(rem>0.5f||(rem==0.5f&&(n&1)))++n;
    float y=float(n)*step;if(y>448.0f)y=448.0f;
    return (u<0?-y:y)*scale;
}
extern "C" __global__ __aicore__ void ds_kv_fp8_kernel(GM_ADDR X,uint32_t rows,uint32_t blocks){
    if(GetBlockIdx()>=blocks)return;
    GlobalTensor<bfloat16_t> gx;gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);
    TPipe p;TBuf<TPosition::VECCALC> bh,bf;p.InitBuffer(bh,64);p.InitBuffer(bf,128);
    auto h=bh.Get<bfloat16_t>();auto f=bf.Get<float>();
    for(uint32_t task=GetBlockIdx();task<rows*14;task+=blocks){
        uint64_t off=uint64_t(task/14)*512+(task%14)*32;
        DataCopy(h,gx[off],32);PipeBarrier<PIPE_ALL>();Cast(f,h,RoundMode::CAST_NONE,int32_t(32));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float amax=1e-4f;for(int j=0;j<32;++j){float x=f.GetValue(j),a=x<0?-x:x;if(a>amax)amax=a;}
        union{float f;uint32_t u;}v;v.f=amax/448.0f;
        int e=int((v.u>>23)&255)-127;float scale=ds_pow2(e);if(scale<v.f)scale*=2.0f;
        for(int j=0;j<32;++j)f.SetValue(j,ds_fp8_value(f.GetValue(j),scale));
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Cast(h,f,RoundMode::CAST_RINT,int32_t(32));PipeBarrier<PIPE_ALL>();
        DataCopy(gx[off],h,32);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_kv_fp8(void* stream,void* kv,uint32_t batch){
    if(!stream||!aligned(kv)||!batch||batch>MAX_BATCH)return -1;
    uint32_t rows=batch*6,blocks=blocks_for(rows*14);
    ds_kv_fp8_kernel<<<blocks,nullptr,stream>>>((uint8_t*)kv,rows,blocks);return 0;
}

// DSpark target features: call at ENTRY to target layers 37/38/39 with index 0/1/2.
// BF16 x[B,6,4,5120] -> BF16 features[B,6,15360], contiguous base pointers.
// Only features[...,index*5120:(index+1)*5120] is written; other slices survive.
// Matches prefill torch.mean(h,dim=-2): sequential FP32 sum of the four HC
// lanes, then *0.25 and one BF16 RNE. No RMS, PRE weights, mask or workspace.
// Inputs and the ENTIRE output allocation must be disjoint and 32-byte aligned.
// B=1..MAX_BATCH. Caller supplies dtype/extent, persistent storage, live stream,
// ordering and warmup before capture. No host/device allocation or synchronization.
extern "C" __global__ __aicore__ void decode_ds_features_kernel(
    GM_ADDR X,GM_ADDR F,uint32_t rows,uint32_t index,uint32_t blocks) {
    if(GetBlockIdx()>=blocks)return;
    constexpr int32_t D=5120,T=1280;
    TPipe pipe;TBuf<TPosition::VECCALC> bh,bv;
    pipe.InitBuffer(bh,4*T*2);pipe.InitBuffer(bv,4*T*4);
    auto h=bh.Get<bfloat16_t>();auto v=bv.Get<float>();
    GlobalTensor<bfloat16_t> gx,gf;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)X);
    gf.SetGlobalBuffer((__gm__ bfloat16_t*)F);
    for(uint32_t task=GetBlockIdx();task<rows*4;task+=blocks) {
        uint32_t row=task/4,col=(task%4)*T;
        for(int32_t lane=0;lane<4;++lane)
            DataCopy(h[lane*T],gx[(uint64_t(row)*4+lane)*D+col],T);
        PipeBarrier<PIPE_ALL>();
        Cast(v,h,RoundMode::CAST_NONE,4*T);PipeBarrier<PIPE_V>();
        // The reference reduction starts from +0, including all-negative-zero rows.
        Adds(v,v,0.0f,T);PipeBarrier<PIPE_V>();
        Add(v,v,v[T],T);PipeBarrier<PIPE_V>();
        Add(v,v,v[2*T],T);PipeBarrier<PIPE_V>();
        Add(v,v,v[3*T],T);PipeBarrier<PIPE_V>();
        Muls(v,v,0.25f,T);PipeBarrier<PIPE_V>();
        Cast(h,v,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_ALL>();
        DataCopy(gf[(uint64_t(row)*3+index)*D+col],h,T);
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_ds_features(void* stream,void* x,void* features,
    uint32_t batch,uint32_t index) {
    if(!stream||!aligned(x)||!aligned(features)||!batch||batch>MAX_BATCH||index>2)
        return -1;
    uint32_t rows=batch*Q,blocks=blocks_for(rows*4);
    if(overlap(x,uint64_t(rows)*4*5120*2,features,uint64_t(rows)*3*5120*2))
        return -1;
    decode_ds_features_kernel<<<blocks,nullptr,stream>>>((uint8_t*)x,
        (uint8_t*)features,rows,index,blocks);
    return 0; // Submitted, not synchronized; asynchronous errors belong to caller.
}

// Fused HC normalization and gates; the sole HC gate implementation.
extern "C" __global__ __aicore__ void decode_hc_scale_gates_kernel(
    GM_ADDR Z,GM_ADDR ST,GM_ADDR SC,GM_ADDR BASE,GM_ADDR PRE,GM_ADDR POST,GM_ADDR COMB,
    uint32_t rows,uint32_t iters,float eps,uint32_t blocks) {
    if(GetBlockIdx()>=blocks)return;
    TPipe pipe;TBuf<TPosition::VECCALC> bz,bb,bs,bv,bt,br,bg,bo,bst;
    pipe.InitBuffer(bst,32);pipe.InitBuffer(bz,128);pipe.InitBuffer(bb,128);pipe.InitBuffer(bs,32);
    pipe.InitBuffer(bv,128);pipe.InitBuffer(bt,128);pipe.InitBuffer(br,128);
    pipe.InitBuffer(bg,128);pipe.InitBuffer(bo,128);
    auto stat=bst.Get<float>();auto z=bz.Get<float>(),base=bb.Get<float>(),sc=bs.Get<float>();
    auto v=bv.Get<float>(),tmp=bt.Get<float>(),red=br.Get<float>();
    auto gate=bg.Get<float>(),out=bo.Get<float>();
    GlobalTensor<float> gz,gb,gs,gpre,gpost,gcomb,gst;
    gst.SetGlobalBuffer((__gm__ float*)ST);gz.SetGlobalBuffer((__gm__ float*)Z);gb.SetGlobalBuffer((__gm__ float*)BASE);
    gs.SetGlobalBuffer((__gm__ float*)SC);gpre.SetGlobalBuffer((__gm__ float*)PRE);
    gpost.SetGlobalBuffer((__gm__ float*)POST);gcomb.SetGlobalBuffer((__gm__ float*)COMB);
    DataCopy(base,gb,int32_t(24));
    DataCopyPad(sc,gs,DataCopyExtParams{1,12,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    float scales[3];for(int i=0;i<3;++i)scales[i]=sc.GetValue(i);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks){
        DataCopy(z,gz[uint64_t(row)*24],int32_t(24));
        DataCopyPad(stat,gst[row],DataCopyExtParams{1,4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        // Same vector FP32 multiply and materialized z as dec_hc_scale.
        Muls(z,z,stat.GetValue(0),int32_t(24));PipeBarrier<PIPE_ALL>();
        DataCopy(gz[uint64_t(row)*24],z,int32_t(24));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(int i=0;i<8;++i)gate.SetValue(i,-(z.GetValue(i)*scales[i/4]+base.GetValue(i)));
        for(int i=0;i<4;++i)for(int j=0;j<8;++j)
            v.SetValue(i*8+j,j<4?z.GetValue(8+i*4+j)*scales[2]+base.GetValue(8+i*4+j):-FLT_MAX);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Exp(gate,gate,int32_t(8));Duplicate(tmp,1.0f,int32_t(8));PipeBarrier<PIPE_V>();
        Adds(gate,gate,1.0f,int32_t(8));PipeBarrier<PIPE_V>();
        Div(gate,tmp,gate,int32_t(8));PipeBarrier<PIPE_V>();
        // Softmax each four-column row; padding lanes exponentiate to zero.
        for(int i=0;i<4;++i){
            ReduceMax(red,v[i*8],tmp,int32_t(8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float mx=red.GetValue(0);Adds(v[i*8],v[i*8],-mx,int32_t(8));PipeBarrier<PIPE_V>();
            Exp(v[i*8],v[i*8],int32_t(8));PipeBarrier<PIPE_V>();
            ReduceSum(red,v[i*8],tmp,int32_t(8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float sum=red.GetValue(0);Duplicate(tmp,sum,int32_t(8));PipeBarrier<PIPE_V>();
            Div(v[i*8],v[i*8],tmp,int32_t(8));PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(int i=0;i<4;++i){
            out.SetValue(i,gate.GetValue(i)+eps);out.SetValue(8+i,2.0f*gate.GetValue(4+i));
            for(int j=0;j<4;++j)v.SetValue(i*8+j,v.GetValue(i*8+j)+eps);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        // Initial column norm, then (row,column) iters-1 times.
        for(uint32_t it=0;it<iters;++it){
            if(it){
                // Independent rows retain the exact original ReduceSum(8) tree.
                // Complete reductions before reusing tmp for the four denominators.
                for(int i=0;i<4;++i){
                    ReduceSum(red[i*8],v[i*8],tmp,int32_t(8));PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for(int i=0;i<4;++i){
                    float sum=red.GetValue(i*8)+eps;Duplicate(tmp[i*8],sum,int32_t(8));
                }
                PipeBarrier<PIPE_V>();
                for(int i=0;i<4;++i)Div(v[i*8],v[i*8],tmp[i*8],int32_t(8));
                PipeBarrier<PIPE_V>();
            }
            Add(tmp,v,v[8],int32_t(8));PipeBarrier<PIPE_V>();
            Add(tmp,tmp,v[16],int32_t(8));PipeBarrier<PIPE_V>();
            Add(tmp,tmp,v[24],int32_t(8));PipeBarrier<PIPE_V>();
            Adds(tmp,tmp,eps,int32_t(8));PipeBarrier<PIPE_V>();
            for(int i=0;i<4;++i)Div(v[i*8],v[i*8],tmp,int32_t(8));
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(int i=0;i<4;++i)for(int j=0;j<4;++j)gate.SetValue(i*4+j,v.GetValue(i*8+j));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gpre[uint64_t(row)*4],out,DataCopyExtParams{1,16,0,0,0});
        DataCopyPad(gpost[uint64_t(row)*4],out[8],DataCopyExtParams{1,16,0,0,0});
        DataCopy(gcomb[uint64_t(row)*16],gate,int32_t(16));PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int dec_hc_scale_gates(void* stream,void* z,void* stats,void* scale,void* base,
    void* pre,void* post,void* comb,uint32_t batch,uint32_t iters,float eps){
    if(!stream||!batch||batch>MAX_BATCH||!iters||iters>100||!(eps>0&&eps<=FLT_MAX))return -1;
    void* ptrs[]={z,stats,scale,base,pre,post,comb};
    uint64_t rows=uint64_t(batch)*Q,sizes[]={rows*96,rows*4,12,96,rows*16,rows*16,rows*64};
    for(int i=0;i<7;++i)if(!aligned(ptrs[i]))return -1;
    for(int i=0;i<7;++i)for(int j=0;j<i;++j)
        if((i>=4||j==0)&&overlap(ptrs[i],sizes[i],ptrs[j],sizes[j]))return -1;
    uint32_t blocks=blocks_for(uint32_t(rows));
    decode_hc_scale_gates_kernel<<<blocks,nullptr,stream>>>((uint8_t*)z,(uint8_t*)stats,
        (uint8_t*)scale,(uint8_t*)base,(uint8_t*)pre,(uint8_t*)post,(uint8_t*)comb,
        uint32_t(rows),iters,eps,blocks);
    return 0;
}
