#ifndef HEADGROUP
#define HEADGROUP 2
#endif
#ifndef TILEROWS
#define TILEROWS 32
#endif
#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
constexpr int DMAX=512;
template<HardEvent E> __aicore__ inline void fence() { SetFlag<E>(EVENT_ID0); WaitFlag<E>(EVENT_ID0); }
__aicore__ inline void sync() {
    PipeBarrier<PIPE_ALL>();
    fence<HardEvent::V_S>(); fence<HardEvent::MTE2_S>(); fence<HardEvent::MTE3_S>();
    fence<HardEvent::S_V>(); fence<HardEvent::S_MTE2>(); fence<HardEvent::S_MTE3>();
}
// Raw scalar GM accesses use the AIV data cache, not the DMA engine.
// Invalidate on entry and publish on exit; writers own whole 128-byte lines.
__aicore__ inline void cache(GM_ADDR p) {
    GlobalTensor<uint8_t> g; g.SetGlobalBuffer(p);
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(g);
    sync();
}

// Vector-only build: launch two explicit AIV tasks per logical block.
// Keep logical block count and task stride unchanged from mixed execution.
__aicore__ inline int lane() { return GetBlockIdx(); }
__aicore__ inline int lanes(int blocks) { return 2 * blocks; }
__aicore__ inline float hi(float a, float b) { return a > b ? a : b; }
struct Vec {
    TPipe pipe;
    TBuf<TPosition::VECCALC> fb, bb;
    LocalTensor<float> a,b,c,d,z,w;
    LocalTensor<bfloat16_t> bf;
    __aicore__ inline Vec() {
        pipe.InitBuffer(fb,6*DMAX*4); pipe.InitBuffer(bb,DMAX*2);
        a=fb.Get<float>(); b=a[DMAX]; c=a[2*DMAX]; d=a[3*DMAX]; z=a[4*DMAX]; w=a[5*DMAX];
        bf=bb.Get<bfloat16_t>();
    }
    __aicore__ inline void load(LocalTensor<float> dst, __gm__ bfloat16_t* p, int n) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        DataCopy(bf,g,n); fence<HardEvent::MTE2_V>();
        Cast(dst,bf,RoundMode::CAST_NONE,int32_t(n)); PipeBarrier<PIPE_V>(); fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void load(LocalTensor<float> dst, __gm__ float* p, int n) {
        GlobalTensor<float> g; g.SetGlobalBuffer(p);
        DataCopy(dst,g,n); sync();
    }
    __aicore__ inline void store(__gm__ bfloat16_t* p, LocalTensor<float> src, int n) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        sync(); Cast(bf,src,RoundMode::CAST_RINT,int32_t(n));
        sync(); DataCopy(g,bf,n); sync();
    }
    __aicore__ inline void store(__gm__ float* p, LocalTensor<float> src, int n) {
        GlobalTensor<float> g; g.SetGlobalBuffer(p);
        sync(); DataCopy(g,src,n); sync();
    }
    __aicore__ inline float exp(float x) {
        Duplicate(z,x,int32_t(8)); PipeBarrier<PIPE_V>();
        Exp(z,z,int32_t(8)); fence<HardEvent::V_S>(); float v=z.GetValue(0); fence<HardEvent::S_V>(); return v;
    }
    __aicore__ inline float dot(int n) {
        Mul(c,a,b,int32_t(n)); PipeBarrier<PIPE_V>();
        ReduceSum(z,c,w,int32_t(n)); fence<HardEvent::V_S>(); float v=z.GetValue(0); fence<HardEvent::S_V>(); return v;
    }
};


// Read-only SWA over committed canonical ring and separate pending Q6 rows.
// q/out BF16[B,6,H,512], pending BF16[B,6,512], ring BF16[S,pad+mod,512].
// slots/start/active INT64[B]; sink FP32[H]. Device slots/start must be valid;
// inactive or invalid slots produce zero. Caller owns lifetimes and ordering.
// This leaf neither quantizes KV nor publishes any persistent state.
// Isolated target attention candidate: 16 keys / AIV-local tile.
// No GM workspace, no host effective length, and no change to other kernels.
// Explicit UB: 87424 bytes (71040 FP32 + 16384 BF16), excluding compiler stack.
// --- v14: head-group shared KV tile -------------------------------------
// Identical per-head numerics; one resolved KV tile is loaded and cast once,
// then consumed by G queries of the same (b,t).
template<int G> struct TargetAttentionTile {
    static constexpr int K = TILEROWS;
    static constexpr int D = 512;
    static constexpr int NF = 2*G*D + 2*K*D + K*8 + 64 + K*8 + 32;
    TPipe pipe;
    TBuf<TPosition::VECCALC> fb, bb;
    LocalTensor<float> q, acc, kv, work, part, score, broad, stat;
    LocalTensor<bfloat16_t> bf;
    __aicore__ inline TargetAttentionTile() {
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
        fence<HardEvent::MTE2_V>();
        Cast(q, bf, RoundMode::CAST_NONE, int32_t(ge*D));
        PipeBarrier<PIPE_V>();
        fence<HardEvent::V_MTE2>(); // BF16 staging may now be overwritten.
    }
    __aicore__ inline void load_run(__gm__ bfloat16_t* p, int offset, int rows) {
        GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer(p);
        DataCopy(bf[offset*D], g, int32_t(rows*D));
    }
    // One cast per tile, shared by every head in the group.
    __aicore__ inline void cast_tile(int n) {
        fence<HardEvent::MTE2_V>();
        Cast(kv, bf, RoundMode::CAST_NONE, int32_t(n*D));
        PipeBarrier<PIPE_V>();
        fence<HardEvent::V_MTE2>();
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
        fence<HardEvent::V_S>();
        float next = hi(mx, stat.GetValue(0));
        fence<HardEvent::S_V>();
        Adds(score, score, -next, int32_t(K));
        Duplicate(score[32], mx==next ? 0.f : mx-next, int32_t(8));
        PipeBarrier<PIPE_V>();
        Exp(score, score, int32_t(64));
        PipeBarrier<PIPE_V>();
        WholeReduceSum(stat[8], score, int32_t(n), 1, 1, 1, 8);
        Brcb(broad, score, uint8_t(K/8), BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        fence<HardEvent::V_S>();
        float alpha = mx==next ? 1.f : score.GetValue(32);
        float tile_den = stat.GetValue(8);
        fence<HardEvent::S_V>();
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
        sync();
        Cast(bf, acc, RoundMode::CAST_RINT, int32_t(ge*D));
        sync(); DataCopy(g, bf, ge*D); sync();
    }
};

template<int RATIO> __global__ __aicore__ void swa_q6(GM_ADDR qp,GM_ADDR rp,GM_ADDR np,
        GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR sp,GM_ADDR op,
        int batch,int heads,int slots,int mod,int pad,int blocks,
        GM_ADDR gp,GM_ADDR ptp,GM_ADDR cp,GM_ADDR ip,
        int pages,int maxpages,int rpp) {
    cache(slp);cache(stp);cache(ap);
    if constexpr(RATIO>0) { cache(ptp);cache(ip); }
    constexpr int G = HEADGROUP;
    using T = TargetAttentionTile<G>;
    T v;
    auto sl=(__gm__ int64_t*)slp;auto start=(__gm__ int64_t*)stp;
    auto active=(__gm__ int64_t*)ap;
    int groups=(heads+G-1)/G;
    for(int job=lane();job<batch*6*groups;job+=lanes(blocks)) {
        int b=job/(6*groups),t=(job/groups)%6,hg=job%groups;
        int h0=hg*G; int ge=heads-h0; if(ge>G)ge=G;
        int64_t slot=sl[b],base=start[b],pos=base+t;
        int64_t qrow=(int64_t(b)*6+t)*heads+h0;
        Duplicate(v.acc,0.f,int32_t(G*512));PipeBarrier<PIPE_V>();
        if(active[b] && slot>=0 && slot<slots && base>=0) {
            v.load_query((__gm__ bfloat16_t*)qp+qrow*512, ge);
            float mx[G],den[G];
            for(int g=0;g<ge;++g){ mx[g]=((__gm__ float*)sp)[h0+g]; den[g]=1.f; }
            for(int tile=0;tile<128+(RATIO?512:0);tile+=T::K) {
                int n=0, run_rows=0, run_offset=0;
                __gm__ bfloat16_t* run=nullptr;
                for(int j=tile;j<tile+T::K;++j) {
                    __gm__ bfloat16_t* row=nullptr;
                    if(j<128) {
                        int64_t k=pos-127+j;
                        if(k<0)continue;
                        if(k>=base)row=(__gm__ bfloat16_t*)np+(int64_t(b)*6+k-base)*512;
                        else row=(__gm__ bfloat16_t*)rp+(slot*(pad+mod)+pad+k%mod)*512;
                    } else if constexpr(RATIO>0) {
                        int64_t id=((__gm__ int64_t*)ip)[(int64_t(b)*6+t)*512+j-128];
                        if(id<0 || id>=(pos+1)/RATIO)continue;
                        int64_t committed=base/RATIO;
                        if(id>=committed) {
                            int64_t offset=id-committed;
                            if(offset>=6)continue;
                            row=(__gm__ bfloat16_t*)cp+(int64_t(b)*6+offset)*512;
                        } else {
                            int64_t page_index=id/rpp;
                            if(page_index>=maxpages)continue;
                            int64_t page=((__gm__ int64_t*)ptp)[slot*maxpages+page_index];
                            if(page<0 || page>=pages)continue;
                            row=(__gm__ bfloat16_t*)gp+(page*rpp+id%rpp)*512;
                        }
                    }
                    if(run_rows && row==run+int64_t(run_rows)*512) ++run_rows;
                    else {
                        if(run_rows) v.load_run(run,run_offset,run_rows);
                        run=row; run_offset=n; run_rows=1;
                    }
                    ++n;
                }
                if(run_rows) v.load_run(run,run_offset,run_rows);
                if(n) {
                    v.cast_tile(n);
                    for(int g=0;g<ge;++g) v.consume(n,g,mx[g],den[g]);
                }
            }
            for(int g=0;g<ge;++g) {
                Muls(v.acc[g*512],v.acc[g*512],1.f/den[g],int32_t(512));
                PipeBarrier<PIPE_V>();
            }
        }
        v.store((__gm__ bfloat16_t*)op+qrow*512, ge);
    }
}
extern "C" int dec_swa_q6(void* stream,void* q,void* ring,void* pending,
        void* slots,void* start,void* active,void* sink,void* out,
        int batch,int heads,int nslots,int modulus,int pad) {
    if(batch<1||batch>8||heads<1||heads>128||nslots<batch||modulus<128||pad<0)return -1;
    void* ptrs[]={q,ring,pending,slots,start,active,sink,out};
    for(auto ptr:ptrs)if(!ptr||(uintptr_t(ptr)&31))return -1;
    // All ranges are caller-validated disjoint; in-place attention is forbidden.
    for(int i=0;i<7;++i)if(out==ptrs[i])return -1;
    int blocks=(batch*6*((heads+HEADGROUP-1)/HEADGROUP)+1)/2;if(blocks>40)blocks=40;
    swa_q6<0><<<(2*blocks),nullptr,stream>>>((uint8_t*)q,(uint8_t*)ring,(uint8_t*)pending,
        (uint8_t*)slots,(uint8_t*)start,(uint8_t*)active,(uint8_t*)sink,(uint8_t*)out,
        batch,heads,nslots,modulus,pad,blocks,nullptr,nullptr,nullptr,nullptr,0,0,0);
    return 0;
}

// Joint CSA: one softmax over SWA, selected compressed rows and the sink.
// bank BF16[pages,rpp,512], table INT64[nslots,maxpages], ids INT64[B,6,512].
// compressed_pending BF16[B,6,512] begins at logical row floor(start/ratio).
// Only its first floor((start+6)/ratio)-floor(start/ratio) rows are read.
// Selected nonnegative IDs must be unique per query. Local/global duplicates
// are intentional distinct entries. Caller reserves committed pages; no writes
// to Past, bank gather, allocation or host-dependent replay shape changes.
extern "C" int dec_csa_q6(void* stream,void* q,void* ring,void* pending,
        void* slots,void* start,void* active,void* sink,void* out,
        void* bank,void* table,void* compressed_pending,void* ids,
        int batch,int heads,int nslots,int modulus,int pad,
        int pages,int maxpages,int rpp,int ratio) {
    if(batch<1||batch>8||heads<1||heads>128||nslots<batch||modulus<128||pad<0
       ||pages<1||maxpages<1||rpp<1||(ratio!=1&&ratio!=2))return -1;
    void* ptrs[]={q,ring,pending,slots,start,active,sink,bank,table,compressed_pending,ids,out};
    for(auto ptr:ptrs)if(!ptr||(uintptr_t(ptr)&31))return -1;
    for(int i=0;i<11;++i)if(out==ptrs[i])return -1;
    int blocks=(batch*6*((heads+HEADGROUP-1)/HEADGROUP)+1)/2;if(blocks>40)blocks=40;
    if(ratio==1)
        swa_q6<1><<<(2*blocks),nullptr,stream>>>((uint8_t*)q,(uint8_t*)ring,(uint8_t*)pending,
            (uint8_t*)slots,(uint8_t*)start,(uint8_t*)active,(uint8_t*)sink,(uint8_t*)out,
            batch,heads,nslots,modulus,pad,blocks,(uint8_t*)bank,(uint8_t*)table,
            (uint8_t*)compressed_pending,(uint8_t*)ids,pages,maxpages,rpp);
    else
        swa_q6<2><<<(2*blocks),nullptr,stream>>>((uint8_t*)q,(uint8_t*)ring,(uint8_t*)pending,
            (uint8_t*)slots,(uint8_t*)start,(uint8_t*)active,(uint8_t*)sink,(uint8_t*)out,
            batch,heads,nslots,modulus,pad,blocks,(uint8_t*)bank,(uint8_t*)table,
            (uint8_t*)compressed_pending,(uint8_t*)ids,pages,maxpages,rpp);
    return 0;
}

// Local KV FP8/E4M3FN round trip, block32 power-of-two scale, finite BF16 input.
// Caller owns contiguous BF16[B,6,512] buffers. Exact in-place is supported;
// partial overlap is rejected. Does not publish Past or allocate storage.
__aicore__ inline float kv_powceil(float a) {
    union { float f; uint32_t u; } x; x.f=a;
    x.u=(x.u&0x7f800000u)+((x.u&0x7fffffu)?0x800000u:0u);return x.f;
}
__global__ __aicore__ void decode_kv_qdq_kernel(GM_ADDR xp,GM_ADDR yp,int N,int B) {
    constexpr int M=512;
    TPipe pipe; TBuf<TPosition::VECCALC> fb,bb,ib;
    pipe.InitBuffer(fb,7*M*4); pipe.InitBuffer(bb,M*2); pipe.InitBuffer(ib,M*4);
    auto x=fb.Get<float>(), a=x[M], scale=x[2*M], u=x[3*M], step=x[4*M], y=x[5*M], tmp=x[6*M];
    auto bf=bb.Get<bfloat16_t>(); auto ints=ib.Get<int32_t>();
    GlobalTensor<bfloat16_t> gx,gy;gx.SetGlobalBuffer((__gm__ bfloat16_t*)xp);gy.SetGlobalBuffer((__gm__ bfloat16_t*)yp);
    for(int off=lane()*M;off<N;off+=lanes(B)*M){
        int n=N-off<M?N-off:M;
        DataCopy(bf,gx[off],n);fence<HardEvent::MTE2_V>();
        Cast(x,bf,RoundMode::CAST_NONE,int32_t(n));PipeBarrier<PIPE_V>();
        Abs(a,x,int32_t(n));PipeBarrier<PIPE_V>();
        for(int j=0;j<n;j+=32){
            ReduceMax(tmp,a[j],y,int32_t(32),false);fence<HardEvent::V_S>();
            float s=kv_powceil(hi(tmp.GetValue(0),1e-4f)/448.f);
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
        Cast(bf,y,RoundMode::CAST_RINT,int32_t(n));fence<HardEvent::V_MTE3>();
        DataCopy(gy[off],bf,n);fence<HardEvent::MTE3_MTE2>();
    }
}

extern "C" int dec_kv_qdq(void* stream,void* x,void* y,int batch) {
    if(batch<1||batch>8||!x||!y||(uintptr_t(x)&31)||(uintptr_t(y)&31))return -1;
    uintptr_t a=uintptr_t(x),b=uintptr_t(y);uint64_t bytes=uint64_t(batch)*6*512*2;
    if(a!=b&&(a<b?b-a:a-b)<bytes)return -1;
    int blocks=(batch*6+1)/2;
    decode_kv_qdq_kernel<<<(2*blocks),nullptr,stream>>>((uint8_t*)x,(uint8_t*)y,batch*6*512,blocks);
    return 0;
}

// Decode source: fixed Q6, read-only committed history/carry; no publication.
// Phases are gathered from the caller's canonical FP32[L,32,2] RoPE table.
__global__ __aicore__ void source_prepare(GM_ADDR xp,GM_ADDR fp,GM_ADDR slp,
        GM_ADDR stp,GM_ADDR ap,GM_ADDR tp,GM_ADDR gp,GM_ADDR hp,
        int batch,int ratio,int nslots,int freqrows,int widen,int blocks) {
    cache(slp);cache(stp);cache(ap); Vec v;
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    for(int row=lane();row<batch*6;row+=lanes(blocks)) {
        int b=row/6,t=row%6;
        bool live=ac[b] && sl[b]>=0 && sl[b]<nslots && st[b]>=0 && st[b]<=freqrows-6;
        int64_t pos=live?st[b]+t:0, group=live?(st[b]/ratio+t)*ratio:0;
        // Unused ratio2 pending rows have identity phase and zero pool output.
        if(t>=6/ratio)group=0;
        v.load(v.a,(__gm__ float*)fp+pos*64,64);
        v.store((__gm__ float*)tp+int64_t(row)*64,v.a,64);
        v.load(v.a,(__gm__ float*)fp+group*64,64);
        v.store((__gm__ float*)gp+int64_t(row)*64,v.a,64);
        if(widen)for(int d=0;d<5120;d+=512) {
            v.load(v.a,(__gm__ bfloat16_t*)xp+int64_t(row)*5120+d,512);
            v.store((__gm__ float*)hp+int64_t(row)*5120+d,v.a,512);
        }
    }
}
extern "C" int dec_source_prepare(void* stream,void* x,void* freqs,void* slots,
        void* start,void* active,void* token,void* group,void* hidden_f32,
        int batch,int ratio,int nslots,int freqrows,int widen) {
    if(batch<1||batch>8||(ratio!=1&&ratio!=2)||nslots<batch||freqrows<6||widen<0||widen>1)return -1;
    void* ps[]={x,freqs,slots,start,active,token,group};
    for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    if(widen&&(!hidden_f32||(uintptr_t(hidden_f32)&31)))return -1;
    int blocks=(batch*6+1)/2;
    source_prepare<<<(2*blocks),nullptr,stream>>>((uint8_t*)x,(uint8_t*)freqs,(uint8_t*)slots,
        (uint8_t*)start,(uint8_t*)active,(uint8_t*)token,(uint8_t*)group,(uint8_t*)hidden_f32,
        batch,ratio,nslots,freqrows,widen,blocks); return 0;
}

__global__ __aicore__ void source_pool(GM_ADDR vp,GM_ADDR sp,GM_ADDR cvp,GM_ADDR csp,
        GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,int batch,int ratio,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap); Vec v;
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    for(int row=lane();row<batch*6;row+=lanes(blocks)) {
        int b=row/6,g=row%6;int64_t base=st[b],slot=sl[b];
        if(!ac[b]||slot<0||slot>=nslots||base<0||g>=6/ratio) {
            Duplicate(v.a,0.f,int32_t(512));PipeBarrier<PIPE_V>();
        } else if(ratio==1) v.load(v.a,(__gm__ bfloat16_t*)vp+int64_t(row)*512,512);
        else {
            int rel=2*g-int(base%2);
            auto v0=rel<0?(__gm__ float*)cvp+(slot*4+(base-1)%4)*512:
                (__gm__ float*)vp+(int64_t(b)*6+rel)*512;
            auto s0=rel<0?(__gm__ float*)csp+(slot*4+(base-1)%4)*512:
                (__gm__ float*)sp+(int64_t(b)*6+rel)*512;
            v.load(v.a,v0,512);v.load(v.b,(__gm__ float*)vp+(int64_t(b)*6+rel+1)*512,512);
            v.load(v.c,s0,512);v.load(v.d,(__gm__ float*)sp+(int64_t(b)*6+rel+1)*512,512);
            // Use the dedicated 512-float workspace for softmax intermediates.
            Max(v.w,v.c,v.d,int32_t(512));PipeBarrier<PIPE_V>();
            Sub(v.c,v.c,v.w,int32_t(512));Sub(v.d,v.d,v.w,int32_t(512));PipeBarrier<PIPE_V>();
            Exp(v.c,v.c,int32_t(512));Exp(v.d,v.d,int32_t(512));PipeBarrier<PIPE_V>();
            Add(v.w,v.c,v.d,int32_t(512));PipeBarrier<PIPE_V>();
            Div(v.c,v.c,v.w,int32_t(512));Div(v.d,v.d,v.w,int32_t(512));PipeBarrier<PIPE_V>();
            Mul(v.a,v.a,v.c,int32_t(512));Mul(v.b,v.b,v.d,int32_t(512));PipeBarrier<PIPE_V>();
            Add(v.a,v.a,v.b,int32_t(512));PipeBarrier<PIPE_V>();
        }
        v.store((__gm__ bfloat16_t*)op+int64_t(row)*512,v.a,512);
    }
}
extern "C" int dec_source_pool(void* stream,void* values,void* gates,void* carry_values,
        void* carry_scores,void* slots,void* start,void* active,void* out,int batch,int ratio,int nslots) {
    if(batch<1||batch>8||(ratio!=1&&ratio!=2)||nslots<batch)return -1;
    void* ps[]={values,slots,start,active,out};for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    if(ratio==2) {void* qs[]={gates,carry_values,carry_scores};for(auto p:qs)if(!p||(uintptr_t(p)&31))return -1;}
    int blocks=(batch*6+1)/2;
    source_pool<<<(2*blocks),nullptr,stream>>>((uint8_t*)values,(uint8_t*)gates,(uint8_t*)carry_values,
        (uint8_t*)carry_scores,(uint8_t*)slots,(uint8_t*)start,(uint8_t*)active,(uint8_t*)out,
        batch,ratio,nslots,blocks);return 0;
}

// E2M1 roundtrip: index Q/K block32 power-of-two scale; compressed KV
// block16 E4M3 scale. Scalar grid ties use even CODE, notably 5 -> 4.
__aicore__ inline float source_rne(float x) {
    int q=int(x);float f=x-float(q);return float(q+(f>0.5f||(f==0.5f&&(q&1))));
}
__aicore__ inline float source_e4m3(float x) {
    union {float f;uint32_t u;} a;a.f=hi(x,0.015625f);
    a.u&=0x7f800000u;float step=a.f*0.125f;
    return hi(0.f,source_rne(x/step)*step>448.f?448.f:source_rne(x/step)*step);
}
__aicore__ inline float source_e2m1(float x) {
    // Midpoints .25,.75,1.25,1.75,2.5,3.5,5 resolve to even payload code.
    if(x<=0.25f)return 0.f;if(x<0.75f)return 0.5f;
    if(x<=1.25f)return 1.f;if(x<1.75f)return 1.5f;
    if(x<=2.5f)return 2.f;if(x<3.5f)return 3.f;if(x<=5.f)return 4.f;return 6.f;
}
__global__ __aicore__ void source_fp4(GM_ADDR xp,GM_ADDR yp,int rows,int dim,int block,int e4,int blocks) {
    Vec v;
    for(int row=lane();row<rows;row+=lanes(blocks)) {
        v.load(v.a,(__gm__ bfloat16_t*)xp+int64_t(row)*dim,dim);
        fence<HardEvent::V_S>();
        for(int off=0;off<dim;off+=block) {
            float mx=0.f;
            for(int d=0;d<block;++d) {float x=v.a.GetValue(off+d);mx=hi(mx,x<0?-x:x);}
            float scale=e4?source_e4m3(hi(mx,0.01171875f)/6.f):kv_powceil(hi(mx,7.0529661e-38f)/6.f);
            for(int d=0;d<block;++d) {
                float x=v.a.GetValue(off+d),a=x<0?-x:x;
                float y=source_e2m1(a/scale)*scale;
                union {float f;uint32_t u;} in,out;in.f=x;out.f=y;out.u|=in.u&0x80000000u;
                v.b.SetValue(off+d,out.f);
            }
        }
        fence<HardEvent::S_V>();v.store((__gm__ bfloat16_t*)yp+int64_t(row)*dim,v.b,dim);
    }
}
extern "C" int dec_source_fp4(void* stream,void* x,void* y,int rows,int dim,int block,int e4) {
    if(rows<1||rows>192||dim<32||dim>512||dim%32||!x||!y||
        (uintptr_t(x)&31)||(uintptr_t(y)&31)||!((block==16&&e4==1)||(block==32&&e4==0)))return -1;
    uint64_t bytes=uint64_t(rows)*dim*2;uintptr_t a=uintptr_t(x),b=uintptr_t(y);
    if(a!=b&&(a<b?b-a:a-b)<bytes)return -1;
    int blocks=(rows+1)/2;if(blocks>40)blocks=40;
    source_fp4<<<(2*blocks),nullptr,stream>>>((uint8_t*)x,(uint8_t*)y,rows,dim,block,e4,blocks);return 0;
}

// Each job writes a disjoint 32-FP32 DMA tile. Invalid/tail entries are -inf
// on every rank, not zero (weights can be negative). Pending overrides bank.
// Tile-staged, fully vectorised paged scoring.
//
// A 64-key tile never crosses a page boundary (TILE divides rpp), so one DMA
// stages the whole tile and all four heads score it without a single scalar
// read-back: broadcast Mul over the staged keys, two BlockReduceSum levels to
// fold 128 lanes down to one score per key, then the weighted accumulation.
// Tiles straddling the pending rows keep the original per-key path. The fixed
// -inf tail past `limit` is no longer materialised because source_select only
// scans [0, limit).
__global__ __aicore__ void source_scores(GM_ADDR qp,GM_ADDR wp,GM_ADDR bankp,GM_ADDR ptp,
        GM_ADDR pendingp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,GM_ADDR logitsp,
        int batch,int ratio,int width,int nslots,int pages,int maxpages,int rpp,int npmax,int blocks) {
    cache(slp);cache(stp);cache(ap);cache(ptp);cache(wp); Vec v;
    constexpr int TILE=64;
    TBuf<TPosition::VECCALC> ob;v.pipe.InitBuffer(ob,TILE*4);auto out=ob.Get<float>();
    TBuf<TPosition::VECCALC> qb;v.pipe.InitBuffer(qb,4*128*4);auto queries=qb.Get<float>();
    TBuf<TPosition::VECCALC> sb;v.pipe.InitBuffer(sb,TILE*128*4);auto keys=sb.Get<float>();
    TBuf<TPosition::VECCALC> rb;v.pipe.InitBuffer(rb,TILE*128*2);auto raw=rb.Get<bfloat16_t>();
    TBuf<TPosition::VECCALC> pb;v.pipe.InitBuffer(pb,2*TILE*64*4);auto prodA=pb.Get<float>();
    auto prodB=prodA[TILE*64];
    TBuf<TPosition::VECCALC> tb;v.pipe.InitBuffer(tb,(TILE*8+3*TILE)*4);auto fold=tb.Get<float>();
    auto sA=fold[TILE*8],sB=fold[TILE*8+TILE],acc=fold[TILE*8+2*TILE];
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    auto pt=(__gm__ int64_t*)ptp;auto weights=(__gm__ uint16_t*)wp;
    const auto vb0=v.b;
    const int groups = batch*6 < 8 ? batch*6 : 8;
    const int team_size = lanes(blocks)/groups;
    if(lane() >= groups*team_size)return;
    const int group = lane()/team_size, worker = lane()%team_size;
    BinaryRepeatParams bcast;
    bcast.dstBlkStride=1;bcast.src0BlkStride=1;bcast.src1BlkStride=1;
    bcast.dstRepStride=8;bcast.src0RepStride=16;bcast.src1RepStride=0;
    for(int row=group;row<batch*6;row+=groups) {
        int b=row/6,t=row%6;
        int64_t base=st[b],slot=sl[b];bool live=ac[b]&&slot>=0&&slot<nslots&&base>=0;
        int64_t origin=live?base/ratio:0,limit=live?(base+t+1)/ratio:0;
        if(limit>width)limit=width;
        if(!live||limit<=0)continue;
        float wf[4];
        for(int h=0;h<4;++h) {
            v.load(queries[h*128],(__gm__ bfloat16_t*)qp+(int64_t(row)*4+h)*128,128);
            union {uint32_t u;float f;} hw;hw.u=uint32_t(weights[int64_t(row)*4+h])<<16;wf[h]=hw.f;
        }
        int tiles=int((limit+TILE-1)/TILE);
        for(int tile=worker;tile<tiles;tile+=team_size) {
            int off=tile*TILE,n=width-off;if(n>TILE)n=TILE;
            // Whole tile inside one committed page -> single DMA + vector path.
            // Cube pre-pass scored every committed key; fold the four heads.
            bool bulk=false;int64_t cbase=0;
            if(n==TILE && int64_t(off)+n<=origin && int64_t(off)/rpp<maxpages) {
                int64_t lp=int64_t(off)/rpp,page=pt[slot*maxpages+lp];
                if(page>=0&&page<pages) {
                    cbase=((int64_t(b)*npmax+lp)*24+int64_t(t)*4)*rpp+int64_t(off)%rpp;
                    bulk=true;
                }
            }
            if(bulk) {
                Duplicate(acc,0.f,TILE);PipeBarrier<PIPE_V>();
                for(int h=0;h<4;++h) {
                    GlobalTensor<float> gl;gl.SetGlobalBuffer((__gm__ float*)logitsp+cbase+int64_t(h)*rpp);
                    DataCopy(sA,gl,TILE);fence<HardEvent::MTE2_V>();
                    Maxs(sA,sA,0.f,TILE);PipeBarrier<PIPE_V>();
                    Muls(sB,sA,wf[h],TILE);PipeBarrier<PIPE_V>();
                    Add(acc,acc,sB,TILE);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE2>();
                }
                Muls(out,acc,0.015625f,TILE);PipeBarrier<PIPE_V>();
                fence<HardEvent::V_MTE2>();fence<HardEvent::V_MTE3>();
            } else {
                for(int j=0;j<n;++j) {
                    int id=off+j;bool have=false;
                    if(id<limit) {
                        v.b=vb0;__gm__ bfloat16_t* key=nullptr;
                        if(id>=origin) {int64_t rel=id-origin;if(rel<6/ratio)key=(__gm__ bfloat16_t*)pendingp+(int64_t(b)*6+rel)*128;}
                        else if(id/rpp<maxpages) {int64_t page=pt[slot*maxpages+id/rpp];
                            if(page>=0&&page<pages)key=(__gm__ bfloat16_t*)bankp+(page*rpp+id%rpp)*128;}
                        if(key) {v.load(v.b,key,128);have=true;}
                    }
                    float sum=-__builtin_inff();
                    if(have) {
                        sum=0.f;
                        for(int h=0;h<4;++h) {v.a=queries[h*128];sum+=hi(v.dot(128),0.f)*wf[h];}
                        sum*=0.015625f;
                    }
                    out.SetValue(j,sum);
                }
                v.b=vb0;
                fence<HardEvent::S_MTE3>();
            }
            GlobalTensor<float> dst;dst.SetGlobalBuffer((__gm__ float*)op+int64_t(row)*width+off);
            DataCopy(dst,out,n);fence<HardEvent::MTE3_S>();fence<HardEvent::MTE3_V>();
        }
    }
}


extern "C" int dec_source_scores(void* stream,void* q,void* weight,void* bank,void* table,
        void* pending,void* slots,void* start,void* active,void* out,void* logits,
        int batch,int ratio,int width,int nslots,int pages,int maxpages,int rpp,int npmax) {
    if(batch<1||batch>8||(ratio!=1&&ratio!=2)||width<8||width>1048576||width%8||
        nslots<batch||pages<1||maxpages<1||rpp<1||npmax<1||int64_t(npmax)*rpp<int64_t(width)||int64_t(width)>int64_t(maxpages)*rpp)return -1;
    void* ps[]={q,weight,bank,table,pending,slots,start,active,out,logits};for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    int blocks=40;
    source_scores<<<(2*blocks),nullptr,stream>>>((uint8_t*)q,(uint8_t*)weight,(uint8_t*)bank,(uint8_t*)table,
        (uint8_t*)pending,(uint8_t*)slots,(uint8_t*)start,(uint8_t*)active,(uint8_t*)out,(uint8_t*)logits,
        batch,ratio,width,nslots,pages,maxpages,rpp,npmax,blocks);return 0;
}

// Streaming min-heap in UB: O(N log 512), fixed storage; ties prefer lower ID.
#define SELCHUNK 2048
__aicore__ inline bool source_worse(float a,int ai,float b,int bi) {
    return a<b || (a==b && ai>bi);
}
__aicore__ inline void source_sift(LocalTensor<float> val,LocalTensor<int32_t> ids,int n,int at) {
    while(at*2+1<n) {
        int c=at*2+1;
        if(c+1<n&&source_worse(val.GetValue(c+1),ids.GetValue(c+1),val.GetValue(c),ids.GetValue(c)))++c;
        if(!source_worse(val.GetValue(c),ids.GetValue(c),val.GetValue(at),ids.GetValue(at)))break;
        float v=val.GetValue(at);int id=ids.GetValue(at);
        val.SetValue(at,val.GetValue(c));ids.SetValue(at,ids.GetValue(c));val.SetValue(c,v);ids.SetValue(c,id);at=c;
    }
}
__global__ __aicore__ void sel_p1(GM_ADDR sp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,GM_ADDR wsp,int batch,int ratio,int width,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap);
    TPipe pipe;TBuf<TPosition::VECCALC> hb,ib,rb,ob,mb,cb,xb,zb,tb,vb,wb;
    pipe.InitBuffer(hb,512*4);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(rb,SELCHUNK*4);pipe.InitBuffer(ob,512*8);
    pipe.InitBuffer(mb,512*4);pipe.InitBuffer(cb,SELCHUNK/2);pipe.InitBuffer(xb,SELCHUNK*4);pipe.InitBuffer(zb,SELCHUNK*4);
    pipe.InitBuffer(tb,256);pipe.InitBuffer(vb,512*4);pipe.InitBuffer(wb,512*4);
    auto val=hb.Get<float>();auto ids=ib.Get<int32_t>();auto buf=rb.Get<float>();auto out=ob.Get<int64_t>();
    auto red=mb.Get<float>();auto seg=red[256];auto cmp=cb.Get<uint8_t>();auto cmp32=cb.Get<uint32_t>();
    auto idx=xb.Get<int32_t>();auto cidx=zb.Get<int32_t>();auto thr=tb.Get<float>();
    auto cval=vb.Get<float>();auto cid=wb.Get<int32_t>();
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    const int rows=batch*6;
    const int L=lanes(blocks);
    int team=L/rows;if(team<1)team=1;
    const int SEGSTRIDE=(((width+63)/64)+7)/8*8;
    const int T0_BASE=rows*SEGSTRIDE;
    const int CAND_BASE=T0_BASE+rows*8;
    const int CSTRIDE=1032;
    const int me=lane();
    const int myrow=(me<rows*team)?me/team:-1;
    const int myidx=(me<rows*team)?me%team:0;
    GlobalTensor<float> wf;wf.SetGlobalBuffer((__gm__ float*)wsp);
    GlobalTensor<int32_t> wi;wi.SetGlobalBuffer((__gm__ int32_t*)wsp);
    int64_t limit=0;
    if(myrow>=0){int b=myrow/6,t=myrow%6;
        limit=(ac[b]&&sl[b]>=0&&sl[b]<nslots&&st[b]>=0)?(st[b]+t+1)/ratio:0;
        if(limit>width)limit=width;}
    for(int off=myidx*SELCHUNK;off<limit;off+=team*SELCHUNK){
        int count=width-off;if(count>SELCHUNK)count=SELCHUNK;
        GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)sp+int64_t(myrow)*width+off);
        Duplicate(buf,-__builtin_inff(),SELCHUNK);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE2>();
        DataCopy(buf,src,count);fence<HardEvent::MTE2_V>();
        BlockReduceMax<float>(red,buf,SELCHUNK/64,64,1,1,8);PipeBarrier<PIPE_V>();
        BlockReduceMax<float>(seg,red,SELCHUNK/512,64,1,1,8);PipeBarrier<PIPE_V>();
        fence<HardEvent::V_MTE3>();
        DataCopy(wf[myrow*SEGSTRIDE+off/64],seg,SELCHUNK/64);
        fence<HardEvent::MTE3_V>();
    }
}
__global__ __aicore__ void sel_p2(GM_ADDR sp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,GM_ADDR wsp,int batch,int ratio,int width,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap);
    TPipe pipe;TBuf<TPosition::VECCALC> hb,ib,rb,ob,mb,cb,xb,zb,tb,vb,wb;
    pipe.InitBuffer(hb,512*4);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(rb,SELCHUNK*4);pipe.InitBuffer(ob,512*8);
    pipe.InitBuffer(mb,512*4);pipe.InitBuffer(cb,SELCHUNK/2);pipe.InitBuffer(xb,SELCHUNK*4);pipe.InitBuffer(zb,SELCHUNK*4);
    pipe.InitBuffer(tb,256);pipe.InitBuffer(vb,512*4);pipe.InitBuffer(wb,512*4);
    TBuf<TPosition::VECCALC> scb,sdb,stb2,sib,svb,sxb;
    pipe.InitBuffer(scb,SELCHUNK*8);pipe.InitBuffer(sdb,SELCHUNK*8);pipe.InitBuffer(stb2,SELCHUNK*8);
    pipe.InitBuffer(sib,SELCHUNK*4);pipe.InitBuffer(svb,512*4);pipe.InitBuffer(sxb,512*4);
    auto val=hb.Get<float>();auto ids=ib.Get<int32_t>();auto buf=rb.Get<float>();auto out=ob.Get<int64_t>();
    auto red=mb.Get<float>();auto seg=red[256];auto cmp=cb.Get<uint8_t>();auto cmp32=cb.Get<uint32_t>();
    auto idx=xb.Get<int32_t>();auto cidx=zb.Get<int32_t>();auto thr=tb.Get<float>();
    auto cval=vb.Get<float>();auto cid=wb.Get<int32_t>();
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    const int rows=batch*6;
    const int L=lanes(blocks);
    int team=L/rows;if(team<1)team=1;
    const int SEGSTRIDE=(((width+63)/64)+7)/8*8;
    const int T0_BASE=rows*SEGSTRIDE;
    const int CAND_BASE=T0_BASE+rows*8;
    const int CSTRIDE=1032;
    const int me=lane();
    const int myrow=(me<rows*team)?me/team:-1;
    const int myidx=(me<rows*team)?me%team:0;
    GlobalTensor<float> wf;wf.SetGlobalBuffer((__gm__ float*)wsp);
    GlobalTensor<int32_t> wi;wi.SetGlobalBuffer((__gm__ int32_t*)wsp);
    int64_t limit=0;
    if(myrow>=0){int b=myrow/6,t=myrow%6;
        limit=(ac[b]&&sl[b]>=0&&sl[b]<nslots&&st[b]>=0)?(st[b]+t+1)/ratio:0;
        if(limit>width)limit=width;}
    const int nblk=(int)((limit+SELCHUNK-1)/SELCHUNK);
    const int nsegv=nblk*(SELCHUNK/64);
    if(myrow>=0&&myidx==0&&nsegv>=512){
        auto sconc=scb.Get<float>();auto sdst=sdb.Get<float>();auto stmp=stb2.Get<float>();
        auto sidx=sib.Get<uint32_t>();auto sidxi=sib.Get<int32_t>();
        auto sval=svb.Get<float>();auto soid=sxb.Get<uint32_t>();
        // nsegv may exceed SELCHUNK once limit>64*SELCHUNK (ctx>131072). Slide an
        // overlapping window of <=SELCHUNK segment maxima and keep the smallest
        // per-window 512th value: each window's 512th max is <= the global 512th
        // max, so the min stays a valid (conservative) lower bound for phase 3.
        const int win=(nsegv<SELCHUNK)?nsegv:SELCHUNK;
        const int32_t rpt=(int32_t)(win/32);
        float t0=__builtin_inff();
        for(int base=0;;base+=win){
            if(base+win>nsegv)base=nsegv-win;
            fence<HardEvent::S_MTE2>();
            DataCopy(buf,wf[myrow*SEGSTRIDE+base],win);fence<HardEvent::MTE2_V>();
            Duplicate(sidxi,0,int32_t(win));PipeBarrier<PIPE_V>();
            Concat(sconc,buf,stmp,rpt);PipeBarrier<PIPE_V>();
            Sort<float,true>(sdst,sconc,sidx,stmp,rpt);PipeBarrier<PIPE_V>();
            Extract(sval,soid,sdst,int32_t(16));PipeBarrier<PIPE_V>();
            fence<HardEvent::V_S>();
            float v=sval.GetValue(511);
            if(v<t0)t0=v;
            fence<HardEvent::S_V>();
            if(base+win>=nsegv)break;
        }
        Duplicate(thr,t0,64);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE3>();
        DataCopy(wf[T0_BASE+myrow*8],thr,8);fence<HardEvent::MTE3_V>();
    } else if(myrow>=0&&myidx==0){
        int n=0;
        for(int off=0;off<limit;off+=SELCHUNK){
            int count=width-off;if(count>SELCHUNK)count=SELCHUNK;
            DataCopy(buf,wf[myrow*SEGSTRIDE+off/64],SELCHUNK/64);
            fence<HardEvent::MTE2_S>();
            for(int s=0;s*64<count;++s){
                float v=buf.GetValue(s);int id=off+s*64;
                if(!(v>-__builtin_inff()&&v<__builtin_inff()))continue;
                if(n<512){int at=n++;val.SetValue(at,v);ids.SetValue(at,id);
                    while(at>0){int par=(at-1)/2;
                        if(!source_worse(v,id,val.GetValue(par),ids.GetValue(par)))break;
                        val.SetValue(at,val.GetValue(par));ids.SetValue(at,ids.GetValue(par));at=par;
                        val.SetValue(at,v);ids.SetValue(at,id);}
                } else if(source_worse(val.GetValue(0),ids.GetValue(0),v,id)){
                    val.SetValue(0,v);ids.SetValue(0,id);source_sift(val,ids,n,0);}
            }
            fence<HardEvent::S_MTE2>();
        }
        float t0=(n>=512)?val.GetValue(0):-__builtin_inff();
        Duplicate(thr,t0,64);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE3>();
        DataCopy(wf[T0_BASE+myrow*8],thr,8);fence<HardEvent::MTE3_V>();
    }
}
__global__ __aicore__ void sel_p3(GM_ADDR sp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,GM_ADDR wsp,int batch,int ratio,int width,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap);
    TPipe pipe;TBuf<TPosition::VECCALC> hb,ib,rb,ob,mb,cb,xb,zb,tb,vb,wb;
    pipe.InitBuffer(hb,512*4);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(rb,SELCHUNK*4);pipe.InitBuffer(ob,512*8);
    pipe.InitBuffer(mb,512*4);pipe.InitBuffer(cb,SELCHUNK/2);pipe.InitBuffer(xb,SELCHUNK*4);pipe.InitBuffer(zb,SELCHUNK*4);
    pipe.InitBuffer(tb,256);pipe.InitBuffer(vb,512*4);pipe.InitBuffer(wb,512*4);
    auto val=hb.Get<float>();auto ids=ib.Get<int32_t>();auto buf=rb.Get<float>();auto out=ob.Get<int64_t>();
    auto red=mb.Get<float>();auto seg=red[256];auto cmp=cb.Get<uint8_t>();auto cmp32=cb.Get<uint32_t>();
    auto idx=xb.Get<int32_t>();auto cidx=zb.Get<int32_t>();auto thr=tb.Get<float>();
    auto cval=vb.Get<float>();auto cid=wb.Get<int32_t>();
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    const int rows=batch*6;
    const int L=lanes(blocks);
    int team=L/rows;if(team<1)team=1;
    const int SEGSTRIDE=(((width+63)/64)+7)/8*8;
    const int T0_BASE=rows*SEGSTRIDE;
    const int CAND_BASE=T0_BASE+rows*8;
    const int CSTRIDE=1032;
    const int me=lane();
    const int myrow=(me<rows*team)?me/team:-1;
    const int myidx=(me<rows*team)?me%team:0;
    GlobalTensor<float> wf;wf.SetGlobalBuffer((__gm__ float*)wsp);
    GlobalTensor<int32_t> wi;wi.SetGlobalBuffer((__gm__ int32_t*)wsp);
    int64_t limit=0;
    if(myrow>=0){int b=myrow/6,t=myrow%6;
        limit=(ac[b]&&sl[b]>=0&&sl[b]<nslots&&st[b]>=0)?(st[b]+t+1)/ratio:0;
        if(limit>width)limit=width;}
    int ncand=0;
    if(myrow>=0){
        DataCopy(thr,wf[T0_BASE+myrow*8],8);fence<HardEvent::MTE2_S>();
        float t0=thr.GetValue(0);
        fence<HardEvent::S_V>();
        Duplicate(thr,t0,64);CreateVecIndex(idx,(int32_t)0,SELCHUNK);
        Duplicate(val,-3.4e38f,512);Duplicate(ids,(int32_t)-1,512);PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();
        for(int off=myidx*SELCHUNK;off<limit;off+=team*SELCHUNK){
            int count=width-off;if(count>SELCHUNK)count=SELCHUNK;
            GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)sp+int64_t(myrow)*width+off);
            Duplicate(buf,-__builtin_inff(),SELCHUNK);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE2>();
            DataCopy(buf,src,count);fence<HardEvent::MTE2_V>();
            Duplicate(cidx,(int32_t)-1,SELCHUNK);PipeBarrier<PIPE_V>();
            for(int q=0;q<SELCHUNK/64;++q){Compare(cmp[q*32],buf[q*64],thr,CMPMODE::GE,64);}
            PipeBarrier<PIPE_V>();
            uint64_t rc=0;GatherMaskParams gp(1,1,8,8);
            for(int q=0;q<SELCHUNK/64;++q){GatherMask(cidx[q*64],idx[q*64],cmp32[q*8],true,(uint32_t)64,gp,rc);}
            PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();
            for(int q=0;q<SELCHUNK/64;++q)for(int j=q*64;j<q*64+64;++j){
                int lo=cidx.GetValue(j);if(lo<0)break;
                float v=buf.GetValue(lo);int id=off+lo;if(id>=limit)continue;
                if(!(v>-__builtin_inff()&&v<__builtin_inff()))continue;
                if(ncand<512){int at=ncand++;val.SetValue(at,v);ids.SetValue(at,id);
                    while(at>0){int par=(at-1)/2;
                        if(!source_worse(v,id,val.GetValue(par),ids.GetValue(par)))break;
                        val.SetValue(at,val.GetValue(par));ids.SetValue(at,ids.GetValue(par));at=par;
                        val.SetValue(at,v);ids.SetValue(at,id);}
                } else if(source_worse(val.GetValue(0),ids.GetValue(0),v,id)){
                    val.SetValue(0,v);ids.SetValue(0,id);source_sift(val,ids,ncand,0);}
            }
            fence<HardEvent::S_V>();
        }
        int sb=CAND_BASE+(myrow*team+myidx)*CSTRIDE;
        fence<HardEvent::S_MTE3>();
        DataCopy(wf[sb],val,512);DataCopy(wi[sb+512],ids,512);
        fence<HardEvent::MTE3_V>();
        Duplicate(thr,(float)ncand,64);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE3>();
        DataCopy(wf[sb+1024],thr,8);fence<HardEvent::MTE3_V>();
    }
}
__global__ __aicore__ void sel_p4(GM_ADDR sp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR op,GM_ADDR wsp,int batch,int ratio,int width,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap);
    TPipe pipe;TBuf<TPosition::VECCALC> hb,ib,rb,ob,mb,cb,xb,zb,tb,vb,wb;
    pipe.InitBuffer(hb,512*4);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(rb,SELCHUNK*4);pipe.InitBuffer(ob,512*8);
    pipe.InitBuffer(mb,512*4);pipe.InitBuffer(cb,SELCHUNK/2);pipe.InitBuffer(xb,SELCHUNK*4);pipe.InitBuffer(zb,SELCHUNK*4);
    pipe.InitBuffer(tb,256);pipe.InitBuffer(vb,512*4);pipe.InitBuffer(wb,512*4);
    TBuf<TPosition::VECCALC> pbk,pmg,pgg,pcc,ptt,pvv,pxx,pnn;
    pipe.InitBuffer(pbk,16*1024*4);pipe.InitBuffer(pmg,2048*2*4);pipe.InitBuffer(pgg,4*1024*4);
    pipe.InitBuffer(pcc,1024*4);pipe.InitBuffer(ptt,1024*4);pipe.InitBuffer(pvv,512*4);
    pipe.InitBuffer(pxx,512*4);pipe.InitBuffer(pnn,16*8*4);
    auto val=hb.Get<float>();auto ids=ib.Get<int32_t>();auto buf=rb.Get<float>();auto out=ob.Get<int64_t>();
    auto red=mb.Get<float>();auto seg=red[256];auto cmp=cb.Get<uint8_t>();auto cmp32=cb.Get<uint32_t>();
    auto idx=xb.Get<int32_t>();auto cidx=zb.Get<int32_t>();auto thr=tb.Get<float>();
    auto cval=vb.Get<float>();auto cid=wb.Get<int32_t>();
    auto sblk=pbk.Get<float>();auto smrg=pmg.Get<float>();auto sg=pgg.Get<float>();
    auto sconc=pcc.Get<float>();auto stmp=ptt.Get<float>();auto sval=pvv.Get<float>();
    auto soid=pxx.Get<uint32_t>();auto cntb=pnn.Get<float>();
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    const int rows=batch*6;
    const int L=lanes(blocks);
    int team=L/rows;if(team<1)team=1;
    const int SEGSTRIDE=(((width+63)/64)+7)/8*8;
    const int T0_BASE=rows*SEGSTRIDE;
    const int CAND_BASE=T0_BASE+rows*8;
    const int CSTRIDE=1032;
    const int me=lane();
    const int myrow=(me<rows*team)?me/team:-1;
    const int myidx=(me<rows*team)?me%team:0;
    GlobalTensor<float> wf;wf.SetGlobalBuffer((__gm__ float*)wsp);
    GlobalTensor<int32_t> wi;wi.SetGlobalBuffer((__gm__ int32_t*)wsp);
    int64_t limit=0;
    if(myrow>=0){int b=myrow/6,t=myrow%6;
        limit=(ac[b]&&sl[b]>=0&&sl[b]<nslots&&st[b]>=0)?(st[b]+t+1)/ratio:0;
        if(limit>width)limit=width;}
    if(myrow>=0&&myidx==0){
        for(int j=0;j<512;++j)out.SetValue(j,-1);
        const int MAXB=16;
        bool fastok=(team<=MAXB);
        int cnts[16];int tot=0;
        if(fastok){
            fence<HardEvent::S_MTE2>();
            for(int k=0;k<team;++k){int sb=CAND_BASE+(myrow*team+k)*CSTRIDE;DataCopy(cntb[k*8],wf[sb+1024],8);}
            fence<HardEvent::MTE2_S>();
            for(int k=0;k<team;++k){int c=(int)cntb.GetValue(k*8);if(c<0)c=0;if(c>512)c=512;cnts[k]=c;tot+=c;}
            if(tot<512)fastok=false;
        }
        if(fastok){
            int nb=0;
            fence<HardEvent::S_MTE2>();
            for(int k=0;k<team;++k){
                if(cnts[k]<=0)continue;
                int sb=CAND_BASE+(myrow*team+k)*CSTRIDE;
                DataCopy(cval,wf[sb],512);DataCopy(cid,wi[sb+512],512);
                fence<HardEvent::MTE2_V>();
                auto cidu=cid.ReinterpretCast<uint32_t>();
                Concat(sconc,cval,stmp,(int32_t)16);PipeBarrier<PIPE_V>();
                Sort<float,true>(sblk[nb*1024],sconc,cidu,stmp,(int32_t)16);PipeBarrier<PIPE_V>();
                fence<HardEvent::V_MTE2>();
                nb++;
            }
            int ng=0;
            for(int g=0;g<nb;g+=4){
                int m=nb-g;if(m>4)m=4;
                if(m==1){
                    DataCopy(sg[ng*1024],sblk[g*1024],1024);PipeBarrier<PIPE_V>();
                }else{
                    MrgSortSrcList<float> lst(sblk[g*1024],sblk[(g+1)*1024],
                        sblk[((m>2)?(g+2):g)*1024],sblk[((m>3)?(g+3):g)*1024]);
                    uint16_t ec[4]={512,512,512,512};uint32_t sn[4]={0,0,0,0};
                    MrgSort<float,false>(smrg,lst,ec,sn,(uint16_t)((1u<<m)-1u),(int32_t)1);
                    PipeBarrier<PIPE_V>();
                    DataCopy(sg[ng*1024],smrg,1024);PipeBarrier<PIPE_V>();
                }
                ng++;
            }
            if(ng==1){
                Extract(sval,soid,sg,(int32_t)16);
            }else{
                MrgSortSrcList<float> lst2(sg[0],sg[1024],
                    sg[((ng>2)?2:0)*1024],sg[((ng>3)?3:0)*1024]);
                uint16_t ec2[4]={512,512,512,512};uint32_t sn2[4]={0,0,0,0};
                MrgSort<float,false>(smrg,lst2,ec2,sn2,(uint16_t)((1u<<ng)-1u),(int32_t)1);
                PipeBarrier<PIPE_V>();
                Extract(sval,soid,smrg,(int32_t)16);
            }
            PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();
            for(int j=0;j<512;++j)out.SetValue(j,(int64_t)(int32_t)soid.GetValue(j));
        }else{
        int n=0;
        for(int k=0;k<team;++k){
            int sb=CAND_BASE+(myrow*team+k)*CSTRIDE;
            fence<HardEvent::S_MTE2>();
            DataCopy(thr,wf[sb+1024],8);DataCopy(cval,wf[sb],512);DataCopy(cid,wi[sb+512],512);
            fence<HardEvent::MTE2_S>();
            int cnt=(int)thr.GetValue(0);
            for(int q=0;q<cnt;++q){
                float v=cval.GetValue(q);int id=cid.GetValue(q);
                if(n<512){int at=n++;val.SetValue(at,v);ids.SetValue(at,id);
                    while(at>0){int par=(at-1)/2;
                        if(!source_worse(v,id,val.GetValue(par),ids.GetValue(par)))break;
                        val.SetValue(at,val.GetValue(par));ids.SetValue(at,ids.GetValue(par));at=par;
                        val.SetValue(at,v);ids.SetValue(at,id);}
                } else if(source_worse(val.GetValue(0),ids.GetValue(0),v,id)){
                    val.SetValue(0,v);ids.SetValue(0,id);source_sift(val,ids,n,0);}
            }
        }
        while(n>0){out.SetValue(n-1,(int64_t)ids.GetValue(0));--n;
            if(n){val.SetValue(0,val.GetValue(n));ids.SetValue(0,ids.GetValue(n));source_sift(val,ids,n,0);}
        }
        }
        GlobalTensor<int64_t> dst;dst.SetGlobalBuffer((__gm__ int64_t*)op+int64_t(myrow)*512);
        fence<HardEvent::S_MTE3>();DataCopy(dst,out,512);fence<HardEvent::MTE3_S>();
    }
}
extern "C" int dec_source_select(void* stream,void* scores,void* slots,void* start,void* active,
        void* out,void* workspace,int batch,int ratio,int width,int nslots) {
    if(batch<1||batch>8)return -1;
    if(ratio!=1&&ratio!=2)return -1;
    if(width<8||width>1048576||width%8)return -1;
    if(nslots<batch)return -1;
    void* ps[]={scores,slots,start,active,out,workspace};for(int i=0;i<6;++i)if(!ps[i])return -(20+i);
    for(int i=0;i<6;++i)if(uintptr_t(ps[i])&31)return -(30+i);
    const int blocks=40;
    auto A=(uint8_t*)scores;auto B=(uint8_t*)slots;auto C=(uint8_t*)start;auto D=(uint8_t*)active;
    auto E=(uint8_t*)out;auto F=(uint8_t*)workspace;
    sel_p1<<<(2*blocks),nullptr,stream>>>(A,B,C,D,E,F,batch,ratio,width,nslots,blocks);
    sel_p2<<<(2*blocks),nullptr,stream>>>(A,B,C,D,E,F,batch,ratio,width,nslots,blocks);
    sel_p3<<<(2*blocks),nullptr,stream>>>(A,B,C,D,E,F,batch,ratio,width,nslots,blocks);
    sel_p4<<<(2*blocks),nullptr,stream>>>(A,B,C,D,E,F,batch,ratio,width,nslots,blocks);
    return 0;
}



// HSI: reduced Full scores -> top 2047 earlier blocks + newest block.
// This is downstream of TP SUM. Pool storage survives all decoder FFNs.
__aicore__ inline void hsi_offer(LocalTensor<float> val,LocalTensor<int32_t> ids,
        int &n,int cap,float score,int id) {
    if(!(score>-__builtin_inff() && score<__builtin_inff()))return;
    if(n<cap) {
        int at=n++;
        while(at>0) {
            int par=(at-1)/2;
            if(!source_worse(score,id,val.GetValue(par),ids.GetValue(par)))break;
            val.SetValue(at,val.GetValue(par));ids.SetValue(at,ids.GetValue(par));at=par;
        }
        val.SetValue(at,score);ids.SetValue(at,id);
    } else if(source_worse(val.GetValue(0),ids.GetValue(0),score,id)) {
        val.SetValue(0,score);ids.SetValue(0,id);source_sift(val,ids,n,0);
    }
}
__global__ __aicore__ void hsi_pool(GM_ADDR sp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,
        GM_ADDR op,int batch,int width,int nslots,int blocks) {
    cache(slp);cache(stp);cache(ap);
    TPipe pipe;TBuf<TPosition::VECCALC> rb,mb,ob,xb,pbg,pmg,pac,pcc,ptt,pvv,pxx;
    // The pool keeps the 2047 highest-scoring blocks out of up to 16384, so this
    // is a running top-k, not a full sort.  Each round sorts one 2048-block tile
    // with the vector sort pipeline and merges it into the running winner list,
    // which keeps the cost at O(nblk) vector work instead of O(nblk) scalar heap
    // operations.
    constexpr int CH=2048;constexpr int NB=CH/8;constexpr int TILE=2048;
    constexpr int KEEP=2047;constexpr int HALF=1024;
    pipe.InitBuffer(rb,CH*4);pipe.InitBuffer(mb,TILE*4);
    pipe.InitBuffer(ob,HALF*8*8);pipe.InitBuffer(xb,TILE*4);
    pipe.InitBuffer(pbg,TILE*2*2*4);   // doubles as the four sorted runs
    pipe.InitBuffer(pmg,TILE*2*4);pipe.InitBuffer(pac,TILE*2*4);
    pipe.InitBuffer(pcc,HALF*4);pipe.InitBuffer(ptt,HALF*4);
    pipe.InitBuffer(pvv,HALF*4);pipe.InitBuffer(pxx,HALF*4);
    auto buf=rb.Get<float>();auto bmax=mb.Get<float>();auto out=ob.Get<int64_t>();
    auto o32=out.template ReinterpretCast<int32_t>();
    auto idx=xb.Get<int32_t>();auto idxu=xb.Get<uint32_t>();
    auto sbig=pbg.Get<float>();auto sblk=pbg.Get<float>();
    auto smrg=pmg.Get<float>();auto sacc=pac.Get<float>();
    auto sconc=pcc.Get<float>();auto stmp=ptt.Get<float>();
    auto sval=pvv.Get<float>();auto soid=pxx.Get<uint32_t>();
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    const float NEG=-__builtin_inff();
    for(int row=lane();row<batch*6;row+=lanes(blocks)) {
        int b=row/6,t=row%6;
        int64_t limit=(ac[b] && sl[b]>=0 && sl[b]<nslots && st[b]>=0)?st[b]+t+1:0;
        if(limit>width)limit=width;
        int newest=limit>0?int((limit-1)/8):-1;
        int cand=newest>0?newest:0;
        int nsel=cand<KEEP?cand:KEEP;
        GlobalTensor<int64_t> dst;dst.SetGlobalBuffer((__gm__ int64_t*)op+int64_t(row)*16384);
        for(int base=0;base<cand;base+=TILE) {
            int cnt=cand-base;if(cnt>TILE)cnt=TILE;
            Duplicate(bmax,NEG,TILE);PipeBarrier<PIPE_V>();
            for(int off=0;off<cnt*8;off+=CH) {
                int count=cnt*8-off;if(count>CH)count=CH;
                GlobalTensor<float> src;
                src.SetGlobalBuffer((__gm__ float*)sp+int64_t(row)*width+base*8+off);
                Duplicate(buf,NEG,CH);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE2>();
                DataCopy(buf,src,count);fence<HardEvent::MTE2_V>();
                BlockReduceMax<float>(bmax[off/8],buf,NB/8,64,1,1,8);PipeBarrier<PIPE_V>();
            }
            // Index rides through the sort so the block id survives the merge.
            CreateVecIndex(idx,int32_t(base),TILE);PipeBarrier<PIPE_V>();
            for(int k=0;k<4;++k) {
                Concat(sconc,bmax[k*512],stmp,16);PipeBarrier<PIPE_V>();
                Sort<float,true>(sblk[k*HALF],sconc,idxu[k*512],stmp,16);PipeBarrier<PIPE_V>();
            }
            MrgSortSrcList<float> lst(sblk,sblk[HALF],sblk[2*HALF],sblk[3*HALF]);
            uint16_t ec[4]={512,512,512,512};uint32_t sn[4]={0,0,0,0};
            MrgSort<float,false>(smrg,lst,ec,sn,(uint16_t)0xF,(int32_t)1);
            PipeBarrier<PIPE_V>();
            if(base==0) {DataCopy(sacc,smrg,TILE*2);PipeBarrier<PIPE_V>();}
            else {
                // Keep only the best TILE entries seen so far; sbig aliases the
                // sorted runs, which are already consumed at this point.
                MrgSortSrcList<float> lst2(sacc,smrg,sacc,smrg);
                uint16_t ec2[4]={TILE,TILE,0,0};uint32_t sn2[4]={0,0,0,0};
                MrgSort<float,false>(sbig,lst2,ec2,sn2,(uint16_t)0x3,(int32_t)1);
                PipeBarrier<PIPE_V>();
                DataCopy(sacc,sbig,TILE*2);PipeBarrier<PIPE_V>();
            }
        }
        for(int k=0;k<2;++k) {
            for(int z=0;z<HALF*16;z+=8192)Duplicate(o32[z],int32_t(-1),8192);
            PipeBarrier<PIPE_V>();
            if(nsel>k*HALF) {
                Extract(sval,soid,sacc[k*2*HALF],16);
                Extract(sval[512],soid[512],sacc[k*2*HALF+HALF],16);
                PipeBarrier<PIPE_V>();
            }
            fence<HardEvent::V_S>();
            int lo=k*HALF,hi=lo+HALF;if(hi>nsel)hi=nsel;
            for(int i=lo;i<hi;++i) {
                int64_t blk=int64_t(int32_t(soid.GetValue(i-lo)));
                for(int j=0;j<8;++j)out.SetValue((i-lo)*8+j,blk*8+j);
            }
            if(k==1)for(int j=0;j<8;++j) {
                int64_t id=int64_t(newest)*8+j;
                out.SetValue((HALF-1)*8+j,newest>=0 && id<limit?id:-1);
            }
            fence<HardEvent::S_MTE3>();DataCopy(dst[k*HALF*8],out,HALF*8);fence<HardEvent::MTE3_S>();
        }
    }
}

extern "C" int dec_hsi_pool(void* stream,void* scores,void* slots,void* start,void* active,void* pool,
        int batch,int width,int nslots) {
    if(batch<1||batch>4||width<8||width>1048576||width%8||nslots<batch)return -1;
    void* ps[]={scores,slots,start,active,pool};for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    hsi_pool<<<48,nullptr,stream>>>((GM_ADDR)scores,(GM_ADDR)slots,(GM_ADDR)start,(GM_ADDR)active,
        (GM_ADDR)pool,batch,width,nslots,24);return 0;
}

// Score only candidate IDs. Resolve logical IDs through page table or pending
// override; neither cube logits nor context-wide score buffers are read here.
__global__ __aicore__ void hsi_scores(GM_ADDR qp,GM_ADDR wp,GM_ADDR bankp,GM_ADDR ptp,
        GM_ADDR pendingp,GM_ADDR slp,GM_ADDR stp,GM_ADDR ap,GM_ADDR ip,GM_ADDR op,
        int batch,int nslots,int pages,int maxpages,int rpp,int blocks) {
    cache(slp);cache(stp);cache(ap);cache(ptp);cache(wp);cache(ip);Vec v;
    TBuf<TPosition::VECCALC> qb,kb,pb,rb,sb,bb,nb,tb;
    v.pipe.InitBuffer(qb,4*8*128*4);v.pipe.InitBuffer(kb,8*128*4);
    v.pipe.InitBuffer(pb,8*128*4);v.pipe.InitBuffer(rb,8*32*4);
    v.pipe.InitBuffer(sb,8*512*4);v.pipe.InitBuffer(bb,8*128*2);v.pipe.InitBuffer(nb,32*8);v.pipe.InitBuffer(tb,128*4);
    auto qc=qb.Get<float>();auto keys=kb.Get<float>();auto prod=pb.Get<float>();
    auto red=rb.Get<float>();auto scratch=sb.Get<float>();auto bf=bb.Get<bfloat16_t>();
    auto idbuf=nb.Get<int64_t>();auto tv=tb.Get<float>();
    auto wvec=tv;auto okb=tv[32];auto acc=tv[64];auto aux=tv[96];
    auto sl=(__gm__ int64_t*)slp;auto st=(__gm__ int64_t*)stp;auto ac=(__gm__ int64_t*)ap;
    auto pt=(__gm__ int64_t*)ptp;auto ids=(__gm__ int64_t*)ip;auto weights=(__gm__ uint16_t*)wp;
    int last=-1;float wf[4];
    for(int chunk=lane();chunk<batch*6*512;chunk+=lanes(blocks)) {
        int row=chunk/512,b=row/6,t=row%6,c0=(chunk%512)*32;
        int64_t slot=sl[b],origin=st[b],limit=origin+t+1;
        bool live=ac[b]&&slot>=0&&slot<nslots&&origin>=0;
        if(row!=last) {
            for(int h=0;h<4;++h) {
                GlobalTensor<bfloat16_t> q;q.SetGlobalBuffer((__gm__ bfloat16_t*)qp+(int64_t(row)*4+h)*128);
                for(int j=0;j<8;++j)DataCopy(bf[j*128],q,128);
                sync();Cast(qc[h*1024],bf,RoundMode::CAST_NONE,int32_t(1024));sync();
                union {uint32_t u;float f;} hw;hw.u=uint32_t(weights[int64_t(row)*4+h])<<16;wf[h]=hw.f;
            }
            last=row;
            fence<HardEvent::S_V>();
            for(int h=0;h<4;++h)Duplicate(wvec[h*8],wf[h],int32_t(8));
            PipeBarrier<PIPE_V>();
        }
        GlobalTensor<int64_t> idsrc;idsrc.SetGlobalBuffer(ids+int64_t(row)*16384+c0);
        DataCopy(idbuf,idsrc,32);fence<HardEvent::MTE2_S>();
        for(int base=0;base<32;base+=8) {
            for(int j=0;j<8;++j) {
                int64_t id=idbuf.GetValue(base+j);
                __gm__ bfloat16_t* key=nullptr;
                if(live && id>=0 && id<limit) {
                    if(id>=origin) {
                        int64_t rel=id-origin;
                        if(rel<6)key=(__gm__ bfloat16_t*)pendingp+(int64_t(b)*6+rel)*128;
                    } else if(id/rpp<maxpages) {
                        int64_t page=pt[slot*maxpages+id/rpp];
                        if(page>=0&&page<pages)key=(__gm__ bfloat16_t*)bankp+(page*rpp+id%rpp)*128;
                    }
                }
                okb.SetValue(j,key!=nullptr?1.f:0.f);
                if(key) {GlobalTensor<bfloat16_t> k;k.SetGlobalBuffer(key);DataCopy(bf[j*128],k,128);}
            }
            // Fully vectorized: no scalar read-back, no PIPE_ALL sync in the inner loop.
            fence<HardEvent::S_V>();fence<HardEvent::MTE2_V>();
            Cast(keys,bf,RoundMode::CAST_NONE,int32_t(1024));PipeBarrier<PIPE_V>();
            for(int h=0;h<4;++h) {
                Mul(prod,qc[h*1024],keys,int32_t(1024));PipeBarrier<PIPE_V>();
                // Fold each 128-wide dot onto 64 lanes, then one WholeReduceSum emits the
                // 8 packed sums for this head: 8 small ReduceSum calls -> 2 vector ops.
                Add(scratch,prod,prod[64],uint64_t(64),uint8_t(8),BinaryRepeatParams(1,1,1,8,16,16));
                PipeBarrier<PIPE_V>();
                WholeReduceSum(red[h*8],scratch,int32_t(64),int32_t(8),1,1,8);PipeBarrier<PIPE_V>();
            }
            Adds(acc,red,0.f,int32_t(32));PipeBarrier<PIPE_V>();
            Maxs(acc,acc,0.f,int32_t(32));PipeBarrier<PIPE_V>();
            Mul(acc,acc,wvec,int32_t(32));PipeBarrier<PIPE_V>();
            Add(acc,acc,acc[8],int32_t(8));Add(acc[16],acc[16],acc[24],int32_t(8));PipeBarrier<PIPE_V>();
            Add(acc,acc,acc[16],int32_t(8));PipeBarrier<PIPE_V>();
            Muls(acc,acc,0.015625f,int32_t(8));PipeBarrier<PIPE_V>();
            // Clamp keeps dead lanes finite so the arithmetic mask cannot make NaN.
            Mins(acc,acc,1e30f,int32_t(8));PipeBarrier<PIPE_V>();
            Mul(acc,acc,okb,int32_t(8));Adds(aux,okb,-1.f,int32_t(8));PipeBarrier<PIPE_V>();
            Muls(aux,aux,3.4e38f,int32_t(8));PipeBarrier<PIPE_V>();
            Add(v.d[base],acc,aux,int32_t(8));PipeBarrier<PIPE_V>();
        }
        v.store((__gm__ float*)op+int64_t(chunk)*32,v.d,32);
    }
}
extern "C" int dec_hsi_scores(void* stream,void* q,void* weight,void* bank,void* table,void* pending,
        void* slots,void* start,void* active,void* pool,void* out,
        int batch,int nslots,int pages,int maxpages,int rpp) {
    if(batch<1||batch>4||nslots<batch||pages<1||maxpages<1||rpp<1)return -1;
    void* ps[]={q,weight,bank,table,pending,slots,start,active,pool,out};
    for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    hsi_scores<<<80,nullptr,stream>>>((GM_ADDR)q,(GM_ADDR)weight,(GM_ADDR)bank,(GM_ADDR)table,
        (GM_ADDR)pending,(GM_ADDR)slots,(GM_ADDR)start,(GM_ADDR)active,(GM_ADDR)pool,(GM_ADDR)out,
        batch,nslots,pages,maxpages,rpp,40);return 0;
}

// Three-stage HSI selection. local_scores is dead after TP SUM and serves
// as caller-owned scratch: rows * (2048 maxima + 8 threshold) floats.
// Each maximum witnesses a DISTINCT group of 8 inputs. The 512th largest
// finite maximum is therefore <= the true 512th finite input score.
__global__ __aicore__ void hsi_select_p1(GM_ADDR sp,GM_ADDR wp,int rows) {
    TPipe pipe;TBuf<TPosition::VECCALC> bb,rb;
    pipe.InitBuffer(bb,2048*4);pipe.InitBuffer(rb,256*4);
    auto buf=bb.Get<float>();auto red=rb.Get<float>();
    const int team=48/rows,me=lane(),row=me/team,part=me%team;
    if(row>=rows)return;
    GlobalTensor<float> src,dst;
    src.SetGlobalBuffer((__gm__ float*)sp+int64_t(row)*16384);
    dst.SetGlobalBuffer((__gm__ float*)wp+row*2056);
    for(int off=part*2048;off<16384;off+=team*2048) {
        DataCopy(buf,src[off],2048);fence<HardEvent::MTE2_V>();
        BlockReduceMax<float>(red,buf,32,64,1,1,8);PipeBarrier<PIPE_V>();
        fence<HardEvent::V_MTE3>();DataCopy(dst[off/8],red,256);
        fence<HardEvent::MTE3_V>();fence<HardEvent::V_MTE2>();
    }
}
__global__ __aicore__ void hsi_select_p2(GM_ADDR wp,int rows) {
    const int row=lane();if(row>=rows)return;
    TPipe pipe;TBuf<TPosition::VECCALC> bb,ib,cb,tb,sb,mb,vb,xb;
    pipe.InitBuffer(bb,2048*4);pipe.InitBuffer(ib,2048*4);
    pipe.InitBuffer(cb,1024*4);pipe.InitBuffer(tb,4096*4);
    pipe.InitBuffer(sb,4096*4);pipe.InitBuffer(mb,4096*4);
    pipe.InitBuffer(vb,512*4);pipe.InitBuffer(xb,512*4);
    auto buf=bb.Get<float>();auto idx=ib.Get<uint32_t>();
    auto cat=cb.Get<float>();auto tmp=tb.Get<float>();
    auto runs=sb.Get<float>();auto merged=mb.Get<float>();
    auto vals=vb.Get<float>();auto ids=xb.Get<uint32_t>();
    GlobalTensor<float> ws;ws.SetGlobalBuffer((__gm__ float*)wp+row*2056);
    DataCopy(buf,ws,2048);fence<HardEvent::MTE2_S>();
    // Nonfinite maxima supply no valid witness. Discarding a group only
    // weakens the lower bound; it cannot discard a true finite top-512.
    for(int i=0;i<2048;++i) {
        float v=buf.GetValue(i);
        if(!(v>-__builtin_inff()&&v<__builtin_inff()))buf.SetValue(i,-__builtin_inff());
    }
    fence<HardEvent::S_V>();Duplicate(ib.Get<int32_t>(),0,2048);PipeBarrier<PIPE_V>();
    for(int k=0;k<4;++k) {
        Concat(cat,buf[k*512],tmp,16);PipeBarrier<PIPE_V>();
        Sort<float,true>(runs[k*1024],cat,idx[k*512],tmp,16);PipeBarrier<PIPE_V>();
    }
    MrgSortSrcList<float> list(runs,runs[1024],runs[2048],runs[3072]);
    uint16_t count[4]={512,512,512,512};uint32_t consumed[4]={0,0,0,0};
    MrgSort<float,false>(merged,list,count,consumed,(uint16_t)0xF,(int32_t)1);
    PipeBarrier<PIPE_V>();Extract(vals,ids,merged,16);PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();
    float t0=vals.GetValue(511);fence<HardEvent::S_V>();
    Duplicate(vals,t0,8);PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE3>();
    DataCopy(ws[2048],vals,8);fence<HardEvent::MTE3_V>();
}
// Stable top-512: score descending, logical KV row ascending on ties.
__global__ __aicore__ void hsi_select_p3(GM_ADDR sp,GM_ADDR ip,GM_ADDR op,GM_ADDR wp,int rows,int blocks) {
    cache(ip);TPipe pipe;TBuf<TPosition::VECCALC> hb,ib,rb,pb,ob,cb,xb,zb,tb;
    constexpr int CHUNK=1024;
    pipe.InitBuffer(hb,512*4);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(rb,CHUNK*4);
    pipe.InitBuffer(pb,CHUNK*8);pipe.InitBuffer(ob,512*8);
    pipe.InitBuffer(cb,CHUNK/2);pipe.InitBuffer(xb,CHUNK*4);
    pipe.InitBuffer(zb,CHUNK*4);pipe.InitBuffer(tb,64*4);
    auto val=hb.Get<float>();auto ids=ib.Get<int32_t>();auto buf=rb.Get<float>();auto out=ob.Get<int64_t>();
    auto pbuf=pb.Get<int64_t>();auto cmp=cb.Get<uint8_t>();auto cm32=cb.Get<uint32_t>();
    auto idx=xb.Get<int32_t>();auto keep=zb.Get<int32_t>();auto thr=tb.Get<float>();
    auto o32=out.template ReinterpretCast<int32_t>();
    CreateVecIndex(idx,int32_t(0),CHUNK);PipeBarrier<PIPE_V>();
    for(int row=lane();row<rows;row+=lanes(blocks)) {
        GlobalTensor<float> ws;ws.SetGlobalBuffer((__gm__ float*)wp+row*2056+2048);
        DataCopy(thr,ws,8);fence<HardEvent::MTE2_S>();
        const float lower=thr.GetValue(0);
        int n=0;
        Duplicate(o32,int32_t(-1),1024);PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();
        for(int off=0;off<16384;off+=CHUNK) {
            // A stale heap threshold is a lower bound. GE keeps every tie;
            // hsi_offer applies exact score-descending/logical-ID-ascending order.
            float t0=n>=512?val.GetValue(0):lower;
            if(t0<lower)t0=lower;
            fence<HardEvent::S_V>();Duplicate(thr,t0,64);Duplicate(keep,int32_t(-1),CHUNK);
            PipeBarrier<PIPE_V>();
            GlobalTensor<float> src;src.SetGlobalBuffer((__gm__ float*)sp+int64_t(row)*16384+off);
            GlobalTensor<int64_t> psrc;psrc.SetGlobalBuffer((__gm__ int64_t*)ip+int64_t(row)*16384+off);
            DataCopy(buf,src,CHUNK);DataCopy(pbuf,psrc,CHUNK);fence<HardEvent::MTE2_V>();
            // Before the heap fills, -inf is not eligible (hsi_offer rejects it).
            // Once full, GE is necessary to retain candidates tied at the cutoff.
            if(t0==-__builtin_inff()) {
                for(int q=0;q<CHUNK/64;++q)Compare(cmp[q*32],buf[q*64],thr,CMPMODE::GT,64);
            } else {
                for(int q=0;q<CHUNK/64;++q)Compare(cmp[q*32],buf[q*64],thr,CMPMODE::GE,64);
            }
            PipeBarrier<PIPE_V>();fence<HardEvent::V_S>();fence<HardEvent::MTE2_S>();
            for(int q=0;q<CHUNK/64;++q)for(int w=0;w<2;++w) {
                uint32_t bits=cm32.GetValue(q*8+w);
                while(bits) {
                    int bit=0;uint32_t scan=bits;
                    while(!(scan&1u)){scan>>=1;++bit;}
                    int at=q*64+w*32+bit;bits&=bits-1;
                    int64_t id=pbuf.GetValue(at);
                    if(id>=0)hsi_offer(val,ids,n,512,buf.GetValue(at),int(id));
                }
            }
            fence<HardEvent::S_MTE2>();fence<HardEvent::S_V>();
        }
        while(n>0) {
            out.SetValue(n-1,ids.GetValue(0));
            --n;if(n) {val.SetValue(0,val.GetValue(n));ids.SetValue(0,ids.GetValue(n));source_sift(val,ids,n,0);}
        }
        GlobalTensor<int64_t> dst;dst.SetGlobalBuffer((__gm__ int64_t*)op+int64_t(row)*512);
        fence<HardEvent::S_MTE3>();DataCopy(dst,out,512);fence<HardEvent::MTE3_S>();
    }
}
extern "C" int dec_hsi_select(void* stream,void* scores,void* pool,void* out,void* workspace,int batch) {
    if(batch<1||batch>4)return -1;
    void* ps[]={scores,pool,out,workspace};for(auto p:ps)if(!p||(uintptr_t(p)&31))return -1;
    if(workspace==scores||workspace==pool||workspace==out)return -1;
    hsi_select_p1<<<48,nullptr,stream>>>((GM_ADDR)scores,(GM_ADDR)workspace,batch*6);
    hsi_select_p2<<<48,nullptr,stream>>>((GM_ADDR)workspace,batch*6);
    hsi_select_p3<<<48,nullptr,stream>>>((GM_ADDR)scores,(GM_ADDR)pool,(GM_ADDR)out,(GM_ADDR)workspace,batch*6,24);
    return 0;
}
