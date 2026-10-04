// 910B3 / dav-c220. Build command and caller-owned ABI: ../prefill/attention.py.
#include <kernel_operator.h>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
constexpr int DMAX = 512, KMAX = 512;
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

// Every kernel is vector-only: dav-c220 launches TWO AIVs per block.
__aicore__ inline int lane() { return GetBlockIdx(); }
__aicore__ inline int lanes(int blocks) { return 2 * blocks; }
__aicore__ inline float hi(float a, float b) { return a > b ? a : b; }
__aicore__ inline float ab(float a) { return a < 0 ? -a : a; }
__aicore__ inline float bits(uint32_t u) { union { uint32_t u; float f; } x; x.u=u; return x.f; }
__aicore__ inline uint32_t bits(float f) { union { uint32_t u; float f; } x; x.f=f; return x.u; }
__aicore__ inline float neginf() { return bits(uint32_t(0xff800000)); }
__aicore__ inline float powceil(float a) {
    uint32_t u=bits(a); return bits((u & 0x7f800000u) + ((u & 0x7fffffu) ? 0x800000u : 0u));
}
__aicore__ inline float e4(float a) {
    // Nonnegative E4M3FN, RNE including subnormals and overflow-to-NaN.
    if (a > 464.f) return bits(uint32_t(0x7fc00000));
    int e=int((bits(a)>>23)&255)-127;
    float step=bits(uint32_t((e-3 > -9 ? e-3 : -9)+127)<<23);
    float z=a/step; int n=int(z); float f=z-n;
    n += f>.5f || (f==.5f && (n&1));
    return n*step;
}
__aicore__ inline float e2(float a) {
    const float v[8]={0,.5f,1,1.5f,2,3,4,6};
    int j=0;
    for (int i=0;i<7;++i) {
        float m=(v[i]+v[i+1])*.5f;
        j += a>m || (a==m && (i&1));
    }
    return v[j];
}

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
        DataCopy(bf,g,n); sync();
        Cast(dst,bf,RoundMode::CAST_NONE,int32_t(n)); sync();
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
        Duplicate(z,x,int32_t(8)); sync();
        Exp(z,z,int32_t(8)); sync(); return z.GetValue(0);
    }
    __aicore__ inline float dot(int n) {
        Mul(c,a,b,int32_t(n)); sync();
        ReduceSum(z,c,w,int32_t(n)); sync(); return z.GetValue(0);
    }
};

template<bool PREFIX=false> __global__ __aicore__ void compress_kernel(GM_ADDR vp, GM_ADDR sp, GM_ADDR cvp,
        GM_ADDR csp, GM_ADDR pp, GM_ADDR op, GM_ADDR mp, GM_ADDR np, int T, int D, int B) {
    cache(pp); Vec v;
    auto x=(__gm__ float*)vp; auto s=(__gm__ float*)sp;
    auto cv=(__gm__ float*)cvp; auto cs=(__gm__ float*)csp;
    int64_t start=((__gm__ int64_t*)pp)[0], origin=start/2;
    int valid=T;
    if constexpr(PREFIX) { cache(np); valid=((__gm__ int64_t*)np)[0]; }
    int count=(start+valid)/2-origin;
    if (lane()==0) { ((__gm__ int64_t*)mp)[0]=origin; ((__gm__ int64_t*)mp)[1]=count; cache(mp); }
    GlobalTensor<float> gx, gs, gcv, gcs, go;
    gx.SetGlobalBuffer(x); gs.SetGlobalBuffer(s);
    gcv.SetGlobalBuffer(cv); gcs.SetGlobalBuffer(cs);
    go.SetGlobalBuffer((__gm__ float*)op);
    for (int j=lane();j<(T+1)/2;j+=lanes(B)) {
        int i=2*j-int(start%2);
        if (j<count) {
            if(i<0) {
                DataCopy(v.a,gcv[((start-1)%4)*D],D);
                DataCopy(v.c,gcs[((start-1)%4)*D],D);
            } else {
                DataCopy(v.a,gx[int64_t(i)*D],D);
                DataCopy(v.c,gs[int64_t(i)*D],D);
            }
            DataCopy(v.b,gx[int64_t(i+1)*D],D);
            DataCopy(v.d,gs[int64_t(i+1)*D],D);
            fence<HardEvent::MTE2_V>();
            Max(v.z,v.c,v.d,int32_t(D)); PipeBarrier<PIPE_V>();
            Sub(v.c,v.c,v.z,int32_t(D)); Sub(v.d,v.d,v.z,int32_t(D)); PipeBarrier<PIPE_V>();
            Exp(v.c,v.c,int32_t(D)); Exp(v.d,v.d,int32_t(D)); PipeBarrier<PIPE_V>();
            Add(v.z,v.c,v.d,int32_t(D)); PipeBarrier<PIPE_V>();
            Div(v.c,v.c,v.z,int32_t(D)); Div(v.d,v.d,v.z,int32_t(D)); PipeBarrier<PIPE_V>();
            Mul(v.a,v.a,v.c,int32_t(D)); Mul(v.b,v.b,v.d,int32_t(D)); PipeBarrier<PIPE_V>();
            Add(v.d,v.a,v.b,int32_t(D)); PipeBarrier<PIPE_V>();
        } else {
            Duplicate(v.d,0.f,int32_t(D));
        }
        fence<HardEvent::V_MTE3>();
        DataCopy(go[int64_t(j)*D],v.d,D);
        // Protect both next DMA loads and zero-fill from output buffer reuse.
        fence<HardEvent::MTE3_MTE2>();
        fence<HardEvent::MTE3_V>();
    }
}
template<int TILE, bool PREFIX=false> __global__ __aicore__ void compress_tile_kernel(GM_ADDR vp, GM_ADDR sp, GM_ADDR cvp,
        GM_ADDR csp, GM_ADDR pp, GM_ADDR op, GM_ADDR mp, GM_ADDR np, int T, int D, int B) {
    cache(pp);
    constexpr int CAP=TILE*DMAX;
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,5*CAP*4);
    auto va=buf.Get<float>(); auto vb=va[CAP]; auto vc=va[2*CAP];
    auto vd=va[3*CAP]; auto vz=va[4*CAP];
    int64_t start=((__gm__ int64_t*)pp)[0], origin=start/2;
    int valid=T;
    if constexpr(PREFIX) { cache(np); valid=((__gm__ int64_t*)np)[0]; }
    int count=(start+valid)/2-origin, rows=(T+1)/2;
    if(lane()==0) { ((__gm__ int64_t*)mp)[0]=origin; ((__gm__ int64_t*)mp)[1]=count; cache(mp); }
    GlobalTensor<float> gx,gs,gcv,gcs,go;
    gx.SetGlobalBuffer((__gm__ float*)vp); gs.SetGlobalBuffer((__gm__ float*)sp);
    gcv.SetGlobalBuffer((__gm__ float*)cvp); gcs.SetGlobalBuffer((__gm__ float*)csp);
    go.SetGlobalBuffer((__gm__ float*)op);
    for(int j=lane()*TILE;j<rows;j+=lanes(B)*TILE) {
        int total=rows-j<TILE?rows-j:TILE;
        int n=count-j; n=n<0?0:(n<total?n:total);
        int i=2*j-int(start%2);
        if(n>0) {
            int first=0;
            if(i<0) {
                DataCopy(va,gcv[((start-1)%4)*D],D);
                DataCopy(vc,gcs[((start-1)%4)*D],D);
                first=1;
            }
            if(n>first) {
                DataCopyParams cp{uint16_t(n-first),uint16_t(D/8),uint16_t(D/8),0};
                DataCopy(va[first*D],gx[int64_t(i+2*first)*D],cp);
                DataCopy(vc[first*D],gs[int64_t(i+2*first)*D],cp);
            }
            DataCopyParams cp{uint16_t(n),uint16_t(D/8),uint16_t(D/8),0};
            DataCopy(vb,gx[int64_t(i+1)*D],cp);
            DataCopy(vd,gs[int64_t(i+1)*D],cp);
            fence<HardEvent::MTE2_V>();
            int32_t k=n*D;
            Max(vz,vc,vd,k); PipeBarrier<PIPE_V>();
            Sub(vc,vc,vz,k); Sub(vd,vd,vz,k); PipeBarrier<PIPE_V>();
            Exp(vc,vc,k); Exp(vd,vd,k); PipeBarrier<PIPE_V>();
            Add(vz,vc,vd,k); PipeBarrier<PIPE_V>();
            Div(vc,vc,vz,k); Div(vd,vd,vz,k); PipeBarrier<PIPE_V>();
            Mul(va,va,vc,k); Mul(vb,vb,vd,k); PipeBarrier<PIPE_V>();
            Add(vd,va,vb,k); PipeBarrier<PIPE_V>();
        }
        if(n<total) Duplicate(vd[n*D],0.f,int32_t((total-n)*D));
        fence<HardEvent::V_MTE3>();
        DataCopy(go[int64_t(j)*D],vd,total*D);
        fence<HardEvent::MTE3_MTE2>(); fence<HardEvent::MTE3_V>();
    }
}
template<bool PREFIX=false> __global__ __aicore__ void carry_kernel(GM_ADDR vp, GM_ADDR sp, GM_ADDR cvp,
        GM_ADDR csp, GM_ADDR pp, GM_ADDR np, int T, int D, int B) {
    cache(pp); Vec v; int64_t start=((__gm__ int64_t*)pp)[0];
    int valid=T;
    if constexpr(PREFIX) { cache(np); valid=((__gm__ int64_t*)np)[0]; }
    int begin=valid>4 ? valid-4 : 0;
    for (int i=begin+lane();i<valid;i+=lanes(B)) {
        v.load(v.a,(__gm__ float*)vp+int64_t(i)*D,D);
        v.load(v.b,(__gm__ float*)sp+int64_t(i)*D,D);
        v.store((__gm__ float*)cvp+((start+i)%4)*D,v.a,D);
        v.store((__gm__ float*)csp+((start+i)%4)*D,v.b,D);
    }
}

template<int MODE, int TILE=32> __global__ __aicore__ void qdq_kernel(GM_ADDR xp, GM_ADDR yp, int N, int B) {
    Vec v; constexpr int W=MODE==1 ? 16 : 32;
    // Batch DMA/casts; preserve per-block scale and scalar rounding.
    for (int row=lane();row<(N+TILE-1)/TILE;row+=lanes(B)) {
        int off=row*TILE, n=N-off<TILE ? N-off : TILE;
        v.load(v.a,(__gm__ bfloat16_t*)xp+off,n);
        for (int base=0;base<n;base+=W) {
            float m=0; for (int j=0;j<W;++j) m=hi(m,ab(v.a.GetValue(base+j)));
            float scale;
            if constexpr (MODE==0) scale=powceil(hi(m,1e-4f)/448.f);
            if constexpr (MODE==1) scale=e4(hi(m,6.f/512.f)/6.f);
            if constexpr (MODE==2) scale=powceil(hi(m,6.f*bits(uint32_t(0x00800000)))/6.f);
            for (int j=0;j<W;++j) {
                float x=v.a.GetValue(base+j), u=ab(x/scale), y;
                if constexpr (MODE==0) y=e4(u>448.f ? 448.f : u);
                else y=e2(u>6.f ? 6.f : u);
                v.b.SetValue(base+j,(bits(x)&0x80000000u ? -y : y)*scale);
            }
        }
        v.store((__gm__ bfloat16_t*)yp+off,v.b,n);
    }
}

template<int SIGN> __global__ __aicore__ void rope_kernel(GM_ADDR xp, GM_ADDR fp,
        GM_ADDR yp, int T, int H, int D, int R, int B) {
    // Batch contiguous rows; all offsets depend only on bound geometry.
    constexpr int M=512;
    int tile=R<=64 ? 8 : M/R;
    TPipe pipe; TBuf<TPosition::VECCALC> xb, bb, fb, wb, ib;
    pipe.InitBuffer(xb,8*512*4); pipe.InitBuffer(bb,8*512*2);
    pipe.InitBuffer(fb,M*4); pipe.InitBuffer(wb,6*M*4); pipe.InitBuffer(ib,5*M*4);
    auto x=xb.Get<float>(); auto bf=bb.Get<bfloat16_t>(); auto f=fb.Get<float>();
    auto av=wb.Get<float>(), bv=av[M], c=av[2*M], d=av[3*M], z=av[4*M], sign=av[5*M];
    auto ia=ib.Get<uint32_t>(), ibb=ia[M], ic=ia[2*M], id=ia[3*M], io=ia[4*M];
    for(int k=0;k<tile*R;++k) {
        int row=k/R, q=k%R, base=row*D+D-R;
        ia.SetValue(k,(base+q/2*2)*4); ibb.SetValue(k,(base+q/2*2+1)*4);
        ic.SetValue(k,k*4); id.SetValue(k,(k^1)*4); io.SetValue(k,(base+q)*4);
        z.SetValue(k,(q&1)?1.f:-float(SIGN)); sign.SetValue(k,(q&1)?float(SIGN):1.f);
    }
    fence<HardEvent::S_V>();
    GlobalTensor<bfloat16_t> gx,gy; GlobalTensor<float> gf;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)xp); gy.SetGlobalBuffer((__gm__ bfloat16_t*)yp);
    gf.SetGlobalBuffer((__gm__ float*)fp);
    for(int job=lane();job<(T*H+tile-1)/tile;job+=lanes(B)) {
        int row=job*tile, nr=T*H-row<tile?T*H-row:tile, n=nr*R;
        DataCopy(bf,gx[int64_t(row)*D],nr*D);
        for(int j=0;j<nr;++j) DataCopy(f[j*R],gf[int64_t((row+j)/H)*R],R);
        fence<HardEvent::MTE2_V>();
        Cast(x,bf,RoundMode::CAST_NONE,int32_t(nr*D)); PipeBarrier<PIPE_V>();
        Gather(av,x,ia,uint32_t(0),n); Gather(bv,x,ibb,uint32_t(0),n);
        Gather(c,f,ic,uint32_t(0),n); Gather(d,f,id,uint32_t(0),n); PipeBarrier<PIPE_V>();
        // c=[cos,sin], d=[-sin,cos]; inverse flips each sin before the FMA.
        Mul(c,c,sign,int32_t(n));
        Mul(d,d,z,int32_t(n)); PipeBarrier<PIPE_V>();
        Mul(bv,bv,d,int32_t(n)); PipeBarrier<PIPE_V>();
        FusedMulAdd(av,c,bv,int32_t(n)); PipeBarrier<PIPE_V>();
        for(int j=0;j<nr;++j) DataCopy(x[j*D+D-R],av[j*R],R);
        PipeBarrier<PIPE_V>();
        Cast(bf,x,RoundMode::CAST_RINT,int32_t(nr*D)); fence<HardEvent::V_MTE3>();
        DataCopy(gy[int64_t(row)*D],bf,nr*D); fence<HardEvent::MTE3_MTE2>();
    }
}

// score tile [QT,KT], FP32 throughout; caller reduces this tile over TP FIRST.
template<bool CAND> __global__ __aicore__ void score_kernel(GM_ADDR qp, GM_ADDR wp,
        GM_ADDR kp, GM_ADDR pp, GM_ADDR np, GM_ADDR ip, GM_ADDR op,
        int T, int H, int G, int C, int R, int Q0, int K0, int QT, int KT, float scale, int B) {
    cache(pp); Vec v; auto pos=(__gm__ int64_t*)pp; auto ids=(__gm__ int64_t*)ip;
    int64_t valid=((__gm__ int64_t*)np)[0]; auto w=(__gm__ float*)wp;
    for (int chunk=lane();chunk<QT*KT/32;chunk+=lanes(B)) {
      for (int job=chunk*32;job<(chunk+1)*32;++job) {
        int t=Q0+job/KT, k=K0+job%KT; int64_t id=k;
        if constexpr(CAND) id=(t<T && k<C) ? ids[int64_t(t)*C+k] : -1;
        float result=neginf();
        if (t<T && k<C && pos[t]>=0 && id>=0 && id<G && id<valid && id<(pos[t]+1)/R) {
            result=0; v.load(v.b,(__gm__ bfloat16_t*)kp+id*128,128);
            for (int h=0;h<H;++h) {
                v.load(v.a,(__gm__ bfloat16_t*)qp+(int64_t(t)*H+h)*128,128);
                result += hi(v.dot(128),0.f)*w[int64_t(t)*H+h];
            }
            result *= scale;
        }
        v.d.SetValue(job-chunk*32,result);
      }
      v.store((__gm__ float*)op+chunk*32,v.d,32);
    }
}

// A min heap ordered by (score ascending, logical ID descending).
// Padding -1 is worse than every valid ID. Input candidates must be unique.
__aicore__ inline bool worse(float a, int64_t i, float b, int64_t j) {
    return a<b || (a==b && (i<0 || (j>=0 && i>j)));
}
template<int CAP> struct BasicHeap {
    TPipe pipe;
    TBuf<TPosition::VECCALC> sb, ib, ob, tb;
    LocalTensor<float> s, tile;
    LocalTensor<int64_t> id, out;
    __aicore__ inline BasicHeap() {
        pipe.InitBuffer(sb,CAP*4); pipe.InitBuffer(ib,CAP*8);
        pipe.InitBuffer(ob,CAP*8); pipe.InitBuffer(tb,32*4);
        s=sb.Get<float>(); id=ib.Get<int64_t>(); out=ob.Get<int64_t>(); tile=tb.Get<float>();
    }
    template<class U> __aicore__ inline void load(LocalTensor<U> x, __gm__ U* p, int n) {
        GlobalTensor<U> g; g.SetGlobalBuffer(p); DataCopy(x,g,n); sync();
    }
    template<class U> __aicore__ inline void store(__gm__ U* p, LocalTensor<U> x, int n) {
        GlobalTensor<U> g; g.SetGlobalBuffer(p); sync(); DataCopy(g,x,n); sync();
    }
    __aicore__ inline void down(int k, int n) {
        while (2*k+1<n) {
            int c=2*k+1;
            if (c+1<n && worse(s.GetValue(c+1),id.GetValue(c+1),s.GetValue(c),id.GetValue(c))) ++c;
            if (!worse(s.GetValue(c),id.GetValue(c),s.GetValue(k),id.GetValue(k))) break;
            float v=s.GetValue(k); int64_t i=id.GetValue(k);
            s.SetValue(k,s.GetValue(c)); id.SetValue(k,id.GetValue(c));
            s.SetValue(c,v); id.SetValue(c,i); k=c;
        }
    }
};
using Heap = BasicHeap<KMAX>;
extern "C" __global__ __aicore__ void init_kernel(GM_ADDR sp, GM_ADDR ip, int N, int B) {
    Heap h;
    for (int j=0;j<32;++j) { h.s.SetValue(j,neginf()); h.id.SetValue(j,-1); }
    for (int c=lane();c<N/32;c+=lanes(B)) {
        h.store((__gm__ float*)sp+c*32,h.s,32);
        h.store((__gm__ int64_t*)ip+c*32,h.id,32);
    }
}
template<bool CAND> __global__ __aicore__ void merge_kernel(GM_ADDR sp, GM_ADDR cp,
        GM_ADDR bp, GM_ADDR ip, int T, int C, int Q0, int K0, int QT, int KT, int K, int B) {
    cache(cp); Heap h; auto cand=(__gm__ int64_t*)cp;
    for (int row=lane();row<QT;row+=lanes(B)) {
        h.load(h.s,(__gm__ float*)bp+row*K,K);
        h.load(h.id,(__gm__ int64_t*)ip+row*K,K);
        for (int j0=0;j0<KT;j0+=32) {
            h.load(h.tile,(__gm__ float*)sp+row*KT+j0,32);
            for (int j=j0;j<j0+32;++j) {
                float s=h.tile.GetValue(j-j0); int64_t id=K0+j;
                if constexpr(CAND) id=(Q0+row<T && K0+j<C) ? cand[int64_t(Q0+row)*C+K0+j] : -1;
                if (s>neginf() && id>=0 && worse(h.s.GetValue(0),h.id.GetValue(0),s,id)) {
                    h.s.SetValue(0,s); h.id.SetValue(0,id); h.down(0,K);
                }
            }
        }
        h.store((__gm__ float*)bp+row*K,h.s,K);
        h.store((__gm__ int64_t*)ip+row*K,h.id,K);
    }
}
extern "C" __global__ __aicore__ void finish_kernel(GM_ADDR sp, GM_ADDR ip, GM_ADDR op,
        int T, int Q0, int QT, int K, int B) {
    Heap h;
    for (int row=lane();row<QT && Q0+row<T;row+=lanes(B)) {
        h.load(h.s,(__gm__ float*)sp+row*K,K);
        h.load(h.id,(__gm__ int64_t*)ip+row*K,K);
        for (int n=K;n>0;--n) {
            h.out.SetValue(n-1,h.id.GetValue(0));
            h.s.SetValue(0,h.s.GetValue(n-1)); h.id.SetValue(0,h.id.GetValue(n-1)); h.down(0,n-1);
        }
        h.store((__gm__ int64_t*)op+int64_t(Q0+row)*K,h.out,K);
    }
}

// Single joint online softmax over local + selected global + zero-valued sink.
// meta=[local_start, valid_global_rows]; negative positions are padded queries.
template<int RATIO> __global__ __aicore__ void sparse_kernel(GM_ADDR qp, GM_ADDR lp,
        GM_ADDR gp, GM_ADDR ip, GM_ADDR pp, GM_ADDR mp, GM_ADDR sp, GM_ADDR op,
        int T, int H, int D, int L, int G, int K, int W, float scale, int B) {
    cache(mp); Vec v; auto meta=(__gm__ int64_t*)mp; auto pos=(__gm__ int64_t*)pp;
    for (int job=lane();job<T*H;job+=lanes(B)) {
        int t=job/H, head=job%H; int64_t p=pos[t];
        v.load(v.a,(__gm__ bfloat16_t*)qp+int64_t(job)*D,D);
        Duplicate(v.d,0.f,int32_t(D)); sync();
        float mx=((__gm__ float*)sp)[head], den=1.f;
        for (int j=0;j<W+K;++j) {
            int64_t id=p-W+1+j-meta[0]; bool ok=p>=0 && j<W && p-W+1+j>=0 && id>=0 && id<L;
            __gm__ bfloat16_t* kv=(__gm__ bfloat16_t*)lp;
            if constexpr(RATIO>0) {
                if (j>=W) {
                    id=((__gm__ int64_t*)ip)[int64_t(t)*K+j-W]; kv=(__gm__ bfloat16_t*)gp;
                    ok=p>=0 && id>=0 && id<G && id<meta[1] && id<(p+1)/RATIO;
                }
            }
            if (ok) {
                v.load(v.b,kv+id*D,D); float logit=v.dot(D)*scale;
                float next=hi(mx,logit);
                float alpha=mx==next ? 1.f : v.exp(mx-next), beta=logit==next ? 1.f : v.exp(logit-next);
                Muls(v.d,v.d,alpha,int32_t(D)); sync();
                Axpy(v.d,v.b,beta,int32_t(D)); sync();
                den=den*alpha+beta; mx=next;
            }
        }
        Muls(v.d,v.d,1.f/den,int32_t(D)); sync();
        v.store((__gm__ bfloat16_t*)op+int64_t(job)*D,v.d,D);
    }
}

// Fixed exported leaves: host never decides a math path from run-time data.
#define P (uint8_t*)
extern "C" void pa_compress_prefix(void* st,void* v,void* s,void* cv,void* cs,void* p,void* o,void* m,void* n,int t,int d,int b) {
    if((t+1)/2>=2*b*8) compress_tile_kernel<8,true><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,P o,P m,P n,t,d,b);
    else compress_kernel<true><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,P o,P m,P n,t,d,b);
}
extern "C" void pa_carry_prefix(void* st,void* v,void* s,void* cv,void* cs,void* p,void* n,int t,int d,int b) {
    carry_kernel<true><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,P n,t,d,b);
}

extern "C" void pa_compress(void* st,void* v,void* s,void* cv,void* cs,void* p,void* o,void* m,int t,int d,int b) {
    if((t+1)/2>=2*b*8) compress_tile_kernel<8><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,P o,P m,nullptr,t,d,b);
    else compress_kernel<><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,P o,P m,nullptr,t,d,b);
}
extern "C" void pa_carry(void* st,void* v,void* s,void* cv,void* cs,void* p,int t,int d,int b) {
    carry_kernel<><<<b,nullptr,st>>>(P v,P s,P cv,P cs,P p,nullptr,t,d,b);
}
#define QDQ(NAME,MODE,TILE) extern "C" void NAME(void* st,void* x,void* y,int n,int b) { qdq_kernel<MODE,TILE><<<b,nullptr,st>>>(P x,P y,n,b); }
QDQ(pa_fp8,0,32) QDQ(pa_fp4e4,1,32) QDQ(pa_fp4pow2,2,32)

#define ROPE(NAME,SIGN) extern "C" void NAME(void* st,void* x,void* f,void* y,int t,int h,int d,int r,int b) { rope_kernel<SIGN><<<b,nullptr,st>>>(P x,P f,P y,t,h,d,r,b); }
ROPE(pa_rope,1) ROPE(pa_unrope,-1)
#define SCORE(NAME,CAND) extern "C" void NAME(void* st,void* q,void* w,void* k,void* p,void* n,void* i,void* o,int t,int h,int g,int c,int r,int q0,int k0,int qt,int kt,float s,int b) { score_kernel<CAND><<<b,nullptr,st>>>(P q,P w,P k,P p,P n,P i,P o,t,h,g,c,r,q0,k0,qt,kt,s,b); }
SCORE(pa_score,false)
extern "C" void pa_init(void* st,void* s,void* i,int n,int b) { init_kernel<<<b,nullptr,st>>>(P s,P i,n,b); }
#define MERGE(NAME,CAND) extern "C" void NAME(void* st,void* s,void* c,void* v,void* i,int t,int nc,int q0,int k0,int qt,int kt,int k,int b) { merge_kernel<CAND><<<b,nullptr,st>>>(P s,P c,P v,P i,t,nc,q0,k0,qt,kt,k,b); }
MERGE(pa_merge,false) MERGE(pa_merge_candidates,true)
extern "C" void pa_finish(void* st,void* s,void* i,void* o,int t,int q0,int qt,int k,int b) { finish_kernel<<<b,nullptr,st>>>(P s,P i,P o,t,q0,qt,k,b); }
#define SPARSE(NAME,R) extern "C" void NAME(void* st,void* q,void* l,void* g,void* i,void* p,void* m,void* s,void* o,int t,int h,int d,int nl,int ng,int k,int w,float scale,int b) { sparse_kernel<R><<<b,nullptr,st>>>(P q,P l,P g,P i,P p,P m,P s,P o,t,h,d,nl,ng,k,w,scale,b); }
SPARSE(pa_swa,0) SPARSE(pa_sparse1,1) SPARSE(pa_sparse2,2)

template<int TILE> __global__ __aicore__ void paged_read_kernel(
    GM_ADDR dp, GM_ADDR tp, GM_ADDR sp, GM_ADDR cp, GM_ADDR op,
    int N, int D, int R, int M, int B) {
    GlobalTensor<int64_t> table, slot, count;
    table.SetGlobalBuffer((__gm__ int64_t*)tp);
    slot.SetGlobalBuffer((__gm__ int64_t*)sp);
    count.SetGlobalBuffer((__gm__ int64_t*)cp);
    DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE,
                            DcciDst::CACHELINE_OUT>(table);
    PipeBarrier<PIPE_ALL>();
    int64_t s=slot.GetValue(0), n=count.GetValue(0);
    GlobalTensor<uint16_t> data, out;
    data.SetGlobalBuffer((__gm__ uint16_t*)dp);
    out.SetGlobalBuffer((__gm__ uint16_t*)op);
    TPipe pipe; TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, TILE*D*2);
    auto row=storage.Get<uint16_t>();
    for (int first=lane()*TILE;first<N;first+=lanes(B)*TILE) {
        int end=first+TILE<N?first+TILE:N;
        for (int i=first;i<end;) {
            int rows=end-i;
            if (i<n) {
                // Neither a physical page nor the valid prefix may be crossed.
                int remain=R-i%R;
                if(rows>remain) rows=remain;
                if(int64_t(rows)>n-i) rows=int(n-i);
                int64_t page=table.GetValue(s*M+i/R);
                DataCopy(row,data[(page*R+i%R)*D],rows*D);
                fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(row,uint16_t(0),int32_t(rows*D));
                fence<HardEvent::V_MTE3>();
            }
            DataCopy(out[int64_t(i)*D],row,rows*D);
            // Next segment can load a page or overwrite the buffer with zeros.
            fence<HardEvent::MTE3_MTE2>();
            fence<HardEvent::MTE3_V>();
            i+=rows;
        }
    }
}
extern "C" void paged_read(void* stream,void* data,void* table,void* slot,
    void* count,void* out,int n,int d,int r,int m,int b) {
    if(n>=2*b*16) paged_read_kernel<16><<<b,nullptr,stream>>>((uint8_t*)data,(uint8_t*)table,
        (uint8_t*)slot,(uint8_t*)count,(uint8_t*)out,n,d,r,m,b);
    else paged_read_kernel<1><<<b,nullptr,stream>>>((uint8_t*)data,(uint8_t*)table,
        (uint8_t*)slot,(uint8_t*)count,(uint8_t*)out,n,d,r,m,b);
}

// Caller-owned state movement. Metadata is read on device on every launch.
template<int TILE> __global__ __aicore__ void paged_write_kernel(
    GM_ADDR dp, GM_ADDR tp, GM_ADDR sp, GM_ADDR mp, GM_ADDR vp,
    int N, int D, int R, int M, int B) {
    cache(tp); auto tab=(__gm__ int64_t*)tp;
    int64_t slot=((__gm__ int64_t*)sp)[0];
    auto meta=(__gm__ int64_t*)mp;
    GlobalTensor<uint16_t> data, values;
    data.SetGlobalBuffer((__gm__ uint16_t*)dp);
    values.SetGlobalBuffer((__gm__ uint16_t*)vp);
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,TILE*D*2);
    auto row=buf.Get<uint16_t>();
    int64_t origin=meta[0], valid=meta[1];
    int count=valid<N?int(valid):N;
    for (int first=lane()*TILE;first<count;first+=lanes(B)*TILE) {
        int end=first+TILE<count?first+TILE:count;
        for (int i=first;i<end;) {
            int64_t p=origin+i, page=tab[slot*M+p/R];
            // Split at page boundaries; adjacent logical pages may be unrelated.
            int n=end-i, remain=R-int(p%R);
            if(n>remain) n=remain;
            DataCopy(row,values[int64_t(i)*D],n*D);
            fence<HardEvent::MTE2_MTE3>();
            DataCopy(data[(page*R+p%R)*D],row,n*D);
            fence<HardEvent::MTE3_MTE2>();
            i+=n;
        }
    }
}
extern "C" void pa_paged_write(void* st,void* d,void* t,void* s,void* m,void* v,
    int n,int dim,int r,int pages,int b) {
    if(n>=2*b*16) paged_write_kernel<16><<<b,nullptr,st>>>(P d,P t,P s,P m,P v,n,dim,r,pages,b);
    else paged_write_kernel<1><<<b,nullptr,st>>>(P d,P t,P s,P m,P v,n,dim,r,pages,b);
}

// Pack reads the old ring BEFORE publish. Publish emits only unique last rows.
template<bool PUBLISH, bool PREFIX=false> __global__ __aicore__ void window_kernel(
    GM_ADDR rp, GM_ADDR sp, GM_ADDR pp, GM_ADDR np, GM_ADDR vp, GM_ADDR op,
    int T,int D,int PAD,int RING,int HIST,int B) {
    cache(sp); int64_t slot=((__gm__ int64_t*)sp)[0], start=((__gm__ int64_t*)pp)[0];
    GlobalTensor<uint16_t> ring, values, out;
    ring.SetGlobalBuffer((__gm__ uint16_t*)rp);
    values.SetGlobalBuffer((__gm__ uint16_t*)vp);
    out.SetGlobalBuffer((__gm__ uint16_t*)op);
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,D*2*(PUBLISH?1:16));
    auto row=buf.Get<uint16_t>();
    if constexpr(PUBLISH) {
        int64_t n=((__gm__ int64_t*)np)[0];
        int end=PREFIX?int(n):T, begin=PREFIX?0:T-int(n);
        for(int i=lane();i<end;i+=lanes(B)) if(i>=begin && i>=end-RING && start+i>=0) {
            DataCopy(row,values[int64_t(i)*D],D); fence<HardEvent::MTE2_MTE3>();
            DataCopy(ring[(slot*(PAD+RING)+PAD+(start+i)%RING)*D],row,D); sync();
        }
    } else {
        // The history may wrap, but current KV is one contiguous span.
        for(int i=lane()*16;i<T;i+=lanes(B)*16) {
            int n=(T-i<16?T-i:16)*D;
            DataCopy(row,values[int64_t(i)*D],n); fence<HardEvent::MTE2_MTE3>();
            DataCopy(out[int64_t(HIST+i)*D],row,n); fence<HardEvent::MTE3_MTE2>();
        }
        fence<HardEvent::MTE3_V>();
        for(int i=lane();i<HIST;i+=lanes(B)) {
            int64_t p=start-HIST+i;
            if(p>=0) {
                DataCopy(row,ring[(slot*(PAD+RING)+PAD+p%RING)*D],D); fence<HardEvent::MTE2_MTE3>();
            } else {
                Duplicate(row,uint16_t(0),int32_t(D)); fence<HardEvent::V_MTE3>();
            }
            DataCopy(out[int64_t(i)*D],row,D); sync();
        }
    }
}
#define WINDOW(NAME,PUB) extern "C" void NAME(void* st,void* r,void* s,void* p,void* n,void* v,void* o,int t,int d,int pad,int ring,int hist,int b) { window_kernel<PUB><<<b,nullptr,st>>>(P r,P s,P p,P n,P v,P o,t,d,pad,ring,hist,b); }
WINDOW(pa_window_pack,false) WINDOW(pa_window_publish,true)
extern "C" void pa_window_publish_prefix(void* st,void* r,void* s,void* p,void* n,void* v,void* o,int t,int d,int pad,int ring,int hist,int b) { window_kernel<true,true><<<b,nullptr,st>>>(P r,P s,P p,P n,P v,P o,t,d,pad,ring,hist,b); }

template<bool PUBLISH> __global__ __aicore__ void slot_carry_kernel(
    GM_ADDR vp,GM_ADDR wp,GM_ADDR sp,GM_ADDR cp,GM_ADDR dp,int D,int B) {
    cache(sp); int64_t slot=((__gm__ int64_t*)sp)[0];
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,D*4);
    auto row=buf.Get<uint32_t>();
    for(int i=lane();i<8;i+=lanes(B)) {
        GlobalTensor<uint32_t> state, scratch;
        state.SetGlobalBuffer((__gm__ uint32_t*)(i<4?vp:wp));
        scratch.SetGlobalBuffer((__gm__ uint32_t*)(i<4?cp:dp));
        int r=i%4;
        if constexpr(PUBLISH) {
            DataCopy(row,scratch[r*D],D); fence<HardEvent::MTE2_MTE3>();
            DataCopy(state[(slot*4+r)*D],row,D);
        } else {
            DataCopy(row,state[(slot*4+r)*D],D); fence<HardEvent::MTE2_MTE3>();
            DataCopy(scratch[r*D],row,D);
        }
        sync();
    }
}
#define SLOT_CARRY(NAME,PUB) extern "C" void NAME(void* st,void* v,void* w,void* s,void* c,void* d,int dim,int b) { slot_carry_kernel<PUB><<<b,nullptr,st>>>(P v,P w,P s,P c,P d,dim,b); }
SLOT_CARRY(pa_carry_gather,false) SLOT_CARRY(pa_carry_publish,true)

// Lane zero owns metadata; position writers own disjoint 128-byte lines.
template<bool CED, bool PREFIX=false> __global__ __aicore__ void positions_kernel(
    GM_ADDR cp,GM_ADDR np,GM_ADDR pp,GM_ADDR sp,GM_ADDR mp,GM_ADDR rp,GM_ADDR vp,
    int T,int R,int HIST,int B) {
    cache(cp);
    int64_t start=((__gm__ int64_t*)cp)[0]-(CED?T:0);
    int64_t valid=T;
    if constexpr(PREFIX) { cache(vp); valid=((__gm__ int64_t*)vp)[0]; }
    if(lane()==0) {
        ((__gm__ int64_t*)sp)[0]=start;
        ((__gm__ int64_t*)mp)[0]=start-HIST;
        ((__gm__ int64_t*)mp)[1]=((__gm__ int64_t*)np)[0];
        // CED is read-only: it never publishes global rows.
        ((__gm__ int64_t*)rp)[0]=CED?0:start/R;
        ((__gm__ int64_t*)rp)[1]=CED?0:(start+valid)/R-start/R;
    }
    for(int first=lane()*16;first<T;first+=lanes(B)*16) {
        int end=first+16<T?first+16:T;
        for(int i=first;i<end;++i) ((__gm__ int64_t*)pp)[i]=i<valid?start+i:-1;
    }
    // Publish position lines and lane-zero metadata through the scalar cache.
    cache(pp);
}
#define POSITIONS(NAME,CED) extern "C" void NAME(void* st,void* c,void* n,void* p,void* s,void* m,void* r,int t,int ratio,int hist,int b) { positions_kernel<CED><<<b,nullptr,st>>>(P c,P n,P p,P s,P m,P r,nullptr,t,ratio,hist,b); }
POSITIONS(pa_positions_start,false) POSITIONS(pa_positions_end,true)
// Prefix-padded encoder: fixed storage, device valid count excludes padding.
extern "C" void pa_positions_prefix(void* st,void* c,void* n,void* p,void* s,
    void* m,void* r,void* v,int t,int ratio,int hist,int b) {
    positions_kernel<false,true><<<b,nullptr,st>>>(P c,P n,P p,P s,P m,P r,P v,t,ratio,hist,b);
}
template<int TILE> __global__ __aicore__ void freqs_kernel(
    GM_ADDR fp,GM_ADDR pp,GM_ADDR op,int T,int D,int B) {
    cache(pp); auto pos=(__gm__ int64_t*)pp;
    GlobalTensor<float> table,out; table.SetGlobalBuffer((__gm__ float*)fp); out.SetGlobalBuffer((__gm__ float*)op);
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,TILE*D*4); auto row=buf.Get<float>();
    for(int first=lane()*TILE;first<T;first+=lanes(B)*TILE) {
        int count=T-first<TILE?T-first:TILE;
        for(int j=0;j<count;) {
            int64_t p=pos[first+j];
            if(p>=0) {
                // Merge only positions proved consecutive on this replay.
                int n=1;
                while(j+n<count && pos[first+j+n]==p+n) ++n;
                DataCopy(row[j*D],table[p*D],n*D);
                j+=n;
            } else {
                for(int k=0;k<D;++k) row.SetValue(j*D+k,k%2?0.f:1.f);
                ++j;
            }
        }
        // Disjoint DMA/scalar producers complete before the single output DMA.
        fence<HardEvent::MTE2_MTE3>(); fence<HardEvent::S_MTE3>();
        DataCopy(out[int64_t(first)*D],row,count*D);
        fence<HardEvent::MTE3_S>(); fence<HardEvent::MTE3_MTE2>();
    }
}
extern "C" void pa_freqs(void* st,void* f,void* p,void* o,int t,int d,int b) {
    // Static geometry selects one kernel; device positions remain dynamic.
    if(t>=2*b*16) freqs_kernel<16><<<b,nullptr,st>>>(P f,P p,P o,t,d,b);
    else freqs_kernel<1><<<b,nullptr,st>>>(P f,P p,P o,t,d,b);
}

// CED: TP row-score SUM occurs before this block-max / top-2047 heap.
// The extra storage slot is DMA padding, NOT a 2048th historical block.
extern "C" __global__ __aicore__ void block_merge_kernel(
    GM_ADDR sp,GM_ADDR pp,GM_ADDR bp,GM_ADDR ip,
    int T,int R,int Q0,int K0,int QT,int KT,int B) {
    cache(pp); BasicHeap<2048> h; auto pos=(__gm__ int64_t*)pp;
    for(int row=lane();row<QT;row+=lanes(B)) {
        h.load(h.s,(__gm__ float*)bp+row*2048,2048);
        h.load(h.id,(__gm__ int64_t*)ip+row*2048,2048);
        int64_t lens=Q0+row<T && pos[Q0+row]>=0?(pos[Q0+row]+1)/R:0;
        int64_t newest=lens>0?(lens-1)/8:-1;
        for(int j0=0;j0<KT;j0+=32) {
            h.load(h.tile,(__gm__ float*)sp+row*KT+j0,32);
            for(int j=0;j<32;j+=8) {
                int64_t block=(K0+j0+j)/8; float score=neginf();
                for(int z=0;z<8;++z) score=hi(score,h.tile.GetValue(j+z));
                if(block<newest && score>neginf() && worse(h.s.GetValue(0),h.id.GetValue(0),score,block)) {
                    h.s.SetValue(0,score); h.id.SetValue(0,block); h.down(0,2047);
                }
            }
        }
        h.store((__gm__ float*)bp+row*2048,h.s,2048);
        h.store((__gm__ int64_t*)ip+row*2048,h.id,2048);
    }
}
extern "C" __global__ __aicore__ void block_finish_kernel(
    GM_ADDR sp,GM_ADDR ip,GM_ADDR pp,GM_ADDR np,GM_ADDR op,
    int T,int G,int R,int Q0,int QT,int B) {
    cache(pp); BasicHeap<2048> h; auto pos=(__gm__ int64_t*)pp;
    int64_t count=((__gm__ int64_t*)np)[0];
    for(int row=lane();row<QT && Q0+row<T;row+=lanes(B)) {
        h.load(h.s,(__gm__ float*)sp+row*2048,2048);
        h.load(h.id,(__gm__ int64_t*)ip+row*2048,2048);
        for(int n=2047;n>0;--n) {
            h.out.SetValue(n-1,h.id.GetValue(0));
            h.s.SetValue(0,h.s.GetValue(n-1)); h.id.SetValue(0,h.id.GetValue(n-1)); h.down(0,n-1);
        }
        int64_t lens=pos[Q0+row]>=0?(pos[Q0+row]+1)/R:0;
        h.out.SetValue(2047,lens>0?(lens-1)/8:-1);
        for(int c=0;c<16384;c+=32) {
            for(int j=0;j<32;++j) {
                int64_t block=h.out.GetValue((c+j)/8), id=block*8+(c+j)%8;
                h.id.SetValue(j,block>=0 && id<lens && id<count && id<G?id:-1);
            }
            h.store((__gm__ int64_t*)op+int64_t(Q0+row)*16384+c,h.id,32);
        }
    }
}
extern "C" void pa_block_merge(void* st,void* s,void* p,void* v,void* i,int t,int r,int q0,int k0,int qt,int kt,int b) {
    block_merge_kernel<<<b,nullptr,st>>>(P s,P p,P v,P i,t,r,q0,k0,qt,kt,b);
}
extern "C" void pa_block_finish(void* st,void* s,void* i,void* p,void* n,void* o,int t,int g,int r,int q0,int qt,int b) {
    block_finish_kernel<<<b,nullptr,st>>>(P s,P i,P p,P n,P o,t,g,r,q0,qt,b);
}

// Fused FA statistics have physical [H,T,8] layout despite [T,H,8] shape.
extern "C" __global__ __aicore__ void swa_correct_kernel(
    GM_ADDR op,GM_ADDR xp,GM_ADDR dp,GM_ADDR pp,GM_ADDR mp,
    int T,int H,int D,int W,int B) {
    cache(pp); cache(mp); cache(xp); cache(dp);
    auto pos=(__gm__ int64_t*)pp; auto meta=(__gm__ int64_t*)mp;
    auto mx=(__gm__ float*)xp; auto den=(__gm__ float*)dp;
    Vec v;
    // Test each position once, without assuming monotonic or unpadded rows.
    for(int t=lane();t<T;t+=lanes(B)) {
        const int64_t q=pos[t];
        if(q>=W-1) continue;
        for(int h=0;h<H;++h) {
            const int job=t*H+h;
            auto dst=(__gm__ bfloat16_t*)op+int64_t(job)*D;
            if(q<0) { Duplicate(v.a,0.f,int32_t(D)); sync(); v.store(dst,v.a,D); continue; }
            int64_t lo=q-W+1; if(lo<meta[0]) lo=meta[0];
            if(lo>=0) continue;
            int64_t stat=(int64_t(h)*T+t)*8;
            GlobalTensor<bfloat16_t> output; output.SetGlobalBuffer(dst);
            // Fetch the old output while computing the unchanged correction.
            DataCopy(v.bf,output,D);
            float maximum=mx[stat],sum=den[stat];
            Duplicate(v.z,-maximum,int32_t(8)); PipeBarrier<PIPE_V>();
            Exp(v.z,v.z,int32_t(8)); fence<HardEvent::V_S>();
            float keep=sum-float(-lo)*v.z.GetValue(0);
            fence<HardEvent::S_V>(); fence<HardEvent::MTE2_V>();
            Cast(v.a,v.bf,RoundMode::CAST_NONE,int32_t(D)); PipeBarrier<PIPE_V>();
            Muls(v.a,v.a,sum/keep,int32_t(D)); PipeBarrier<PIPE_V>();
            Cast(v.bf,v.a,RoundMode::CAST_RINT,int32_t(D));
            fence<HardEvent::V_MTE3>(); DataCopy(output,v.bf,D);
            // Both the zero branch and the next DMA reuse bf.
            fence<HardEvent::MTE3_V>(); fence<HardEvent::MTE3_MTE2>();
        }
    }
}
extern "C" void pa_swa_correct(void* st,void* o,void* x,void* d,void* p,void* m,
    int t,int h,int dim,int w,int b) {
    swa_correct_kernel<<<b,nullptr,st>>>((uint8_t*)o,(uint8_t*)x,(uint8_t*)d,
        (uint8_t*)p,(uint8_t*)m,t,h,dim,w,b);
}

// Matrix-score weighted head reduction; caller owns the fixed dot tile.
extern "C" __global__ __aicore__ void score_reduce_kernel(
    GM_ADDR dp, GM_ADDR wp, GM_ADDR pp, GM_ADDR np, GM_ADDR op,
    int T,int H,int G,int R,int Q0,int K0,int QT,int KT,float scale,int B) {
    cache(pp); Vec v;
    GlobalTensor<float> gd; gd.SetGlobalBuffer((__gm__ float*)dp);
    auto weight=(__gm__ float*)wp;
    auto pos=(__gm__ int64_t*)pp;
    int64_t valid=((__gm__ int64_t*)np)[0];
    int chunks=(KT+DMAX-1)/DMAX;
    for(int job=lane();job<QT*chunks;job+=lanes(B)) {
        int row=job/chunks, k=(job%chunks)*DMAX, t=Q0+row;
        int n=KT-k<DMAX ? KT-k : DMAX;
        int64_t limit=0;
        if(t<T && pos[t]>=0) {
            limit=(pos[t]+1)/R;
            if(limit>valid) limit=valid;
            if(limit>G) limit=G;
        }
        if(K0+k>=limit) {
            Duplicate(v.c,neginf(),int32_t(n)); sync();
        } else {
            Duplicate(v.c,0.f,int32_t(n)); sync();
            for(int h=0;h<H;++h) {
                fence<HardEvent::V_MTE2>();
                DataCopy(v.a,gd[(int64_t(row)*H+h)*KT+k],n);
                fence<HardEvent::MTE2_V>();
                Maxs(v.a,v.a,0.f,int32_t(n)); PipeBarrier<PIPE_V>();
                float head_weight=weight[int64_t(t)*H+h];
                Muls(v.a,v.a,head_weight,int32_t(n)); PipeBarrier<PIPE_V>();
                Add(v.c,v.c,v.a,int32_t(n)); PipeBarrier<PIPE_V>();
            }
            Muls(v.c,v.c,scale,int32_t(n)); sync();
            for(int j=int(limit-K0-k);j<n;++j)
                v.c.SetValue(j,neginf());
            sync();
        }
        v.store((__gm__ float*)op+int64_t(row)*KT+k,v.c,n);
    }
}
// One query row per AIV: load all four contiguous heads once, retaining
// the original FP32 head order. No change to score shape or TP summation.
extern "C" __global__ __aicore__ void score_reduce_row_kernel(
    GM_ADDR dp, GM_ADDR wp, GM_ADDR pp, GM_ADDR np, GM_ADDR op,
    int T,int G,int R,int Q0,int K0,int QT,int KT,float scale,int B) {
    cache(pp); TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf,5*KT*4);
    auto x=buf.Get<float>(); auto sum=x[4*KT];
    GlobalTensor<float> gd,go;
    gd.SetGlobalBuffer((__gm__ float*)dp); go.SetGlobalBuffer((__gm__ float*)op);
    auto weight=(__gm__ float*)wp; auto pos=(__gm__ int64_t*)pp;
    int64_t valid=((__gm__ int64_t*)np)[0];
    for(int row=lane();row<QT;row+=lanes(B)) {
        int t=Q0+row; int64_t limit=0;
        if(t<T && pos[t]>=0) {
            limit=(pos[t]+1)/R;
            if(limit>valid)limit=valid;
            if(limit>G)limit=G;
        }
        int n=int(limit-K0); if(n<0)n=0; if(n>KT)n=KT;
        if(n) {
            DataCopy(x,gd[int64_t(row)*4*KT],4*KT);
            fence<HardEvent::MTE2_V>();
            Maxs(x,x,0.f,int32_t(4*KT)); PipeBarrier<PIPE_V>();
            Duplicate(sum,0.f,int32_t(KT)); PipeBarrier<PIPE_V>();
            for(int h=0;h<4;++h) {
                float head_weight=weight[int64_t(t)*4+h];
                Muls(x[h*KT],x[h*KT],head_weight,int32_t(KT));
                PipeBarrier<PIPE_V>();
                Add(sum,sum,x[h*KT],int32_t(KT)); PipeBarrier<PIPE_V>();
            }
            Muls(sum,sum,scale,int32_t(KT)); PipeBarrier<PIPE_V>();
        }
        int aligned=(n+7)/8*8;
        if(aligned<KT)Duplicate(sum[aligned],neginf(),int32_t(KT-aligned));
        sync();
        for(int j=n;j<aligned;++j)sum.SetValue(j,neginf());
        sync(); DataCopy(go[int64_t(row)*KT],sum,KT); sync();
    }
}
extern "C" void pa_score_reduce(void* st,void* d,void* w,void* p,void* v,void* o,
    int t,int h,int g,int r,int q0,int k0,int qt,int kt,float scale,int b) {
    if(h==4 && kt<=8192)
        score_reduce_row_kernel<<<b,nullptr,st>>>((uint8_t*)d,(uint8_t*)w,(uint8_t*)p,
            (uint8_t*)v,(uint8_t*)o,t,g,r,q0,k0,qt,kt,scale,b);
    else
    score_reduce_kernel<<<b,nullptr,st>>>((uint8_t*)d,(uint8_t*)w,(uint8_t*)p,
        (uint8_t*)v,(uint8_t*)o,t,h,g,r,q0,k0,qt,kt,scale,b);
}

extern "C" __global__ __aicore__ void joint_pack_kernel(
 GM_ADDR lp,GM_ADDR gp,GM_ADDR ip,GM_ADDR pp,GM_ADDR mp,GM_ADDR kp,GM_ADDR jp,GM_ADDR cp,
 int T,int L,int G,int D,int K,int W,int R,int B) {
 cache(pp);cache(mp);Vec v;
 TBuf<TPosition::VECCALC> ib,jb;
 v.pipe.InitBuffer(ib,KMAX*8);v.pipe.InitBuffer(jb,(KMAX+128)*4);
 auto in=ib.Get<int64_t>();auto out=jb.Get<int32_t>();
 GlobalTensor<int64_t> gi;gi.SetGlobalBuffer((__gm__ int64_t*)ip);
 GlobalTensor<int32_t> gj;gj.SetGlobalBuffer((__gm__ int32_t*)jp);
 GlobalTensor<bfloat16_t> gl,gg,gk;
 gl.SetGlobalBuffer((__gm__ bfloat16_t*)lp);gg.SetGlobalBuffer((__gm__ bfloat16_t*)gp);gk.SetGlobalBuffer((__gm__ bfloat16_t*)kp);
 auto pos=(__gm__ int64_t*)pp;auto meta=(__gm__ int64_t*)mp;
 auto ids=(__gm__ int64_t*)ip;auto dst=(__gm__ int32_t*)jp;auto missing=(__gm__ int32_t*)cp;
 // Copy contiguous tiles instead of fencing each 512-element row.
 constexpr int COPY_ROWS=16;
 TBuf<TPosition::VECCALC> cb;v.pipe.InitBuffer(cb,COPY_ROWS*DMAX*2);
 auto tile=cb.Get<bfloat16_t>();
 for(int bank=0;bank<2;++bank) {
  const int rows=bank ? G : L,base=bank ? L : 0;
  for(int row=lane()*COPY_ROWS;row<rows;row+=lanes(B)*COPY_ROWS) {
   const int n=(rows-row<COPY_ROWS ? rows-row : COPY_ROWS)*D;
   if(bank)DataCopy(tile,gg[int64_t(row)*D],n);
   else DataCopy(tile,gl[int64_t(row)*D],n);
   fence<HardEvent::MTE2_MTE3>();
   DataCopy(gk[int64_t(base+row)*D],tile,n);
   fence<HardEvent::MTE3_MTE2>();
  }
 }
 if(lane()==0) {
  fence<HardEvent::MTE3_V>();
  Duplicate(tile.ReinterpretCast<uint16_t>(),uint16_t(0),int32_t(D));
  fence<HardEvent::V_MTE3>();DataCopy(gk[int64_t(L+G)*D],tile,D);
 }
 sync();
 // Distinct legal IDs for padded queries; correction discards their output.
 TBuf<TPosition::VECCALC> pb;v.pipe.InitBuffer(pb,(W+K)*4);
 auto padded=pb.Get<int32_t>();
 for(int j=0;j<W+K;++j)padded.SetValue(j,j<L+G+1 ? j : L+G);
 fence<HardEvent::S_V>();
 // Reuse the complete-window ramp; boundary rows keep scalar semantics.
 TBuf<TPosition::VECCALC> rb;v.pipe.InitBuffer(rb,128*4);
 auto ramp=rb.Get<float>();
 for(int j=0;j<W;++j)ramp.SetValue(j,float(j));
 fence<HardEvent::S_V>();
 const int64_t origin=meta[0],valid=meta[1];
 for(int t=lane();t<T;t+=lanes(B)) {
  DataCopy(in,gi[int64_t(t)*K],K);
  int miss=0;const int64_t p=pos[t],first=p-W+1;
  const int64_t start=first-origin;
  if(p>=0 && first>=0 && start>=0 && start<=int64_t(L)-W && L<=16777216) {
   Adds(v.a,ramp,float(start),int32_t(W));PipeBarrier<PIPE_V>();
   Cast(out,v.a,RoundMode::CAST_RINT,int32_t(W));PipeBarrier<PIPE_V>();
  } else {
   for(int j=0;j<W;++j) {
    const int64_t absolute=first+j,id=absolute-origin;
    const bool ok=p>=0 && absolute>=0 && id>=0 && id<L;
    out.SetValue(j,ok ? int32_t(id) : L+G);miss+=!ok;
   }
  }
  int64_t limit=(p+1)/R;
  if(limit>G) limit=G;
  if(limit>valid) limit=valid;
  if(p<0) limit=0;
  // Vector predicate for logical bank IDs; no narrowing of wide invalid IDs.
  fence<HardEvent::MTE2_V>();
  Cast(v.a,in,RoundMode::CAST_RINT,int32_t(K));PipeBarrier<PIPE_V>();
  Adds(v.b,v.a,1.f,int32_t(K));PipeBarrier<PIPE_V>();
  Maxs(v.b,v.b,0.f,int32_t(K));PipeBarrier<PIPE_V>();
  Mins(v.b,v.b,1.f,int32_t(K));
  Muls(v.c,v.a,-1.f,int32_t(K));PipeBarrier<PIPE_V>();
  Adds(v.c,v.c,float(limit),int32_t(K));PipeBarrier<PIPE_V>();
  Maxs(v.c,v.c,0.f,int32_t(K));PipeBarrier<PIPE_V>();
  Mins(v.c,v.c,1.f,int32_t(K));PipeBarrier<PIPE_V>();
  Mul(v.b,v.b,v.c,int32_t(K));PipeBarrier<PIPE_V>();
  ReduceSum(v.z,v.b,v.w,int32_t(K));PipeBarrier<PIPE_V>();
  Muls(v.z,v.z,-1.f,int32_t(1));PipeBarrier<PIPE_V>();
  Adds(v.z,v.z,float(K+miss),int32_t(1));PipeBarrier<PIPE_V>();
  auto count=v.d.ReinterpretCast<int32_t>();
  Cast(count,v.z,RoundMode::CAST_RINT,int32_t(1));PipeBarrier<PIPE_V>();
  Adds(v.a,v.a,float(L),int32_t(K));PipeBarrier<PIPE_V>();
  Mul(v.a,v.a,v.b,int32_t(K));
  Muls(v.c,v.b,-float(L+G),int32_t(K));PipeBarrier<PIPE_V>();
  Adds(v.c,v.c,float(L+G),int32_t(K));PipeBarrier<PIPE_V>();
  Add(v.a,v.a,v.c,int32_t(K));PipeBarrier<PIPE_V>();
  Cast(out[W],v.a,RoundMode::CAST_RINT,int32_t(K));
  PipeBarrier<PIPE_V>();
  if(p<0)Adds(out,padded,int32_t(0),int32_t(W+K));
  // Both scalar boundary IDs and vector IDs must reach the store pipe.
  fence<HardEvent::S_MTE3>();fence<HardEvent::V_MTE3>();
  DataCopy(gj[int64_t(t)*(W+K)],out,W+K);
  GlobalTensor<int32_t> gm;gm.SetGlobalBuffer((__gm__ int32_t*)cp);
  DataCopyPad(gm[int64_t(t)*32],count,DataCopyExtParams{1,4,0,0,0});sync();
 }
 cache(cp);
}
extern "C" void joint_pack(void* st,void* l,void* g,void* i,void* p,void* m,void* k,void* j,void* c,
 int t,int ln,int gn,int d,int kn,int w,int r,int b) {
 joint_pack_kernel<<<b,nullptr,st>>>((uint8_t*)l,(uint8_t*)g,(uint8_t*)i,(uint8_t*)p,(uint8_t*)m,
 (uint8_t*)k,(uint8_t*)j,(uint8_t*)c,t,ln,gn,d,kn,w,r,b);
}



extern "C" __global__ __aicore__ void sorted_finish_kernel(GM_ADDR sp,GM_ADDR ip,GM_ADDR op,
 int T,int Q0,int QT,int KT,int K,int B) {
 Heap h;
 for(int row=lane();row<QT && Q0+row<T;row+=lanes(B)) {
  int n=KT<K?KT:K;
  h.load(h.s,(__gm__ float*)sp+int64_t(row)*KT,n);
  // Stable descending scores: finite/+inf prefix, -inf padding suffix.
  int lo=0,hi=n;
  while(lo<hi) {int mid=(lo+hi)/2;
   if(h.s.GetValue(mid)>neginf())lo=mid+1;else hi=mid;
  }
  Duplicate(h.out.ReinterpretCast<int32_t>(),int32_t(-1),int32_t(K*2));sync();
  if(lo) {
   h.load(h.id,(__gm__ int64_t*)ip+int64_t(row)*KT,n);
   int aligned=lo/4*4;
   if(aligned)DataCopy(h.out,h.id,aligned);
   sync();
   for(int j=aligned;j<lo;++j)h.out.SetValue(j,h.id.GetValue(j));
  }
  h.store((__gm__ int64_t*)op+int64_t(Q0+row)*K,h.out,K);
 }
}

extern "C" void sorted_finish(void* st,void* s,void* i,void* o,int t,int q0,int qt,int kt,int k,int b) {
 sorted_finish_kernel<<<b,nullptr,st>>>((uint8_t*)s,(uint8_t*)i,(uint8_t*)o,t,q0,qt,kt,k,b);
}

// FP8 block32 round-trip: scalar scale per block, vector RNE per element.
__global__ __aicore__ void fp8_vector_kernel(GM_ADDR xp,GM_ADDR yp,int N,int B) {
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
            float s=powceil(hi(tmp.GetValue(0),1e-4f)/448.f);
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
extern "C" void pa_fp8_batch(void* st,void* x,void* y,int n,int b){fp8_vector_kernel<<<b,nullptr,st>>>((uint8_t*)x,(uint8_t*)y,n,b);}

// Eight independent candidates; identical per-dot ReduceSum and head order.
__global__ __aicore__ void score_candidates8_kernel(GM_ADDR qp, GM_ADDR wp,
 GM_ADDR kp, GM_ADDR pp, GM_ADDR np, GM_ADDR ip, GM_ADDR op,
 int T,int H,int G,int C,int R,int Q0,int K0,int QT,int KT,float scale,int B) {
 cache(pp); Vec v;
 TBuf<TPosition::VECCALC> qb,kb,pb,rb,sb,bb;
 v.pipe.InitBuffer(qb,4*8*128*4); v.pipe.InitBuffer(kb,8*128*4);
 v.pipe.InitBuffer(pb,8*128*4); v.pipe.InitBuffer(rb,8*32*4);
 v.pipe.InitBuffer(sb,8*512*4); v.pipe.InitBuffer(bb,8*128*2);
 auto qc=qb.Get<float>(); auto keys=kb.Get<float>(); auto prod=pb.Get<float>();
 auto red=rb.Get<float>(); auto scratch=sb.Get<float>(); auto bf=bb.Get<bfloat16_t>();
 auto pos=(__gm__ int64_t*)pp; auto ids=(__gm__ int64_t*)ip; auto weights=(__gm__ float*)wp;
 int64_t valid=((__gm__ int64_t*)np)[0]; int last=-1;
 for(int chunk=lane();chunk<QT*KT/32;chunk+=lanes(B)) {
  int t=Q0+(chunk*32)/KT;
  if(t<T && t!=last) {
   for(int h=0;h<4;++h) {
    GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer((__gm__ bfloat16_t*)qp+(int64_t(t)*4+h)*128);
    for(int j=0;j<8;++j) DataCopy(bf[j*128],g,128);
    sync(); Cast(qc[h*1024],bf,RoundMode::CAST_NONE,int32_t(1024)); sync();
   }
   last=t;
  }
  for(int base=0;base<32;base+=8) {
   bool ok[8]; float accum[8];
   Duplicate(keys,0.f,int32_t(1024)); sync();
   for(int j=0;j<8;++j) {
    int k=K0+(chunk*32+base+j)%KT;
    int64_t id=(t<T && k<C)?ids[int64_t(t)*C+k]:-1;
    ok[j]=t<T && k<C && pos[t]>=0 && id>=0 && id<G && id<valid && id<(pos[t]+1)/R;
    accum[j]=0.f;
    if(ok[j]) {
     GlobalTensor<bfloat16_t> g; g.SetGlobalBuffer((__gm__ bfloat16_t*)kp+id*128);
     DataCopy(bf[j*128],g,128);
    }
   }
   sync();
   // Invalid lanes' staging bytes are never consumed by the result.
   Cast(keys,bf,RoundMode::CAST_NONE,int32_t(1024)); sync();
   if(t<T) for(int h=0;h<4;++h) {
    Mul(prod,qc[h*1024],keys,int32_t(1024)); sync();
    for(int j=0;j<8;++j) ReduceSum(red[j*32],prod[j*128],scratch[j*512],int32_t(128));
    sync();
    float weight=weights[int64_t(t)*4+h];
    for(int j=0;j<8;++j) accum[j]+=hi(red.GetValue(j*32),0.f)*weight;
   }
   for(int j=0;j<8;++j) v.d.SetValue(base+j,ok[j]?accum[j]*scale:neginf());
  }
  v.store((__gm__ float*)op+chunk*32,v.d,32);
 }
}
extern "C" void pa_score_candidates(void* st,void* q,void* w,void* k,void* p,void* n,void* i,void* o,
 int t,int h,int g,int c,int r,int q0,int k0,int qt,int kt,float scale,int b) {
 score_candidates8_kernel<<<b,nullptr,st>>>((GM_ADDR)q,(GM_ADDR)w,(GM_ADDR)k,(GM_ADDR)p,(GM_ADDR)n,(GM_ADDR)i,(GM_ADDR)o,t,h,g,c,r,q0,k0,qt,kt,scale,b);
}

// Batch independent rows; preserve scalar factor expression and BF16 rounding.
__global__ __aicore__ void joint_correct_kernel(
 GM_ADDR op,GM_ADDR xp,GM_ADDR dp,GM_ADDR cp,GM_ADDR sp,int T,int H,int D,int N,int B) {
 cache(xp);cache(dp);cache(cp);cache(sp);Vec v;
 TBuf<TPosition::VECCALC> fb,bb;
 v.pipe.InitBuffer(fb,8*DMAX*4);v.pipe.InitBuffer(bb,8*DMAX*2);
 auto x=fb.Get<float>();auto bf=bb.Get<bfloat16_t>();
 auto mx=(__gm__ float*)xp;auto den=(__gm__ float*)dp;
 auto missing=(__gm__ int32_t*)cp;auto sink=(__gm__ float*)sp;
 GlobalTensor<bfloat16_t> out;out.SetGlobalBuffer((__gm__ bfloat16_t*)op);
 for(int first=lane()*8;first<T*H;first+=lanes(B)*8) {
  int count=T*H-first<8?T*H-first:8;
  // Start output DMA while the scalar correction factors are prepared.
  DataCopy(bf,out[int64_t(first)*D],count*D);
  float sums[8];int misses[8];
  Duplicate(v.z,0.f,int32_t(32));fence<HardEvent::V_S>();
  for(int j=0;j<count;++j) {
   int job=first+j,h=job%H;misses[j]=missing[(job/H)*32];sums[j]=den[job];
   float m=mx[job],z=hi(m,sink[h]);
   v.z.SetValue(j,m-z);v.z.SetValue(8+j,-z);v.z.SetValue(16+j,sink[h]-z);
  }
  fence<HardEvent::S_V>();Exp(v.z,v.z,int32_t(32));fence<HardEvent::V_S>();
  fence<HardEvent::MTE2_V>();
  Cast(x,bf,RoundMode::CAST_NONE,int32_t(count*D));PipeBarrier<PIPE_V>();
  for(int j=0;j<count;++j) {
   float factor=0.f;
   if(misses[j]<N) {
    float a=v.z.GetValue(j),s=sums[j];
    float keep=s*a-float(misses[j])*v.z.GetValue(8+j)+v.z.GetValue(16+j);
    factor=s*a/keep;
   }
   // Invalid queries may read unused KV containing NaN: multiplication by
   // zero is not a mask. Overwrite the discarded row without reading it.
   if(misses[j]<N)Muls(x[j*D],x[j*D],factor,int32_t(D));
   else Duplicate(x[j*D],0.f,int32_t(D));
  }
  PipeBarrier<PIPE_V>();Cast(bf,x,RoundMode::CAST_RINT,int32_t(count*D));
  fence<HardEvent::V_MTE3>();DataCopy(out[int64_t(first)*D],bf,count*D);
  fence<HardEvent::MTE3_MTE2>();
 }
}
extern "C" void joint_correct(void* st,void* o,void* x,void* d,void* c,void* s,int t,int h,int dim,int n,int b) {
 joint_correct_kernel<<<b,nullptr,st>>>((GM_ADDR)o,(GM_ADDR)x,(GM_ADDR)d,(GM_ADDR)c,(GM_ADDR)s,t,h,dim,n,b);
}

// FP4: exponent-derived half-ULP and RNE implements the E2M1 tie table.
template<int MODE> __global__ __aicore__ void fp4_vector_kernel(GM_ADDR xp,GM_ADDR yp,int N,int B) {
    constexpr int M=512, W=MODE==1?16:32;
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
        for(int j=0;j<n;j+=W){
            ReduceMax(tmp,a[j],y,int32_t(W),false);fence<HardEvent::V_S>();
            float m=tmp.GetValue(0), s;
            if constexpr(MODE==1) s=e4(hi(m,6.f/512.f)/6.f);
            else s=powceil(hi(m,6.f*bits(uint32_t(0x00800000)))/6.f);
            Duplicate(scale[j],s,int32_t(W));PipeBarrier<PIPE_V>();
        }
        Div(u,x,scale,int32_t(n));PipeBarrier<PIPE_V>();
        Mins(u,u,6.f,int32_t(n));PipeBarrier<PIPE_V>();
        Maxs(u,u,-6.f,int32_t(n));PipeBarrier<PIPE_V>();
        Abs(a,u,int32_t(n));PipeBarrier<PIPE_V>();
        Maxs(a,a,1.f,int32_t(n));PipeBarrier<PIPE_V>();
        ShiftRight(step.ReinterpretCast<uint32_t>(),a.ReinterpretCast<uint32_t>(),uint32_t(23),int32_t(n));PipeBarrier<PIPE_V>();
        ShiftLeft(step.ReinterpretCast<uint32_t>(),step.ReinterpretCast<uint32_t>(),uint32_t(23),int32_t(n));PipeBarrier<PIPE_V>();
        Muls(step,step,0.5f,int32_t(n));PipeBarrier<PIPE_V>();
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
extern "C" void pa_fp4e4_batch(void* st,void* x,void* y,int n,int b){fp4_vector_kernel<1><<<b,nullptr,st>>>((uint8_t*)x,(uint8_t*)y,n,b);}
extern "C" void pa_fp4pow2_batch(void* st,void* x,void* y,int n,int b){fp4_vector_kernel<2><<<b,nullptr,st>>>((uint8_t*)x,(uint8_t*)y,n,b);}


// Retain candidate membership in logical-row order; DMA owns output tiles.
__global__ __aicore__ void candidate_mask_kernel(GM_ADDR sp,GM_ADDR cp,GM_ADDR dp,
 int T,int Q0,int QT,int KT,int C,int G,int B) {
 constexpr int M=8192, IC=256;
 TPipe pipe;TBuf<TPosition::VECCALC> sb,ob,ib;
 pipe.InitBuffer(sb,M*4);pipe.InitBuffer(ob,M*4);pipe.InitBuffer(ib,IC*8);
 auto src=sb.Get<float>(),out=ob.Get<float>();auto ids=ib.Get<int64_t>();
 GlobalTensor<float> gs,gd;GlobalTensor<int64_t> gc;
 gs.SetGlobalBuffer((__gm__ float*)sp);gd.SetGlobalBuffer((__gm__ float*)dp);
 gc.SetGlobalBuffer((__gm__ int64_t*)cp);
 int tiles=(KT+M-1)/M;
 for(int job=lane();job<QT*tiles;job+=lanes(B)){
  int row=job/tiles,lo=(job%tiles)*M,n=KT-lo<M?KT-lo:M;
  Duplicate(out,neginf(),int32_t(n));fence<HardEvent::V_S>();
  if(Q0+row<T){
   DataCopy(src,gs[int64_t(row)*KT+lo],n);fence<HardEvent::MTE2_S>();
   for(int j=0;j<C;j+=IC){
    int count=C-j<IC?C-j:IC;
    DataCopy(ids,gc[int64_t(Q0+row)*C+j],count);fence<HardEvent::MTE2_S>();
    for(int k=0;k<count;++k){
     int64_t id=ids.GetValue(k);
     if(id>=lo && id<lo+n && id<G)out.SetValue(id-lo,src.GetValue(id-lo));
    }
    fence<HardEvent::S_MTE2>();
   }
  }
  fence<HardEvent::S_MTE3>();DataCopy(gd[int64_t(row)*KT+lo],out,n);
  fence<HardEvent::MTE3_V>();fence<HardEvent::MTE3_S>();
 }
}
extern "C" void pa_candidate_mask(void* st,void* src,void* ids,void* dst,
 int t,int q0,int qt,int kt,int c,int g,int b){
 candidate_mask_kernel<<<b,nullptr,st>>>((uint8_t*)src,(uint8_t*)ids,(uint8_t*)dst,t,q0,qt,kt,c,g,b);
}

// Full-bank block maxima; invalid/newest blocks are excluded before stable sort.
extern "C" __global__ __aicore__ void candidate_max_kernel(
 GM_ADDR sp,GM_ADDR pp,GM_ADDR op,int T,int R,int Q0,int QT,int KT,int NB,int B) {
 cache(pp);Vec v;auto pos=(__gm__ int64_t*)pp;
 for(int job=lane();job<QT*(NB/32);job+=lanes(B)) {
  int row=job/(NB/32), b0=job%(NB/32)*32,t=Q0+row;
  int64_t lens=t<T && pos[t]>=0?(pos[t]+1)/R:0;
  int64_t newest=lens>0?(lens-1)/8:-1;
  if(b0*8<KT)v.load(v.a,(__gm__ float*)sp+row*KT+b0*8,KT-b0*8<256?KT-b0*8:256);
  for(int b=0;b<32;++b) {
   float x=neginf();
   if((b0+b)*8<KT && b0+b<newest)
    for(int z=0;z<8;++z)x=hi(x,v.a.GetValue(b*8+z));
   v.c.SetValue(b,x);
  }
  v.store((__gm__ float*)op+row*NB+b0,v.c,32);
 }
}
extern "C" void pa_candidate_max(void* st,void* s,void* p,void* o,
 int t,int r,int q0,int qt,int kt,int nb,int b) {
 candidate_max_kernel<<<b,nullptr,st>>>((uint8_t*)s,(uint8_t*)p,(uint8_t*)o,t,r,q0,qt,kt,nb,b);
}
extern "C" __global__ __aicore__ void candidate_expand_kernel(
 GM_ADDR sp,GM_ADDR ip,GM_ADDR pp,GM_ADDR np,GM_ADDR op,
 int T,int G,int R,int Q0,int QT,int NB,int B) {
 cache(sp);cache(ip);cache(pp);cache(np);
 auto score=(__gm__ float*)sp;auto ids=(__gm__ int64_t*)ip;
 auto pos=(__gm__ int64_t*)pp;int64_t count=((__gm__ int64_t*)np)[0];
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,1024*8);
 auto out=buf.Get<int64_t>();GlobalTensor<int64_t> dst;dst.SetGlobalBuffer((__gm__ int64_t*)op);
 for(int job=lane();job<QT*16;job+=lanes(B)) {
  int row=job/16,t=Q0+row,c0=job%16*1024;if(t>=T)continue;
  int64_t lens=pos[t]>=0?(pos[t]+1)/R:0;
  for(int j=0;j<1024;++j) {
   int c=c0+j,k=c/8;int64_t block=-1;
   if(k==2047)block=lens>0?(lens-1)/8:-1;
   else if(k<NB && score[row*NB+k]>neginf())block=ids[row*NB+k];
   int64_t id=block*8+c%8;
   out.SetValue(j,block>=0 && id<lens && id<count && id<G?id:-1);
  }
  sync();DataCopy(dst[int64_t(t)*16384+c0],out,1024);sync();
 }
}
extern "C" void pa_candidate_expand(void* st,void* s,void* i,void* p,void* n,void* o,
 int t,int g,int r,int q0,int qt,int nb,int b) {
 candidate_expand_kernel<<<b,nullptr,st>>>((uint8_t*)s,(uint8_t*)i,(uint8_t*)p,(uint8_t*)n,(uint8_t*)o,t,g,r,q0,qt,nb,b);
}

// Encoder stable TopK: sort in UB and write only 512 IDs, no GM sorted tensors.
extern "C" __global__ __aicore__ void fused_topk_kernel(GM_ADDR xp,GM_ADDR op,int rows,int cols,int B){
 const int padded=cols<544?544:cols;
 TPipe pipe;TBuf<TPosition::VECCALC> xb,ib,ab,tb,ob,gb;
 pipe.InitBuffer(xb,padded*4);pipe.InitBuffer(ib,padded*4);
 pipe.InitBuffer(ab,padded*8);pipe.InitBuffer(tb,padded*8);pipe.InitBuffer(ob,512*8);pipe.InitBuffer(gb,1024*4);
 auto x=xb.Get<float>();auto idx=ib.Get<uint32_t>();auto a=ab.Get<float>();auto tmp=tb.Get<float>();auto out=ob.Get<int64_t>();
 auto offsets=gb.Get<uint32_t>();
 for(int j=0;j<512;++j){offsets.SetValue(2*j,8*j+4);offsets.SetValue(2*j+1,4096);}
 fence<HardEvent::S_V>();
 CreateVecIndex(idx.ReinterpretCast<int32_t>(),int32_t(0),uint32_t(padded));PipeBarrier<PIPE_V>();
 GlobalTensor<float> gx;gx.SetGlobalBuffer((__gm__ float*)xp);
 GlobalTensor<int64_t> go;go.SetGlobalBuffer((__gm__ int64_t*)op);
 for(int row=GetBlockIdx();row<rows;row+=2*B){
  Duplicate(x,-__builtin_inff(),int32_t(padded));PipeBarrier<PIPE_V>();fence<HardEvent::V_MTE2>();
  DataCopy(x,gx[int64_t(row)*cols],cols);fence<HardEvent::MTE2_V>();
  Sort<float,true>(a,x,idx,tmp,int32_t(padded/32));fence<HardEvent::V_S>();
  int lo=0,hi=512;
  while(lo<hi){int mid=(lo+hi)/2;if(a.GetValue(2*mid)>-__builtin_inff())lo=mid+1;else hi=mid;}
  a.SetValue(1024,0.f);fence<HardEvent::S_V>();
  Gather(out.ReinterpretCast<uint32_t>(),a.ReinterpretCast<uint32_t>(),offsets,uint32_t(0),uint32_t(1024));PipeBarrier<PIPE_V>();
  if(lo<512){
   auto pad=tmp.ReinterpretCast<int32_t>();Duplicate(pad,int32_t(-1),int32_t(1024));PipeBarrier<PIPE_V>();
   int aligned=(lo+3)/4*4;
   if(aligned<512)DataCopy(out[aligned],tmp.ReinterpretCast<int64_t>(),512-aligned);
   fence<HardEvent::V_S>();for(int j=lo;j<aligned;++j)out.SetValue(j,-1);
  }
  fence<HardEvent::V_MTE3>();fence<HardEvent::S_MTE3>();DataCopy(go[int64_t(row)*512],out,512);fence<HardEvent::MTE3_MTE2>();
 }
}
extern "C" void fused_topk(void* st,void* x,void* o,int rows,int cols,int b){fused_topk_kernel<<<b,nullptr,st>>>((uint8_t*)x,(uint8_t*)o,rows,cols,b);}

extern "C" __global__ __aicore__ void compact_reduce_kernel(
    GM_ADDR dp, GM_ADDR wp, GM_ADDR pp, GM_ADDR np, GM_ADDR op,
    int T,int H,int G,int R,int Q0,int K0,int QT,int KT,int OUTKT,float scale,int B) {
    cache(pp);
    TPipe pipe; TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 2*4096*4);
    struct { LocalTensor<float> a,c; } v;
    v.a=buffer.Get<float>(); v.c=v.a[4096];
    GlobalTensor<float> gd; gd.SetGlobalBuffer((__gm__ float*)dp);
    auto weight=(__gm__ float*)wp;
    auto pos=(__gm__ int64_t*)pp;
    int64_t valid=((__gm__ int64_t*)np)[0];
    int chunks=(KT+4096-1)/4096;
    for(int job=lane();job<QT*chunks;job+=lanes(B)) {
        int row=job/chunks, k=(job%chunks)*4096, t=Q0+row;
        int n=KT-k<4096 ? KT-k : 4096;
        int64_t limit=0;
        if(t<T && pos[t]>=0) {
            limit=(pos[t]+1)/R;
            if(limit>valid) limit=valid;
            if(limit>G) limit=G;
        }
        if(K0+k>=limit) {
            Duplicate(v.c,neginf(),int32_t(n)); sync();
        } else {
            Duplicate(v.c,0.f,int32_t(n)); sync();
            for(int h=0;h<H;++h) {
                fence<HardEvent::V_MTE2>();
                DataCopy(v.a,gd[(int64_t(row)*H+h)*KT+k],n);
                fence<HardEvent::MTE2_V>();
                Maxs(v.a,v.a,0.f,int32_t(n)); PipeBarrier<PIPE_V>();
                float head_weight=weight[int64_t(t)*H+h];
                Muls(v.a,v.a,head_weight,int32_t(n)); PipeBarrier<PIPE_V>();
                Add(v.c,v.c,v.a,int32_t(n)); PipeBarrier<PIPE_V>();
            }
            Muls(v.c,v.c,scale,int32_t(n)); sync();
            for(int j=int(limit-K0-k);j<n;++j)
                v.c.SetValue(j,neginf());
            sync();
        }
        GlobalTensor<float> go; go.SetGlobalBuffer((__gm__ float*)op);
        sync(); DataCopy(go[int64_t(row)*OUTKT+K0+k],v.c,n); sync();
    }
}

extern "C" void compact_reduce(void* st,void* d,void* w,void* p,void* v,void* o,
 int t,int h,int g,int r,int q0,int k0,int qt,int kt,int outkt,float scale,int b) {
 compact_reduce_kernel<<<b,nullptr,st>>>((uint8_t*)d,(uint8_t*)w,(uint8_t*)p,(uint8_t*)v,(uint8_t*)o,t,h,g,r,q0,k0,qt,kt,outkt,scale,b);
}
