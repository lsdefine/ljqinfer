// Build:
//   $ASCEND/tools/ccec_compiler/bin/bisheng -x cce --cce-aicore-arch=dav-c220 \
//     -O2 -std=c++17 -shared -fPIC -I$ASCEND/compiler/tikcpp/tikcfw{,/impl,/interface} \
//     -I$ASCEND/include ops/kernels/hc_residual.cpp -o ops/kernels/libhc.so
//
// Prefill hyper-connection residual kernels for 910B3 (dav-c220).
// bf16 storage, fp32 math -- same rounding points as the torch reference:
// every product is accumulated in fp32 and rounded to bf16 exactly once, at
// the store.  No runtime branch depends on data: C and D are released
// constants checked once in the launcher, and the only loops are over the
// static token/tile grid.
//
//   expand:   out[t,g,:] = post[t,g]*x[t,:] + sum_s comb[t,s,g]*res[t,s,:]
//             x [T,D] bf16, res [T,C,D] bf16, post [T,C] f32, comb [T,C,C] f32,
//             out [T,C,D] bf16
//   collapse: out[t,:]   = sum_s pre[t,s]*res[t,s,:]
//             res [T,C,D] bf16, pre [T,C] f32, out [T,D] bf16
//
// One (token, D-tile) pair per block iteration.  The C residual rows of a tile
// are fetched with a single strided DataCopy, so each byte of the residual
// stream crosses the bus exactly once.
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t DT = 2560;
constexpr uint32_t CMAX = 4;     // released hc_mult

extern "C" __global__ __aicore__ void pre_hc_expand_kernel(
    GM_ADDR x_gm, GM_ADDR res_gm, GM_ADDR post_gm, GM_ADDR comb_gm,
    GM_ADDR out_gm, uint32_t T, uint32_t C, uint32_t D) {
    constexpr uint32_t DT = 2560;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inputQ;
    TQue<TPosition::VECOUT, 2> outputQ;
    pipe.InitBuffer(inputQ, 1, (CMAX + 1) * DT * sizeof(bfloat16_t));
    pipe.InitBuffer(outputQ, 2, CMAX * DT * sizeof(bfloat16_t));
    TBuf<TPosition::VECCALC> coefficients;
    pipe.InitBuffer(coefficients, 96);
    auto coeff = coefficients.Get<float>();
    GlobalTensor<float> gp, gc;
    gp.SetGlobalBuffer((__gm__ float*)post_gm);
    gc.SetGlobalBuffer((__gm__ float*)comb_gm);
    TBuf<TPosition::VECCALC> bRf, bXf, bAcc, bProd;
    pipe.InitBuffer(bRf, CMAX * DT * sizeof(float));
    pipe.InitBuffer(bXf, DT * sizeof(float));
    pipe.InitBuffer(bProd, CMAX * DT * sizeof(float));
    pipe.InitBuffer(bAcc, 2 * DT * sizeof(float));
    LocalTensor<float> rf = bRf.Get<float>();
    LocalTensor<float> xf = bXf.Get<float>();
    LocalTensor<float> prod = bProd.Get<float>();
    LocalTensor<float> acc = bAcc.Get<float>();

    GlobalTensor<bfloat16_t> gX, gRes, gOut;
    gX.SetGlobalBuffer((__gm__ bfloat16_t*)x_gm);
    gRes.SetGlobalBuffer((__gm__ bfloat16_t*)res_gm);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t*)out_gm);
    __gm__ float* post = (__gm__ float*)post_gm;
    __gm__ float* comb = (__gm__ float*)comb_gm;

    const uint32_t nTile = D / DT;
    DataCopyParams cp;
    cp.blockCount = (uint16_t)C;
    cp.blockLen = (uint16_t)(DT * sizeof(bfloat16_t) / 32);
    cp.srcStride = (uint16_t)((D - DT) * sizeof(bfloat16_t) / 32);
    cp.dstStride = 0;
    DataCopyParams cpo = cp;
    cpo.srcStride = 0;
    cpo.dstStride = cp.srcStride;

    for (uint32_t job = GetBlockIdx(); job < T * nTile; job += GetBlockNum()*2) {
        uint32_t t = job / nTile;
        uint32_t d0 = (job % nTile) * DT;
        DataCopyPad(coeff, gp[uint64_t(t)*4], DataCopyExtParams{1,16,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        DataCopy(coeff[8], gc[uint64_t(t)*16], 16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float pv[CMAX], cv[CMAX * CMAX];
        for (uint32_t g = 0; g < C; ++g) {
            pv[g] = coeff.GetValue(g);
            for (uint32_t s = 0; s < C; ++s)
                cv[g * CMAX + s] = coeff.GetValue(8+s*C+g);
        }
        auto in = inputQ.AllocTensor<bfloat16_t>();
        auto rb = in;
        auto xb = in[CMAX * DT];
        DataCopy(xb, gX[(uint64_t)t * D + d0], DT);
        DataCopy(rb, gRes[(uint64_t)t * C * D + d0], cp);
        inputQ.EnQue(in);
        in = inputQ.DeQue<bfloat16_t>();
        rb = in; xb = in[CMAX * DT];
        auto ob = outputQ.AllocTensor<bfloat16_t>();
        Cast(xf, xb, RoundMode::CAST_NONE, DT);
        for (uint32_t s = 0; s < C; ++s)
            Cast(rf[s * DT], rb[s * DT], RoundMode::CAST_NONE, DT);
        PipeBarrier<PIPE_V>();
        inputQ.FreeTensor(in);
        // Preserve the exact product and sum order while overlapping DMA.
        for (uint32_t g0 = 0; g0 < C; g0 += 2) {
            for (uint32_t s = 0; s < 2; ++s)
                for (uint32_t g = 0; g < 2; ++g)
                    Muls(prod[(s*2+g)*DT], rf[s*DT], cv[(g0+g)*CMAX+s], int32_t(DT));
            PipeBarrier<PIPE_V>();
            Add(acc, prod, prod[2*DT], int32_t(2*DT));
            PipeBarrier<PIPE_V>();
            for (uint32_t s = 2; s < C; ++s)
                for (uint32_t g = 0; g < 2; ++g)
                    Muls(prod[((s-2)*2+g)*DT], rf[s*DT], cv[(g0+g)*CMAX+s], int32_t(DT));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, prod, int32_t(2*DT));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, prod[2*DT], int32_t(2*DT));
            PipeBarrier<PIPE_V>();
            for (uint32_t g = 0; g < 2; ++g)
                Muls(prod[g*DT], xf, pv[g0+g], int32_t(DT));
            PipeBarrier<PIPE_V>();
            Add(acc, prod, acc, int32_t(2*DT));
            PipeBarrier<PIPE_V>();
            Cast(ob[g0*DT], acc, RoundMode::CAST_RINT, 2*DT);
            PipeBarrier<PIPE_V>();
        }
        outputQ.EnQue(ob);
        ob = outputQ.DeQue<bfloat16_t>();
        DataCopy(gOut[(uint64_t)t * C * D + d0], ob, cpo);
        outputQ.FreeTensor(ob);
    }
}


extern "C" __global__ __aicore__ void pre_hc_collapse_kernel(
    GM_ADDR res_gm, GM_ADDR pre_gm, GM_ADDR out_gm,
    uint32_t T, uint32_t C, uint32_t D) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> bRb, bRf, bAcc, bOb, bTmp;
    pipe.InitBuffer(bRb, CMAX * DT * sizeof(bfloat16_t));
    pipe.InitBuffer(bRf, CMAX * DT * sizeof(float));
    pipe.InitBuffer(bTmp, DT * sizeof(float));
    pipe.InitBuffer(bAcc, DT * sizeof(float));
    pipe.InitBuffer(bOb, DT * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> rb = bRb.Get<bfloat16_t>();
    LocalTensor<float> rf = bRf.Get<float>();
    LocalTensor<float> tmp = bTmp.Get<float>();
    LocalTensor<float> acc = bAcc.Get<float>();
    LocalTensor<bfloat16_t> ob = bOb.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> gRes, gOut;
    gRes.SetGlobalBuffer((__gm__ bfloat16_t*)res_gm);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t*)out_gm);
    __gm__ float* pre = (__gm__ float*)pre_gm;

    const uint32_t nTile = D / DT;
    DataCopyParams cp;
    cp.blockCount = (uint16_t)C;
    cp.blockLen = (uint16_t)(DT * sizeof(bfloat16_t) / 32);
    cp.srcStride = (uint16_t)((D - DT) * sizeof(bfloat16_t) / 32);
    cp.dstStride = 0;

    for (uint32_t job = GetBlockIdx(); job < T * nTile; job += GetBlockNum()*2) {
        uint32_t t = job / nTile;
        uint32_t d0 = (job % nTile) * DT;
        float qv[CMAX];
        for (uint32_t s = 0; s < C; ++s) qv[s] = pre[(uint64_t)t * C + s];
        DataCopy(rb, gRes[(uint64_t)t * C * D + d0], cp);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t s = 0; s < C; ++s)
            Cast(rf[s * DT], rb[s * DT], RoundMode::CAST_NONE, DT);
        PipeBarrier<PIPE_V>();
        Muls(acc, rf, qv[0], DT);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 1; s < C; ++s) {
            Muls(tmp, rf[s * DT], qv[s], int32_t(DT));
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp, int32_t(DT));
            PipeBarrier<PIPE_V>();
        }
        Cast(ob, acc, RoundMode::CAST_RINT, DT);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(gOut[(uint64_t)t * D + d0], ob, DT);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

extern "C" int pre_hc_expand_launch(void* stream, void* x, void* res, void* post,
                                    void* comb, void* out, uint32_t T, uint32_t C,
                                    uint32_t D, uint32_t blockDim) {
    if (!stream || !x || !res || !post || !comb || !out) return -1;
    if (!T || C != CMAX || D % DT || !blockDim || blockDim > 48) return -1;
    pre_hc_expand_kernel<<<blockDim, nullptr, stream>>>(
        (uint8_t*)x, (uint8_t*)res, (uint8_t*)post, (uint8_t*)comb,
        (uint8_t*)out, T, C, D);
    return 0;
}

extern "C" int pre_hc_collapse_launch(void* stream, void* res, void* pre, void* out,
                                      uint32_t T, uint32_t C, uint32_t D,
                                      uint32_t blockDim) {
    if (!stream || !res || !pre || !out) return -1;
    if (!T || C != CMAX || D % DT || !blockDim || blockDim > 48) return -1;
    pre_hc_collapse_kernel<<<blockDim, nullptr, stream>>>(
        (uint8_t*)res, (uint8_t*)pre, (uint8_t*)out, T, C, D);
    return 0;
}

// sinkhorn: the remaining 19 trips of (row, column) normalisation, in place on
// a [C,C,T] mix.  Putting the token axis last turns every 4-wide reduction into
// a plain contiguous vector add; the 32 B block granularity of the repeat
// strides forbids that in the [T,C,C] layout.  The mix now crosses the bus once
// instead of 38 times, and each sum still adds eps last, as the reference does.
constexpr uint32_t SB = 1024;           // tokens held per tile
constexpr float HCEPS = 1e-6f;

extern "C" __global__ __aicore__ void pre_hc_sinkhorn_kernel(GM_ADDR comb_gm, uint32_t T, uint32_t iters, uint32_t bd) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> bV, bS;
    pipe.InitBuffer(bV, SB * CMAX * CMAX * sizeof(float));
    pipe.InitBuffer(bS, SB * sizeof(float));
    LocalTensor<float> v = bV.Get<float>();
    LocalTensor<float> s = bS.Get<float>();
    GlobalTensor<float> g;
    g.SetGlobalBuffer((__gm__ float*)comb_gm);

    constexpr uint32_t W = CMAX * CMAX;
    const uint32_t chunks = T / 8;              // 32 B units along the token axis
    const uint32_t nBlk = bd * 2;  // AIV: 2 vector cores per block
    const uint32_t id = GetBlockIdx();
    const uint32_t share = chunks / nBlk;
    const uint32_t extra = chunks % nBlk;
    const uint32_t begin = (share * id + (id < extra ? id : extra)) * 8;
    const uint32_t count = (share + (id < extra ? 1u : 0u)) * 8;

    for (uint32_t done = 0; done < count; done += SB) {
        uint32_t n = count - done < SB ? count - done : SB;
        uint32_t off = begin + done;
        for (uint32_t k = 0; k < W; ++k) DataCopy(v[k * SB], g[(uint64_t)k * T + off], (int32_t)n);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t it = 0; it < iters; ++it) {
            for (uint32_t i = 0; i < CMAX; ++i) {
                Add(s, v[(i * CMAX + 0) * SB], v[(i * CMAX + 1) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Add(s, s, v[(i * CMAX + 2) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Add(s, s, v[(i * CMAX + 3) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Adds(s, s, HCEPS, (int32_t)n);
                PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < CMAX; ++j) {
                    Div(v[(i * CMAX + j) * SB], v[(i * CMAX + j) * SB], s, (int32_t)n);
                }
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t j = 0; j < CMAX; ++j) {
                Add(s, v[(0 * CMAX + j) * SB], v[(1 * CMAX + j) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Add(s, s, v[(2 * CMAX + j) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Add(s, s, v[(3 * CMAX + j) * SB], (int32_t)n);
                PipeBarrier<PIPE_V>();
                Adds(s, s, HCEPS, (int32_t)n);
                PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < CMAX; ++i) {
                    Div(v[(i * CMAX + j) * SB], v[(i * CMAX + j) * SB], s, (int32_t)n);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t k = 0; k < W; ++k) DataCopy(g[(uint64_t)k * T + off], v[k * SB], (int32_t)n);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

extern "C" int pre_hc_sinkhorn_launch(void* stream, void* comb, uint32_t T, uint32_t C,
                                      uint32_t iters, uint32_t blockDim) {
    if (!stream || !comb) return -1;
    if (!T || T % 8 || C != CMAX || !blockDim || blockDim > 48) return -1;
    pre_hc_sinkhorn_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)comb, T, iters, blockDim);
    return 0;
}

// Routed SwiGLU: BF16 [rows, 2*288], FP32 probability [rows], BF16 out.
// Tile geometry and limit are fixed by the released TP8 build. No GM scratch.
constexpr uint32_t K = 288, R = 16, N = R*K;
extern "C" __global__ __aicore__ void routed_swiglu(
        GM_ADDR hidden, GM_ADDR probability, GM_ADDR output, uint32_t rows) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> bg, bu, bf, bv, bt, bp;
    pipe.InitBuffer(bg, N*2); pipe.InitBuffer(bu, N*2);
    pipe.InitBuffer(bf, N*4); pipe.InitBuffer(bv, N*4);
    pipe.InitBuffer(bt, N*4); pipe.InitBuffer(bp, R*4);
    auto gate = bg.Get<bfloat16_t>(); auto up = bu.Get<bfloat16_t>();
    auto g = bf.Get<float>(); auto u = bv.Get<float>();
    auto tmp = bt.Get<float>(); auto prob = bp.Get<float>();
    GlobalTensor<bfloat16_t> h, y;
    GlobalTensor<float> p;
    h.SetGlobalBuffer((__gm__ bfloat16_t*)hidden);
    p.SetGlobalBuffer((__gm__ float*)probability);
    y.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    // AIV has two subcores for each launched block on 910B.
    for (uint32_t tile = GetBlockIdx(); tile < (rows+R-1)/R; tile += GetBlockNum()*2) {
        const uint64_t row = uint64_t(tile)*R;
        const uint16_t active = min(uint32_t(R), rows-uint32_t(row));
        const int32_t count = active*K;
        DataCopyParams cp{active, K/16, K/16, 0};
        DataCopy(gate, h[row*2*K], cp);
        DataCopy(up, h[row*2*K+K], cp);
        DataCopyPad(prob, p[row], DataCopyExtParams{1, uint32_t(active)*4, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(g, gate, RoundMode::CAST_NONE, count);
        Cast(u, up, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Mins(g, g, 10.0f, count);
        Mins(u, u, 10.0f, count);
        PipeBarrier<PIPE_V>();
        Maxs(u, u, -10.0f, count);
        Muls(tmp, g, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, count);
        PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(g, g, tmp, count);
        PipeBarrier<PIPE_V>();
        Mul(g, g, u, count);
        PipeBarrier<PIPE_V>();
        for (uint32_t r=0; r<active; ++r)
            Muls(g[r*K], g[r*K], prob.GetValue(r), K);
        PipeBarrier<PIPE_V>();
        Cast(gate, g, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_ALL>();
        DataCopy(y[row*K], gate, count);
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" void routed_swiglu_launch(uint32_t blocks, void* stream,
        uint8_t* h, uint8_t* p, uint8_t* y, uint32_t rows) {
    routed_swiglu<<<blocks, nullptr, stream>>>(h, p, y, rows);
}

constexpr int32_t CD=5120, CK=6, CT=2560;
extern "C" __global__ __aicore__ void routed_combine(
        GM_ADDR values, GM_ADDR inverse, GM_ADDR output, uint32_t tokens) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> bi, bx, bf, ba;
    pipe.InitBuffer(bi, 64); pipe.InitBuffer(bx, CK*CT*2);
    pipe.InitBuffer(bf, CK*CT*4); pipe.InitBuffer(ba, CT*4);
    auto idx=bi.Get<int64_t>(); auto x=bx.Get<bfloat16_t>();
    auto f=bf.Get<float>(); auto acc=ba.Get<float>();
    GlobalTensor<int64_t> inv; GlobalTensor<bfloat16_t> y; GlobalTensor<float> out;
    inv.SetGlobalBuffer((__gm__ int64_t*)inverse);
    y.SetGlobalBuffer((__gm__ bfloat16_t*)values);
    out.SetGlobalBuffer((__gm__ float*)output);
    for (uint32_t tile=GetBlockIdx(); tile<tokens*2; tile+=GetBlockNum()*2) {
        const uint32_t token=tile/2, col=(tile%2)*CT;
        DataCopyPad(idx, inv[token*CK], DataCopyExtParams{1,CK*8,0,0,0},
                    DataCopyPadExtParams<int64_t>{false,0,0,0});
        PipeBarrier<PIPE_ALL>();
        int64_t rows[CK];
        for(int j=0;j<CK;++j) rows[j]=idx.GetValue(j);
        // Preserve the sorted-bank accumulation order without data-dependent branches.
        for(int end=CK-1;end>0;--end) for(int j=0;j<end;++j) {
            int64_t lo=min(rows[j],rows[j+1]), hi=max(rows[j],rows[j+1]);
            rows[j]=lo;rows[j+1]=hi;
        }
        for (int j=0;j<CK;++j)
            DataCopy(x[j*CT], y[uint64_t(rows[j])*CD+col], CT);
        PipeBarrier<PIPE_ALL>();
        Cast(f,x,RoundMode::CAST_NONE,CK*CT);
        PipeBarrier<PIPE_V>();
        Adds(acc,f,0.0f,CT);
        PipeBarrier<PIPE_V>();
        for (int j=1;j<CK;++j) {
            Add(acc,acc,f[j*CT],CT);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(out[uint64_t(token)*CD+col],acc,CT);
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" void routed_combine_launch(uint32_t blocks, void* stream,
        uint8_t* y, uint8_t* inverse, uint8_t* out, uint32_t tokens) {
    routed_combine<<<blocks,nullptr,stream>>>(y,inverse,out,tokens);
}

constexpr uint32_t RMS_D=5120;
extern "C" __global__ __aicore__ void pre_rms_kernel(GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t rows, float eps, uint32_t blocks) {
    if (GetBlockIdx() >= blocks) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bh,bx,bw,bs,bt,br;
    pipe.InitBuffer(bh,RMS_D*2); pipe.InitBuffer(bx,RMS_D*4);
    pipe.InitBuffer(bw,RMS_D*4); pipe.InitBuffer(bs,RMS_D*4);
    pipe.InitBuffer(bt,RMS_D*4); pipe.InitBuffer(br,32*4);
    auto h=bh.Get<bfloat16_t>(); auto x=bx.Get<float>();
    auto w=bw.Get<float>(); auto s=bs.Get<float>();
    auto tmp=bt.Get<float>(); auto red=br.Get<float>();
    GlobalTensor<bfloat16_t> gx,gy; GlobalTensor<float> gw;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)input,uint64_t(rows)*RMS_D);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)output,uint64_t(rows)*RMS_D);
    gw.SetGlobalBuffer((__gm__ float*)weight,RMS_D);
    DataCopy(w,gw,RMS_D); PipeBarrier<PIPE_ALL>();
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks) {
        DataCopy(h,gx[uint64_t(row)*RMS_D],RMS_D); PipeBarrier<PIPE_ALL>();
        Cast(x,h,RoundMode::CAST_NONE,RMS_D); PipeBarrier<PIPE_V>();
        Mul(s,x,x,int32_t(RMS_D)); PipeBarrier<PIPE_V>();
        ReduceSum(red,s,tmp,int32_t(RMS_D)); PipeBarrier<PIPE_ALL>();
        float sum=red.GetValue(0);
        Duplicate(red,sum,32); PipeBarrier<PIPE_V>();
        Muls(red,red,1.0f/5120.0f,32); PipeBarrier<PIPE_V>();
        Adds(red,red,eps,32); PipeBarrier<PIPE_V>();
        Sqrt(red,red,32); PipeBarrier<PIPE_V>();
        Duplicate(tmp,1.0f,32); PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32)); PipeBarrier<PIPE_ALL>();
        float inv=red.GetValue(0);
        Muls(x,x,inv,RMS_D); PipeBarrier<PIPE_V>();
        Mul(x,x,w,int32_t(RMS_D)); PipeBarrier<PIPE_V>();
        Cast(h,x,RoundMode::CAST_RINT,RMS_D); PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(row)*RMS_D],h,RMS_D); PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int pre_rms_launch(void* stream,void* x,void* w,void* out,uint32_t rows,float eps) {
    pre_rms_kernel<<<40,nullptr,stream>>>((uint8_t*)x,(uint8_t*)w,(uint8_t*)out,rows,eps,40);
    return 0;
}

// Collapse + RMS keeps the mandatory intermediate BF16 rounding in UB.
// Exactly the two original reduction orders, no GM scratch or allocations.
extern "C" __global__ __aicore__ void pre_hc_norm_kernel(
    GM_ADDR residual, GM_ADDR pre, GM_ADDR weight, GM_ADDR output,
    uint32_t rows, float eps) {
    constexpr int32_t D = 5120;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bh,bf,ba,bt,bw,br,bp;
    pipe.InitBuffer(bh,4*D*2); pipe.InitBuffer(bf,4*D*4);
    pipe.InitBuffer(ba,D*4); pipe.InitBuffer(bt,D*4);
    pipe.InitBuffer(bw,D*4); pipe.InitBuffer(br,128); pipe.InitBuffer(bp,32);
    auto h=bh.Get<bfloat16_t>(); auto f=bf.Get<float>();
    auto acc=ba.Get<float>(); auto tmp=bt.Get<float>();
    auto w=bw.Get<float>(); auto red=br.Get<float>(); auto coeff=bp.Get<float>();
    GlobalTensor<bfloat16_t> gx,gy; GlobalTensor<float> gp,gw;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)residual);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    gp.SetGlobalBuffer((__gm__ float*)pre); gw.SetGlobalBuffer((__gm__ float*)weight);
    DataCopy(w,gw,D);
    for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()*2) {
        DataCopy(h,gx[uint64_t(row)*4*D],4*D);
        DataCopyPad(coeff,gp[uint64_t(row)*4],DataCopyExtParams{1,16,0,0,0},
                    DataCopyPadExtParams<float>{false,0,0,0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float q[4];for(int i=0;i<4;++i)q[i]=coeff.GetValue(i);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(f,h,RoundMode::CAST_NONE,4*D);PipeBarrier<PIPE_V>();
        Muls(acc,f,q[0],D);PipeBarrier<PIPE_V>();
        for(int i=1;i<4;++i) {
            Muls(tmp,f[i*D],q[i],D);PipeBarrier<PIPE_V>();
            Add(acc,acc,tmp,D);PipeBarrier<PIPE_V>();
        }
        Cast(h,acc,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_V>();
        Cast(acc,h,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();
        Mul(f,acc,acc,D);PipeBarrier<PIPE_V>();
        ReduceSum(red,f,tmp,D);PipeBarrier<PIPE_ALL>();
        float sum=red.GetValue(0);
        Duplicate(red,sum,32);PipeBarrier<PIPE_V>();
        Muls(red,red,1.0f/5120.0f,32);PipeBarrier<PIPE_V>();
        Adds(red,red,eps,32);PipeBarrier<PIPE_V>();
        Sqrt(red,red,32);PipeBarrier<PIPE_V>();
        Duplicate(tmp,1.0f,32);PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32));PipeBarrier<PIPE_ALL>();
        float inv=red.GetValue(0);
        Muls(acc,acc,inv,D);PipeBarrier<PIPE_V>();
        Mul(acc,acc,w,D);PipeBarrier<PIPE_V>();
        Cast(h,acc,RoundMode::CAST_RINT,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gy[uint64_t(row)*D],h,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int pre_hc_norm_launch(void* stream,void* x,void* pre,void* w,void* out,
                                 uint32_t rows,float eps) {
    pre_hc_norm_kernel<<<24,nullptr,stream>>>((uint8_t*)x,(uint8_t*)pre,
        (uint8_t*)w,(uint8_t*)out,rows,eps);
    return 0;
}

constexpr uint32_t HC_STATS_TILE=5120;
extern "C" __global__ __aicore__ void pre_hc_cast_stats_kernel(
    GM_ADDR input, GM_ADDR h_out, GM_ADDR stats_out, uint32_t rows, float eps, uint32_t blocks) {
    if (GetBlockIdx() >= blocks) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bh,bx,bs,bt,br;
    pipe.InitBuffer(bh,HC_STATS_TILE*2); pipe.InitBuffer(bx,HC_STATS_TILE*4);
    pipe.InitBuffer(bs,HC_STATS_TILE*4); pipe.InitBuffer(bt,HC_STATS_TILE*4); pipe.InitBuffer(br,32*4);
    auto h=bh.Get<bfloat16_t>(); auto x=bx.Get<float>();
    auto s=bs.Get<float>(); auto tmp=bt.Get<float>(); auto red=br.Get<float>();
    GlobalTensor<bfloat16_t> gx; GlobalTensor<float> gh,gs;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)input,uint64_t(rows)*20480);
    gh.SetGlobalBuffer((__gm__ float*)h_out,uint64_t(rows)*20480);
    gs.SetGlobalBuffer((__gm__ float*)stats_out,rows);
    for(uint32_t row=GetBlockIdx();row<rows;row+=blocks) {
        float total=0.0f;
        for(uint32_t tile=0;tile<4;++tile) {
            uint64_t offset=uint64_t(row)*20480+tile*HC_STATS_TILE;
            DataCopy(h,gx[offset],HC_STATS_TILE);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(x,h,RoundMode::CAST_NONE,HC_STATS_TILE);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(gh[offset],x,HC_STATS_TILE);
            PipeBarrier<PIPE_V>();
            Mul(s,x,x,int32_t(HC_STATS_TILE)); PipeBarrier<PIPE_V>();
            ReduceSum(red,s,tmp,int32_t(HC_STATS_TILE)); PipeBarrier<PIPE_ALL>();
            total+=red.GetValue(0);
        }
        Duplicate(red,total,32); PipeBarrier<PIPE_V>();
        Muls(red,red,1.0f/20480.0f,32); PipeBarrier<PIPE_V>();
        Adds(red,red,eps,32); PipeBarrier<PIPE_V>();
        Sqrt(red,red,32); PipeBarrier<PIPE_V>();
        Duplicate(tmp,1.0f,32); PipeBarrier<PIPE_V>();
        Div(red,tmp,red,int32_t(32)); PipeBarrier<PIPE_ALL>();
        DataCopyPad(gs[row],red,DataCopyExtParams{1,4,0,0,0});
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int pre_hc_cast_stats_launch(void* stream,void* x,void* h,void* stats,uint32_t rows,float eps) {
    pre_hc_cast_stats_kernel<<<40,nullptr,stream>>>((uint8_t*)x,(uint8_t*)h,(uint8_t*)stats,rows,eps,40);
    return 0;
}

// Preserve shared BF16 rounding before FP32 routed sum.
constexpr uint32_t MOE_FINISH_D=5120;
extern "C" __global__ __aicore__ void pre_moe_finish_kernel(GM_ADDR routed,GM_ADDR shared,GM_ADDR output,uint32_t rows) {
 if(GetBlockIdx()>=40)return;
 TPipe pipe;TBuf<TPosition::VECCALC> ba,bb,bh;
 pipe.InitBuffer(ba,MOE_FINISH_D*4);pipe.InitBuffer(bb,MOE_FINISH_D*4);pipe.InitBuffer(bh,MOE_FINISH_D*2);
 auto a=ba.Get<float>();auto b=bb.Get<float>();auto h=bh.Get<bfloat16_t>();
 GlobalTensor<float> ga,gb;GlobalTensor<bfloat16_t> go;
 ga.SetGlobalBuffer((__gm__ float*)routed,uint64_t(rows)*MOE_FINISH_D);
 gb.SetGlobalBuffer((__gm__ float*)shared,uint64_t(rows)*MOE_FINISH_D);
 go.SetGlobalBuffer((__gm__ bfloat16_t*)output,uint64_t(rows)*MOE_FINISH_D);
 for(uint32_t row=GetBlockIdx();row<rows;row+=40){
  uint64_t off=uint64_t(row)*MOE_FINISH_D;
  DataCopy(a,ga[off],MOE_FINISH_D);DataCopy(b,gb[off],MOE_FINISH_D);PipeBarrier<PIPE_ALL>();
  Cast(h,b,RoundMode::CAST_RINT,MOE_FINISH_D);PipeBarrier<PIPE_ALL>();
  Cast(b,h,RoundMode::CAST_NONE,MOE_FINISH_D);PipeBarrier<PIPE_ALL>();
  Add(a,a,b,int32_t(MOE_FINISH_D));PipeBarrier<PIPE_ALL>();
  Cast(h,a,RoundMode::CAST_RINT,MOE_FINISH_D);PipeBarrier<PIPE_ALL>();
  DataCopy(go[off],h,MOE_FINISH_D);PipeBarrier<PIPE_ALL>();
 }
}
extern "C" int pre_moe_finish_launch(void* stream,void* routed,void* shared,void* output,uint32_t rows){
 pre_moe_finish_kernel<<<40,nullptr,stream>>>((uint8_t*)routed,(uint8_t*)shared,(uint8_t*)output,rows);return 0;
}

// Engram elementwise stages only. ACL rotation, Mean/ReduceSum and scalar
// gate remain unchanged, including the two separate FP32 dot products.
template<bool F32> __aicore__ inline void engram_products(GM_ADDR rot, GM_ADDR kv,
        GM_ADDR weight, GM_ADDR hs, GM_ADDR ks, uint32_t rows) {
    if(GetBlockIdx()>=48)return;
    constexpr int32_t D=5120;
    TPipe p; TBuf<TPosition::VECCALC> ba,bk,bw,bs,bh;
    p.InitBuffer(ba,D*4);p.InitBuffer(bk,D*4);p.InitBuffer(bw,D*4);
    p.InitBuffer(bs,D*4);p.InitBuffer(bh,D*2);
    auto a=ba.Get<float>();auto k=bk.Get<float>();auto w=bw.Get<float>();
    auto s=bs.Get<float>();auto b=bh.Get<bfloat16_t>();
    GlobalTensor<float> gr,gw,gh,gks,gkf;
    GlobalTensor<bfloat16_t> gkb;
    gr.SetGlobalBuffer((__gm__ float*)rot);gw.SetGlobalBuffer((__gm__ float*)weight);
    gh.SetGlobalBuffer((__gm__ float*)hs);gks.SetGlobalBuffer((__gm__ float*)ks);
    gkf.SetGlobalBuffer((__gm__ float*)kv);gkb.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
    for(uint32_t row=GetBlockIdx();row<rows*4;row+=48){
        uint64_t off=uint64_t(row)*D, ko=uint64_t(row/4)*5*D+(row%4)*D;
        DataCopy(a,gr[off],D);DataCopy(w,gw[(row%4)*D],D);
        if constexpr(F32)DataCopy(k,gkf[ko],D);else DataCopy(b,gkb[ko],D);
        PipeBarrier<PIPE_ALL>();
        if constexpr(!F32){Cast(k,b,RoundMode::CAST_NONE,D);PipeBarrier<PIPE_V>();}
        Mul(s,a,a,D);PipeBarrier<PIPE_ALL>();DataCopy(gh[off],s,D);PipeBarrier<PIPE_ALL>();
        Mul(s,k,k,D);PipeBarrier<PIPE_ALL>();DataCopy(gks[off],s,D);PipeBarrier<PIPE_ALL>();
        Mul(a,a,w,D);PipeBarrier<PIPE_V>();Mul(a,a,k,D);PipeBarrier<PIPE_ALL>();
        DataCopy(gr[off],a,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void pre_engram_products_bf16(GM_ADDR rot,GM_ADDR kv,GM_ADDR w,GM_ADDR hs,GM_ADDR ks,uint32_t rows){engram_products<false>(rot,kv,w,hs,ks,rows);}
extern "C" __global__ __aicore__ void pre_engram_products_f32(GM_ADDR rot,GM_ADDR kv,GM_ADDR w,GM_ADDR hs,GM_ADDR ks,uint32_t rows){engram_products<true>(rot,kv,w,hs,ks,rows);}
extern "C" int pre_engram_products_launch(void* stream,void* rot,void* kv,void* w,void* hs,void* ks,uint32_t rows,uint32_t f32){
    if(f32)pre_engram_products_f32<<<24,nullptr,stream>>>((uint8_t*)rot,(uint8_t*)kv,(uint8_t*)w,(uint8_t*)hs,(uint8_t*)ks,rows);
    else pre_engram_products_bf16<<<24,nullptr,stream>>>((uint8_t*)rot,(uint8_t*)kv,(uint8_t*)w,(uint8_t*)hs,(uint8_t*)ks,rows);
    return 0;
}
template<bool F32> __aicore__ inline void engram_finish(GM_ADDR x,GM_ADDR kv,GM_ADDR gate,GM_ADDR out,uint32_t rows){
    if(GetBlockIdx()>=48)return;
    constexpr int32_t D=5120;
    TPipe p;TBuf<TPosition::VECCALC> bx,bv,ba,bh;
    p.InitBuffer(bx,D*2);p.InitBuffer(bv,D*4);p.InitBuffer(ba,D*4);p.InitBuffer(bh,D*2);
    auto xb=bx.Get<bfloat16_t>();auto vb=bh.Get<bfloat16_t>();
    auto a=ba.Get<float>();auto v=bv.Get<float>();
    GlobalTensor<bfloat16_t> gx,gkv,go;GlobalTensor<float> gkf;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)x);gkv.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)out);gkf.SetGlobalBuffer((__gm__ float*)kv);
    for(uint32_t row=GetBlockIdx();row<rows*4;row+=48){
        uint64_t off=uint64_t(row)*D,vo=uint64_t(row/4)*5*D+4*D;
        float g=((__gm__ float*)gate)[row];
        DataCopy(xb,gx[off],D);
        if constexpr(F32)DataCopy(v,gkf[vo],D);else DataCopy(vb,gkv[vo],D);
        PipeBarrier<PIPE_ALL>();Cast(a,xb,RoundMode::CAST_NONE,D);
        if constexpr(!F32)Cast(v,vb,RoundMode::CAST_NONE,D);
        PipeBarrier<PIPE_V>();Muls(v,v,g,D);PipeBarrier<PIPE_V>();
        Add(a,a,v,D);PipeBarrier<PIPE_V>();Cast(xb,a,RoundMode::CAST_RINT,D);
        PipeBarrier<PIPE_ALL>();DataCopy(go[off],xb,D);PipeBarrier<PIPE_ALL>();
    }
}
extern "C" __global__ __aicore__ void pre_engram_finish_bf16(GM_ADDR x,GM_ADDR kv,GM_ADDR gate,GM_ADDR out,uint32_t rows){engram_finish<false>(x,kv,gate,out,rows);}
extern "C" __global__ __aicore__ void pre_engram_finish_f32(GM_ADDR x,GM_ADDR kv,GM_ADDR gate,GM_ADDR out,uint32_t rows){engram_finish<true>(x,kv,gate,out,rows);}
extern "C" int pre_engram_finish_launch(void* stream,void* x,void* kv,void* gate,void* out,uint32_t rows,uint32_t f32){
    if(f32)pre_engram_finish_f32<<<24,nullptr,stream>>>((uint8_t*)x,(uint8_t*)kv,(uint8_t*)gate,(uint8_t*)out,rows);
    else pre_engram_finish_bf16<<<24,nullptr,stream>>>((uint8_t*)x,(uint8_t*)kv,(uint8_t*)gate,(uint8_t*)out,rows);
    return 0;
}

// Shared SwiGLU: BF16 projection inputs, FP32 activation/product, one BF16 store.
extern "C" __global__ __aicore__ void shared_swiglu_kernel(
        GM_ADDR gate, GM_ADDR up, GM_ADDR out, uint32_t elements, float limit) {
    constexpr int32_t N = 4096;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bb, bg, bu, bt;
    pipe.InitBuffer(bb, N*2); pipe.InitBuffer(bg, N*4);
    pipe.InitBuffer(bu, N*4); pipe.InitBuffer(bt, N*4);
    auto b=bb.Get<bfloat16_t>(); auto g=bg.Get<float>();
    auto u=bu.Get<float>(); auto t=bt.Get<float>();
    GlobalTensor<bfloat16_t> gg, gu, go;
    gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);
    gu.SetGlobalBuffer((__gm__ bfloat16_t*)up);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    for (uint32_t tile=GetBlockIdx(); tile<(elements+N-1)/N; tile+=GetBlockNum()*2) {
        uint32_t off=tile*N;
        int32_t count=min(uint32_t(N), elements-off);
        DataCopy(b, gg[off], count); PipeBarrier<PIPE_ALL>();
        Cast(g,b,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_ALL>();
        DataCopy(b, gu[off], count); PipeBarrier<PIPE_ALL>();
        Cast(u,b,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
        if(limit>0) {
            Mins(g,g,limit,count); Mins(u,u,limit,count); PipeBarrier<PIPE_V>();
            Maxs(u,u,-limit,count); PipeBarrier<PIPE_V>();
        }
        Muls(t,g,-1.0f,count); PipeBarrier<PIPE_V>();
        Exp(t,t,count); PipeBarrier<PIPE_V>();
        Adds(t,t,1.0f,count); PipeBarrier<PIPE_V>();
        Div(g,g,t,count); PipeBarrier<PIPE_V>();
        Mul(g,g,u,count); PipeBarrier<PIPE_V>();
        Cast(b,g,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_ALL>();
        DataCopy(go[off],b,count); PipeBarrier<PIPE_ALL>();
    }
}
extern "C" int shared_swiglu_launch(void* stream, void* gate, void* up, void* out,
                                    uint32_t elements, float limit) {
    if (!elements || elements%16) return -1;
    shared_swiglu_kernel<<<24,nullptr,stream>>>((uint8_t*)gate,(uint8_t*)up,(uint8_t*)out,elements,limit);
    return 0;
}
