#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t T = 8;
constexpr uint32_t NK = 4;
constexpr uint32_t NV = 12;
constexpr uint32_t D = 128;
constexpr uint32_t NSTATE = D * D;
constexpr uint32_t VSTEP = 64;
constexpr uint32_t NSTILE = VSTEP * D;
constexpr uint32_t FP32_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LEN = 64;
constexpr uint32_t MAX_REPEAT = 255;

template<class TLocal>
__aicore__ inline void MatVecMul128(const TLocal &cube, const TLocal &vec,
                                    TLocal &dst, bool add) {
    constexpr uint8_t stride = D / FP32_PER_BLOCK;
    for (uint32_t i = 0; i < D; i += REPEAT_LEN) {
        for (uint32_t j = 0; j < VSTEP; j += MAX_REPEAT) {
            uint64_t reps = VSTEP - j;
            if (reps > MAX_REPEAT) reps = MAX_REPEAT;
            if (add) {
                MulAddDst(dst[j * D + i], cube[j * D + i], vec[i],
                          REPEAT_LEN, reps, {1, 1, 1, stride, stride, 0});
            } else {
                Mul(dst[j * D + i], cube[j * D + i], vec[i],
                    REPEAT_LEN, reps, {1, 1, 1, stride, stride, 0});
            }
        }
    }
}

extern "C" __global__ __aicore__ void gdn_recurrent_b1q8(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR state, GM_ADDR out, GM_ADDR stateIndices, GM_ADDR numAccepted,
    uint32_t vTokenStride) {
    const uint32_t block = GetBlockIdx();
    if (block >= NV * 2) return;
    const uint32_t vh = block >> 1;
    const uint32_t qh = vh / (NV / NK);
    const uint32_t voff = (block & 1) * VSTEP;

    GlobalTensor<bfloat16_t> qgm, kgm, vgm, betagm, stgm, outgm;
    GlobalTensor<float> ggm;
    GlobalTensor<int32_t> idxgm, acceptedgm;
    qgm.SetGlobalBuffer((__gm__ bfloat16_t*)q, T * NK * D);
    kgm.SetGlobalBuffer((__gm__ bfloat16_t*)k, T * NK * D);
    vgm.SetGlobalBuffer((__gm__ bfloat16_t*)v, T * NV * D);
    ggm.SetGlobalBuffer((__gm__ float*)g, T * NV);
    betagm.SetGlobalBuffer((__gm__ bfloat16_t*)beta, T * NV);
    stgm.SetGlobalBuffer((__gm__ bfloat16_t*)state, T * NV * NSTATE);
    outgm.SetGlobalBuffer((__gm__ bfloat16_t*)out, T * NV * D);
    idxgm.SetGlobalBuffer((__gm__ int32_t*)stateIndices, T);
    acceptedgm.SetGlobalBuffer((__gm__ int32_t*)numAccepted, 1);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bqbf, bkbf, bvbf, bbetabf, bstbf;
    TBuf<TPosition::VECCALC> bqf, bkf, bvf, bgf, bbetaf;
    TBuf<TPosition::VECCALC> bstf, btmp, bdelta, battn, boutbf;
    pipe.InitBuffer(bqbf, T * D * sizeof(bfloat16_t));
    pipe.InitBuffer(bkbf, T * D * sizeof(bfloat16_t));
    pipe.InitBuffer(bvbf, T * VSTEP * sizeof(bfloat16_t));
    pipe.InitBuffer(bbetabf, T * NV * sizeof(bfloat16_t));
    pipe.InitBuffer(bstbf, 2 * NSTILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bqf, T * D * sizeof(float));
    pipe.InitBuffer(bkf, T * D * sizeof(float));
    pipe.InitBuffer(bvf, T * VSTEP * sizeof(float));
    pipe.InitBuffer(bgf, T * NV * sizeof(float));
    pipe.InitBuffer(bbetaf, T * NV * sizeof(float));
    pipe.InitBuffer(bstf, NSTILE * sizeof(float));
    pipe.InitBuffer(btmp, NSTILE * sizeof(float));
    pipe.InitBuffer(bdelta, VSTEP * sizeof(float));
    pipe.InitBuffer(battn, VSTEP * sizeof(float));
    pipe.InitBuffer(boutbf, 2 * VSTEP * sizeof(bfloat16_t));

    auto qbf = bqbf.Get<bfloat16_t>();
    auto kbf = bkbf.Get<bfloat16_t>();
    auto vbf = bvbf.Get<bfloat16_t>();
    auto betabf = bbetabf.Get<bfloat16_t>();
    auto stbf = bstbf.Get<bfloat16_t>();
    auto qf = bqf.Get<float>();
    auto kf = bkf.Get<float>();
    auto vf = bvf.Get<float>();
    auto gf = bgf.Get<float>();
    auto betaf = bbetaf.Get<float>();
    auto stf = bstf.Get<float>();
    auto tmp = btmp.Get<float>();
    auto delta = bdelta.Get<float>();
    auto attn = battn.Get<float>();
    auto outbf = boutbf.Get<bfloat16_t>();

    for (uint32_t t = 0; t < T; ++t) {
        DataCopy(qbf[t * D], qgm[(t * NK + qh) * D], D);
        DataCopy(kbf[t * D], kgm[(t * NK + qh) * D], D);
        DataCopy(vbf[t * VSTEP],
                 vgm[t * vTokenStride + vh * D + voff], VSTEP);
    }
    DataCopy(gf, ggm, T * NV);
    DataCopy(betabf, betagm, T * NV);
    PipeBarrier<PIPE_ALL>();
    Cast(qf, qbf, RoundMode::CAST_NONE, T * D);
    Cast(kf, kbf, RoundMode::CAST_NONE, T * D);
    Cast(vf, vbf, RoundMode::CAST_NONE, T * VSTEP);
    Cast(betaf, betabf, RoundMode::CAST_NONE, T * NV);
    Duplicate(tmp, 1.0f, T * NV);
    PipeBarrier<PIPE_V>();
    Muls(betaf, betaf, -1.0f, T * NV);
    PipeBarrier<PIPE_V>();
    Exp(betaf, betaf, T * NV);
    PipeBarrier<PIPE_V>();
    Adds(betaf, betaf, 1.0f, T * NV);
    PipeBarrier<PIPE_V>();
    Div(betaf, tmp, betaf, T * NV);
    PipeBarrier<PIPE_V>();
    Cast(outbf, betaf, RoundMode::CAST_RINT, T * NV);
    PipeBarrier<PIPE_V>();
    Cast(betaf, outbf, RoundMode::CAST_NONE, T * NV);
    PipeBarrier<PIPE_V>();
    Muls(qf, qf, 0.08838834764831845f, T * D);
    Exp(gf, gf, T * NV);
    PipeBarrier<PIPE_V>();

    uint32_t stateShape[2] = {VSTEP, D};
    uint32_t deltaShape[2] = {VSTEP, 1};
    const uint32_t accepted = static_cast<uint32_t>(acceptedgm.GetValue(0));
    const uint32_t base = static_cast<uint32_t>(idxgm.GetValue(accepted - 1));
    DataCopy(stbf, stgm[(base * NV + vh) * NSTATE + voff * D], NSTILE);
    PipeBarrier<PIPE_ALL>();
    Cast(stf, stbf, RoundMode::CAST_NONE, NSTILE);
    PipeBarrier<PIPE_V>();
    for (uint32_t t = 0; t < T; ++t) {
            const float gamma = gf.GetValue(t * NV + vh);
            const float b = betaf.GetValue(t * NV + vh);
            Muls(stf, stf, gamma, NSTILE);
            PipeBarrier<PIPE_V>();
            MatVecMul128(stf, kf[t * D], tmp, false);
            PipeBarrier<PIPE_V>();
            ReduceSum<float, Pattern::Reduce::AR, true>(delta, tmp, stateShape, true);
            PipeBarrier<PIPE_V>();
            Sub(delta, vf[t * VSTEP], delta, VSTEP);
            PipeBarrier<PIPE_V>();
            Muls(delta, delta, b, VSTEP);
            PipeBarrier<PIPE_V>();
            Broadcast<float, 2, 1>(tmp, delta, stateShape, deltaShape);
            PipeBarrier<PIPE_V>();
            MatVecMul128(tmp, kf[t * D], stf, true);
            PipeBarrier<PIPE_V>();
            MatVecMul128(stf, qf[t * D], tmp, false);
            PipeBarrier<PIPE_V>();
            ReduceSum<float, Pattern::Reduce::AR, true>(attn, tmp, stateShape, true);
            PipeBarrier<PIPE_V>();
            const uint32_t lane = t & 1;
            auto stbfOut = stbf[lane * NSTILE];
            auto outbfOut = outbf[lane * VSTEP];
            if (t >= 2) {
                if (lane == 0) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                else WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            Cast(stbfOut, stf, RoundMode::CAST_RINT, NSTILE);
            Cast(outbfOut, attn, RoundMode::CAST_RINT, VSTEP);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t slot = static_cast<uint32_t>(idxgm.GetValue(t));
            DataCopy(stgm[(slot * NV + vh) * NSTATE + voff * D], stbfOut, NSTILE);
            DataCopy(outgm[(t * NV + vh) * D + voff], outbfOut, VSTEP);
            if (lane == 0) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            else SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

extern "C" __global__ __aicore__ void gdn_recurrent_batch_b1q8(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR baseState, GM_ADDR state, GM_ADDR out,
    GM_ADDR stateIndices, GM_ADDR numAccepted,
    uint32_t vTokenStride, uint32_t batch) {
    const uint32_t block = GetBlockIdx();
    if (block >= NV * 2) return;
    const uint32_t vh = block >> 1;
    const uint32_t qh = vh / (NV / NK);
    const uint32_t voff = (block & 1) * VSTEP;

    GlobalTensor<bfloat16_t> qgm, kgm, vgm, betagm, basegm, stgm, outgm;
    GlobalTensor<float> ggm;
    GlobalTensor<int32_t> idxgm, acceptedgm;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bqbf, bkbf, bvbf, bbetabf, bstbf;
    TBuf<TPosition::VECCALC> bqf, bkf, bvf, bgf, bbetaf;
    TBuf<TPosition::VECCALC> bstf, btmp, bdelta, battn, boutbf;
    pipe.InitBuffer(bqbf, T * D * sizeof(bfloat16_t));
    pipe.InitBuffer(bkbf, T * D * sizeof(bfloat16_t));
    pipe.InitBuffer(bvbf, T * VSTEP * sizeof(bfloat16_t));
    pipe.InitBuffer(bbetabf, T * NV * sizeof(bfloat16_t));
    pipe.InitBuffer(bstbf, 2 * NSTILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bqf, T * D * sizeof(float));
    pipe.InitBuffer(bkf, T * D * sizeof(float));
    pipe.InitBuffer(bvf, T * VSTEP * sizeof(float));
    pipe.InitBuffer(bgf, T * NV * sizeof(float));
    pipe.InitBuffer(bbetaf, T * NV * sizeof(float));
    pipe.InitBuffer(bstf, NSTILE * sizeof(float));
    pipe.InitBuffer(btmp, NSTILE * sizeof(float));
    pipe.InitBuffer(bdelta, VSTEP * sizeof(float));
    pipe.InitBuffer(battn, VSTEP * sizeof(float));
    pipe.InitBuffer(boutbf, 2 * VSTEP * sizeof(bfloat16_t));

    auto qbf = bqbf.Get<bfloat16_t>();
    auto kbf = bkbf.Get<bfloat16_t>();
    auto vbf = bvbf.Get<bfloat16_t>();
    auto betabf = bbetabf.Get<bfloat16_t>();
    auto stbf = bstbf.Get<bfloat16_t>();
    auto qf = bqf.Get<float>();
    auto kf = bkf.Get<float>();
    auto vf = bvf.Get<float>();
    auto gf = bgf.Get<float>();
    auto betaf = bbetaf.Get<float>();
    auto stf = bstf.Get<float>();
    auto tmp = btmp.Get<float>();
    auto delta = bdelta.Get<float>();
    auto attn = battn.Get<float>();
    auto outbf = boutbf.Get<bfloat16_t>();

    for (uint32_t row = 0; row < batch; ++row) {
        const uint64_t tokenBase = static_cast<uint64_t>(row) * T;
        qgm.SetGlobalBuffer((__gm__ bfloat16_t*)q + tokenBase * NK * D, T * NK * D);
        kgm.SetGlobalBuffer((__gm__ bfloat16_t*)k + tokenBase * NK * D, T * NK * D);
        vgm.SetGlobalBuffer((__gm__ bfloat16_t*)v + tokenBase * vTokenStride, T * NV * D);
        ggm.SetGlobalBuffer((__gm__ float*)g + tokenBase * NV, T * NV);
        betagm.SetGlobalBuffer((__gm__ bfloat16_t*)beta + tokenBase * NV, T * NV);
        basegm.SetGlobalBuffer((__gm__ bfloat16_t*)baseState +
                               static_cast<uint64_t>(row) * NV * NSTATE,
                               NV * NSTATE);
        stgm.SetGlobalBuffer((__gm__ bfloat16_t*)state + tokenBase * NV * NSTATE, T * NV * NSTATE);
        outgm.SetGlobalBuffer((__gm__ bfloat16_t*)out + tokenBase * NV * D, T * NV * D);
        idxgm.SetGlobalBuffer((__gm__ int32_t*)stateIndices + tokenBase, T);
        acceptedgm.SetGlobalBuffer((__gm__ int32_t*)numAccepted + row, 1);
        for (uint32_t t = 0; t < T; ++t) {
            DataCopy(qbf[t * D], qgm[(t * NK + qh) * D], D);
            DataCopy(kbf[t * D], kgm[(t * NK + qh) * D], D);
            DataCopy(vbf[t * VSTEP],
                     vgm[t * vTokenStride + vh * D + voff], VSTEP);
        }
        DataCopy(gf, ggm, T * NV);
        DataCopy(betabf, betagm, T * NV);
        PipeBarrier<PIPE_ALL>();
        Cast(qf, qbf, RoundMode::CAST_NONE, T * D);
        Cast(kf, kbf, RoundMode::CAST_NONE, T * D);
        Cast(vf, vbf, RoundMode::CAST_NONE, T * VSTEP);
        Cast(betaf, betabf, RoundMode::CAST_NONE, T * NV);
        Duplicate(tmp, 1.0f, T * NV);
        PipeBarrier<PIPE_V>();
        Muls(betaf, betaf, -1.0f, T * NV);
        PipeBarrier<PIPE_V>();
        Exp(betaf, betaf, T * NV);
        PipeBarrier<PIPE_V>();
        Adds(betaf, betaf, 1.0f, T * NV);
        PipeBarrier<PIPE_V>();
        Div(betaf, tmp, betaf, T * NV);
        PipeBarrier<PIPE_V>();
        Cast(outbf, betaf, RoundMode::CAST_RINT, T * NV);
        PipeBarrier<PIPE_V>();
        Cast(betaf, outbf, RoundMode::CAST_NONE, T * NV);
        PipeBarrier<PIPE_V>();
        Muls(qf, qf, 0.08838834764831845f, T * D);
        Exp(gf, gf, T * NV);
        PipeBarrier<PIPE_V>();

        uint32_t stateShape[2] = {VSTEP, D};
        uint32_t deltaShape[2] = {VSTEP, 1};
        DataCopy(stbf, basegm[vh * NSTATE + voff * D], NSTILE);
        PipeBarrier<PIPE_ALL>();
        Cast(stf, stbf, RoundMode::CAST_NONE, NSTILE);
        PipeBarrier<PIPE_V>();
        for (uint32_t t = 0; t < T; ++t) {
                const float gamma = gf.GetValue(t * NV + vh);
                const float b = betaf.GetValue(t * NV + vh);
                Muls(stf, stf, gamma, NSTILE);
                PipeBarrier<PIPE_V>();
                MatVecMul128(stf, kf[t * D], tmp, false);
                PipeBarrier<PIPE_V>();
                ReduceSum<float, Pattern::Reduce::AR, true>(delta, tmp, stateShape, true);
                PipeBarrier<PIPE_V>();
                Sub(delta, vf[t * VSTEP], delta, VSTEP);
                PipeBarrier<PIPE_V>();
                Muls(delta, delta, b, VSTEP);
                PipeBarrier<PIPE_V>();
                Broadcast<float, 2, 1>(tmp, delta, stateShape, deltaShape);
                PipeBarrier<PIPE_V>();
                MatVecMul128(tmp, kf[t * D], stf, true);
                PipeBarrier<PIPE_V>();
                MatVecMul128(stf, qf[t * D], tmp, false);
                PipeBarrier<PIPE_V>();
                ReduceSum<float, Pattern::Reduce::AR, true>(attn, tmp, stateShape, true);
                PipeBarrier<PIPE_V>();
                const uint32_t lane = t & 1;
                auto stbfOut = stbf[lane * NSTILE];
                auto outbfOut = outbf[lane * VSTEP];
                if (t >= 2) {
                    if (lane == 0) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                    else WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                }
                Cast(stbfOut, stf, RoundMode::CAST_RINT, NSTILE);
                Cast(outbfOut, attn, RoundMode::CAST_RINT, VSTEP);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                const uint32_t slot = static_cast<uint32_t>(idxgm.GetValue(t));
                DataCopy(stgm[(slot * NV + vh) * NSTATE + voff * D], stbfOut, NSTILE);
                DataCopy(outgm[(t * NV + vh) * D + voff], outbfOut, VSTEP);
                if (lane == 0) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                else SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
}

extern "C" int gdn_recurrent_baseptr_launch(void* stream, void* q, void* k,
    void* v, void* g, void* beta, void* baseState, void* state, void* out,
    void* stateIndices, void* numAccepted, uint32_t vTokenStride,
    uint32_t batch) {
    if (batch == 1) {
        gdn_recurrent_b1q8<<<24, nullptr, stream>>>(
            (uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)g,
            (uint8_t*)beta, (uint8_t*)state, (uint8_t*)out,
            (uint8_t*)stateIndices, (uint8_t*)numAccepted, vTokenStride);
    } else {
        gdn_recurrent_batch_b1q8<<<24, nullptr, stream>>>(
            (uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)g,
            (uint8_t*)beta, (uint8_t*)baseState, (uint8_t*)state,
            (uint8_t*)out, (uint8_t*)stateIndices,
            (uint8_t*)numAccepted, vTokenStride, batch);
    }
    return 0;
}
