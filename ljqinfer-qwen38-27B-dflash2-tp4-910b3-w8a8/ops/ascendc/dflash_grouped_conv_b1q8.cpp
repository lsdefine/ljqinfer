#include "kernel_operator.h"
using namespace AscendC;

// Fixed DFlash2 decode geometry: Q=8, C=5120, groups=320, group_size=16,
// taps=2. Each block owns 640 channels = 40 complete groups.
constexpr uint32_t Q = 8;
constexpr uint32_t C = 5120;
constexpr uint32_t GROUP_SIZE = 16;
constexpr uint32_t GROUPS = C / GROUP_SIZE;
// `coeff[:, side]` is a view of [Q,2,2,GROUPS], so consecutive q rows
// remain four GROUPS apart even though the visible tensor is [Q,2,GROUPS].
constexpr uint32_t DELTA_Q_STRIDE = 4 * GROUPS;
constexpr uint32_t TILE = 640;
constexpr uint32_t TILE_GROUPS = TILE / GROUP_SIZE;

extern "C" __global__ __aicore__ void dflash_grouped_conv_b1q8(
    GM_ADDR hidden, GM_ADDR delta, GM_ADDR base, GM_ADDR out,
    uint32_t side) {
    const uint32_t bi = GetBlockIdx();
    const uint32_t c0 = bi * TILE;
    if (c0 >= C || side >= 2) return;

    GlobalTensor<bfloat16_t> gh, gd, gb, go;
    gh.SetGlobalBuffer((__gm__ bfloat16_t*)hidden, Q * (uint64_t)C);
    gd.SetGlobalBuffer((__gm__ bfloat16_t*)delta, Q * 2ull * GROUPS);
    gb.SetGlobalBuffer((__gm__ bfloat16_t*)base, 2ull * 2 * C);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)out, Q * (uint64_t)C);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bHidden, bBase0, bBase1;
    TBuf<TPosition::VECCALC> bDeltaBF, bHiddenF, bBaseF, bDeltaF;
    TBuf<TPosition::VECCALC> bCoeffBF, bCoeffF, bProd0BF, bProd1BF;
    TBuf<TPosition::VECCALC> bWork0F, bWork1F, bOutBF;
    pipe.InitBuffer(bHidden, Q * TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bBase0, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bBase1, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bDeltaBF, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bHiddenF, TILE * sizeof(float));
    pipe.InitBuffer(bBaseF, TILE * sizeof(float));
    pipe.InitBuffer(bDeltaF, TILE * sizeof(float));
    pipe.InitBuffer(bCoeffBF, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bCoeffF, TILE * sizeof(float));
    pipe.InitBuffer(bProd0BF, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bProd1BF, TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bWork0F, TILE * sizeof(float));
    pipe.InitBuffer(bWork1F, TILE * sizeof(float));
    pipe.InitBuffer(bOutBF, TILE * sizeof(bfloat16_t));

    LocalTensor<bfloat16_t> h = bHidden.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> base0 = bBase0.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> base1 = bBase1.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> deltaBF = bDeltaBF.Get<bfloat16_t>();
    LocalTensor<float> hiddenF = bHiddenF.Get<float>();
    LocalTensor<float> baseF = bBaseF.Get<float>();
    LocalTensor<float> deltaF = bDeltaF.Get<float>();
    LocalTensor<bfloat16_t> coeffBF = bCoeffBF.Get<bfloat16_t>();
    LocalTensor<float> coeffF = bCoeffF.Get<float>();
    LocalTensor<bfloat16_t> prod0BF = bProd0BF.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> prod1BF = bProd1BF.Get<bfloat16_t>();
    LocalTensor<float> work0F = bWork0F.Get<float>();
    LocalTensor<float> work1F = bWork1F.Get<float>();
    LocalTensor<bfloat16_t> outBF = bOutBF.Get<bfloat16_t>();

    for (uint32_t q = 0; q < Q; ++q)
        DataCopy(h[q * TILE], gh[q * C + c0], TILE);
    DataCopy(base0, gb[(side * 2 + 0) * C + c0], TILE);
    DataCopy(base1, gb[(side * 2 + 1) * C + c0], TILE);
    PipeBarrier<PIPE_ALL>();

    const uint32_t g0 = c0 / GROUP_SIZE;
    for (uint32_t q = 0; q < Q; ++q) {
        Cast(hiddenF, h[q * TILE], RoundMode::CAST_NONE, TILE);
        Cast(baseF, base0, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t g = 0; g < TILE_GROUPS; ++g) {
            bfloat16_t scalar = gd.GetValue(q * DELTA_Q_STRIDE + g0 + g);
            Duplicate(deltaBF[g * GROUP_SIZE], scalar, GROUP_SIZE);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(deltaF, deltaBF, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        Add(work0F, baseF, deltaF, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(coeffBF, work0F, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(coeffF, coeffBF, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        Mul(work0F, hiddenF, coeffF, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(prod0BF, work0F, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_ALL>();

        if (q == 0) {
            DataCopy(go[q * C + c0], prod0BF, TILE);
            PipeBarrier<PIPE_ALL>();
            continue;
        }

        Cast(hiddenF, h[(q - 1) * TILE], RoundMode::CAST_NONE, TILE);
        Cast(baseF, base1, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t g = 0; g < TILE_GROUPS; ++g) {
            bfloat16_t scalar = gd.GetValue(q * DELTA_Q_STRIDE + GROUPS + g0 + g);
            Duplicate(deltaBF[g * GROUP_SIZE], scalar, GROUP_SIZE);
        }
        PipeBarrier<PIPE_ALL>();
        Cast(deltaF, deltaBF, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        Add(work1F, baseF, deltaF, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(coeffBF, work1F, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(coeffF, coeffBF, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        Mul(work1F, hiddenF, coeffF, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(prod1BF, work1F, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_ALL>();

        Cast(work0F, prod0BF, RoundMode::CAST_NONE, TILE);
        Cast(work1F, prod1BF, RoundMode::CAST_NONE, TILE);
        PipeBarrier<PIPE_ALL>();
        Add(work0F, work0F, work1F, TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(outBF, work0F, RoundMode::CAST_RINT, TILE);
        PipeBarrier<PIPE_ALL>();
        DataCopy(go[q * C + c0], outBF, TILE);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int dflash_grouped_conv_b1q8_launch(
    void* stream, void* hidden, void* delta, void* base, void* out,
    uint32_t side) {
    dflash_grouped_conv_b1q8<<<8, nullptr, stream>>>(
        (uint8_t*)hidden, (uint8_t*)delta, (uint8_t*)base,
        (uint8_t*)out, side);
    return 0;
}
