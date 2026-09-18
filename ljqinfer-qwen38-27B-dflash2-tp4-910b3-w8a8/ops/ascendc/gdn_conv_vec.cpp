#include "kernel_operator.h"
using namespace AscendC;

// Fixed production geometry: B=1, Q=8, K=4. Weight is prepacked [K,C].
constexpr uint32_t Q = 8;
constexpr uint32_t K = 4;
constexpr uint32_t S = Q + K - 1;
constexpr uint32_t MAX_TILE = 640;

extern "C" __global__ __aicore__ void gdn_conv_vec(
    GM_ADDR base, GM_ADDR x, GM_ADDR weight_kc, GM_ADDR out, GM_ADDR pending,
    uint32_t channels, uint32_t input_token_stride, uint32_t tile) {
    uint32_t bi = GetBlockIdx();
    uint32_t c0 = bi * tile;
    if (c0 >= channels || tile > MAX_TILE) return;
    uint32_t n = (c0 + tile <= channels) ? tile : (channels - c0);

    GlobalTensor<bfloat16_t> gb, gx, gw, go, gp;
    gb.SetGlobalBuffer((__gm__ bfloat16_t*)base, 3ull * channels);
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)x, Q * (uint64_t)input_token_stride);
    gw.SetGlobalBuffer((__gm__ bfloat16_t*)weight_kc, K * (uint64_t)channels);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)out, Q * (uint64_t)channels);
    gp.SetGlobalBuffer((__gm__ bfloat16_t*)pending, Q * (uint64_t)channels);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bSeq, bW, bSeqF, bWF, bProdBF, bF, bAcc, bOut;
    pipe.InitBuffer(bSeq, S * MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bW, K * MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bSeqF, S * MAX_TILE * sizeof(float));
    pipe.InitBuffer(bWF, K * MAX_TILE * sizeof(float));
    pipe.InitBuffer(bProdBF, MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bF, MAX_TILE * sizeof(float));
    pipe.InitBuffer(bAcc, MAX_TILE * sizeof(float));
    pipe.InitBuffer(bOut, MAX_TILE * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> seq = bSeq.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> w = bW.Get<bfloat16_t>();
    LocalTensor<float> seqf = bSeqF.Get<float>();
    LocalTensor<float> wf = bWF.Get<float>();
    LocalTensor<bfloat16_t> prodBF = bProdBF.Get<bfloat16_t>();
    LocalTensor<float> f = bF.Get<float>();
    LocalTensor<float> acc = bAcc.Get<float>();
    LocalTensor<bfloat16_t> ot = bOut.Get<bfloat16_t>();

    for (uint32_t s = 0; s < 3; ++s)
        DataCopy(seq[s * MAX_TILE], gb[s * channels + c0], n);
    for (uint32_t q = 0; q < Q; ++q)
        DataCopy(seq[(q + 3) * MAX_TILE], gx[q * input_token_stride + c0], n);
    for (uint32_t k = 0; k < K; ++k)
        DataCopy(w[k * MAX_TILE], gw[k * channels + c0], n);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t s = 0; s < S; ++s) {
        Cast(seqf[s * MAX_TILE], seq[s * MAX_TILE], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t k = 0; k < K; ++k) {
        Cast(wf[k * MAX_TILE], w[k * MAX_TILE], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t q = 0; q < Q; ++q) {
        Duplicate(acc, 0.0f, n);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t k = 0; k < K; ++k) {
            Mul(f, seqf[(q + k) * MAX_TILE], wf[k * MAX_TILE], n);
            PipeBarrier<PIPE_ALL>();
            Cast(prodBF, f, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            Cast(f, prodBF, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_ALL>();
            Add(acc, acc, f, n);
            PipeBarrier<PIPE_ALL>();
        }
        Cast(ot, acc, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_ALL>();
        DataCopy(go[q * channels + c0], ot, n);
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t q = 0; q < Q; ++q) {
        DataCopy(gp[q * channels + c0], seq[(q + 3) * MAX_TILE], n);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void gdn_conv_vec_batch(
    GM_ADDR base, GM_ADDR x, GM_ADDR weight_kc, GM_ADDR out, GM_ADDR pending,
    uint32_t channels, uint32_t input_token_stride, uint32_t tile,
    uint32_t input_batch_stride, uint32_t base_batch_stride,
    uint32_t out_batch_stride, uint32_t pending_batch_stride,
    uint32_t batch) {
    // Preserve four channel tiles per row; parallelize independent batch rows.
    // Each block owns one (batch row, channel tile), with disjoint output writes.
    uint32_t blocks = (channels + tile - 1) / tile;
    uint32_t batchRow = GetBlockIdx() / blocks;
    uint32_t c0 = (GetBlockIdx() % blocks) * tile;
    if (c0 >= channels || tile > MAX_TILE) return;
    uint32_t n = (c0 + tile <= channels) ? tile : (channels - c0);

    GlobalTensor<bfloat16_t> gb, gx, gw, go;
    gw.SetGlobalBuffer((__gm__ bfloat16_t*)weight_kc, K * (uint64_t)channels);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bSeq, bW, bSeqF, bWF, bProdBF, bF, bAcc, bOut;
    pipe.InitBuffer(bSeq, S * MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bW, K * MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bSeqF, S * MAX_TILE * sizeof(float));
    pipe.InitBuffer(bWF, K * MAX_TILE * sizeof(float));
    pipe.InitBuffer(bProdBF, MAX_TILE * sizeof(bfloat16_t));
    pipe.InitBuffer(bF, MAX_TILE * sizeof(float));
    pipe.InitBuffer(bAcc, MAX_TILE * sizeof(float));
    pipe.InitBuffer(bOut, MAX_TILE * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> seq = bSeq.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> w = bW.Get<bfloat16_t>();
    LocalTensor<float> seqf = bSeqF.Get<float>();
    LocalTensor<float> wf = bWF.Get<float>();
    LocalTensor<bfloat16_t> prodBF = bProdBF.Get<bfloat16_t>();
    LocalTensor<float> f = bF.Get<float>();
    LocalTensor<float> acc = bAcc.Get<float>();
    LocalTensor<bfloat16_t> ot = bOut.Get<bfloat16_t>();

    // Weight is row-invariant; load and widen it once per channel block.
    for (uint32_t k = 0; k < K; ++k)
        DataCopy(w[k * MAX_TILE], gw[k * channels + c0], n);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t k = 0; k < K; ++k) {
        Cast(wf[k * MAX_TILE], w[k * MAX_TILE], RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_ALL>();
    }

    for (uint32_t row = batchRow; row < batchRow + 1; ++row) {
        gb.SetGlobalBuffer((__gm__ bfloat16_t*)base +
                           (uint64_t)row * base_batch_stride, 3ull * channels);
        gx.SetGlobalBuffer((__gm__ bfloat16_t*)x +
                           (uint64_t)row * input_batch_stride,
                           Q * (uint64_t)input_token_stride);
        go.SetGlobalBuffer((__gm__ bfloat16_t*)out +
                           (uint64_t)row * out_batch_stride,
                           Q * (uint64_t)channels);

        for (uint32_t s = 0; s < 3; ++s)
            DataCopy(seq[s * MAX_TILE], gb[s * channels + c0], n);
        for (uint32_t q = 0; q < Q; ++q)
            DataCopy(seq[(q + 3) * MAX_TILE],
                     gx[q * input_token_stride + c0], n);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t s = 0; s < S; ++s) {
            Cast(seqf[s * MAX_TILE], seq[s * MAX_TILE],
                 RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_ALL>();
        }

        for (uint32_t q = 0; q < Q; ++q) {
            Duplicate(acc, 0.0f, n);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0; k < K; ++k) {
                Mul(f, seqf[(q + k) * MAX_TILE], wf[k * MAX_TILE], n);
                PipeBarrier<PIPE_ALL>();
                Cast(prodBF, f, RoundMode::CAST_RINT, n);
                PipeBarrier<PIPE_ALL>();
                Cast(f, prodBF, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_ALL>();
                Add(acc, acc, f, n);
                PipeBarrier<PIPE_ALL>();
            }
            Cast(ot, acc, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(go[q * channels + c0], ot, n);
            // `ot` is reused by the next q/row; wait until MTE3 has consumed it.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

extern "C" int gdn_conv_vec_launch(void* stream, void* base, void* x,
    void* weight_kc, void* out, void* pending, uint32_t channels,
    uint32_t inputTokenStride, uint32_t inputBatchStride,
    uint32_t baseBatchStride, uint32_t outBatchStride,
    uint32_t pendingBatchStride, uint32_t tile,
    uint32_t blocksPerBatch, uint32_t batch) {
    // Preserve the proven B1 launch exactly.  For B2-B4, use one four-block
    // launch and process rows sequentially inside each channel block.
    if (batch == 1) {
        gdn_conv_vec<<<blocksPerBatch, nullptr, stream>>>(
            (uint8_t*)base, (uint8_t*)x, (uint8_t*)weight_kc,
            (uint8_t*)out, (uint8_t*)pending,
            channels, inputTokenStride, tile);
    } else {
        // This kernel writes only `out`; the wrapper performs the graph-
        // capturable `pending.copy_(x)` on the same stream for B2-B4.
        gdn_conv_vec_batch<<<blocksPerBatch * batch, nullptr, stream>>>(
            (uint8_t*)base, (uint8_t*)x, (uint8_t*)weight_kc,
            (uint8_t*)out, (uint8_t*)pending,
            channels, inputTokenStride, tile, inputBatchStride,
            baseBatchStride, outBatchStride, pendingBatchStride,
            batch);
    }
    return 0;
}
