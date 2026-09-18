#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t WIDTH = 128;
constexpr uint32_t PAD = 128;
constexpr float EPS = 1e-6f;

__aicore__ inline void NormalizeOne(
    GlobalTensor<bfloat16_t> &gx, GlobalTensor<bfloat16_t> &gy,
    uint64_t src_off, uint64_t dst_off,
    LocalTensor<bfloat16_t> &h, LocalTensor<float> &f,
    LocalTensor<float> &sq, LocalTensor<float> &red,
    LocalTensor<bfloat16_t> &scaleBF, LocalTensor<bfloat16_t> &out) {
    DataCopy(h, gx[src_off], WIDTH);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(f, h, RoundMode::CAST_NONE, WIDTH);
    PipeBarrier<PIPE_V>();
    Mul(sq, f, f, WIDTH);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, sq, red, WIDTH);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float sum = red.GetValue(0);
    float scale32 = 1.0f / sqrt(sum < EPS ? EPS : sum);
    Duplicate(red, scale32, 1);
    PipeBarrier<PIPE_V>();
    Cast(scaleBF, red, RoundMode::CAST_RINT, 1);
    PipeBarrier<PIPE_V>();
    Cast(red, scaleBF, RoundMode::CAST_NONE, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float scale = red.GetValue(0);
    Muls(f, f, scale, WIDTH);
    PipeBarrier<PIPE_V>();
    Cast(out, f, RoundMode::CAST_RINT, WIDTH);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(gy[dst_off], out, WIDTH);
}

extern "C" __global__ __aicore__ void gdn_l2_pair(
    GM_ADDR q, GM_ADDR k, GM_ADDR qo, GM_ADDR ko, uint32_t rows,
    uint32_t heads, uint32_t group_stride) {
    uint32_t r = GetBlockIdx();
    if (r >= rows) return;
    uint64_t groups = rows / heads;
    uint64_t input_span = (groups - 1) * group_stride + heads * WIDTH;
    GlobalTensor<bfloat16_t> qx, kx, qy, ky;
    qx.SetGlobalBuffer((__gm__ bfloat16_t*)q, input_span);
    kx.SetGlobalBuffer((__gm__ bfloat16_t*)k, input_span);
    qy.SetGlobalBuffer((__gm__ bfloat16_t*)qo, (uint64_t)rows * WIDTH);
    ky.SetGlobalBuffer((__gm__ bfloat16_t*)ko, (uint64_t)rows * WIDTH);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bh, bf, bsq, bred, bscale, bout;
    pipe.InitBuffer(bh, PAD * sizeof(bfloat16_t));
    pipe.InitBuffer(bf, PAD * sizeof(float));
    pipe.InitBuffer(bsq, PAD * sizeof(float));
    pipe.InitBuffer(bred, PAD * sizeof(float));
    pipe.InitBuffer(bscale, PAD * sizeof(bfloat16_t));
    pipe.InitBuffer(bout, PAD * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> h = bh.Get<bfloat16_t>();
    LocalTensor<float> f = bf.Get<float>();
    LocalTensor<float> sq = bsq.Get<float>();
    LocalTensor<float> red = bred.Get<float>();
    LocalTensor<bfloat16_t> scaleBF = bscale.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> out = bout.Get<bfloat16_t>();

    uint64_t src_off = (uint64_t)(r / heads) * group_stride +
                       (uint64_t)(r % heads) * WIDTH;
    uint64_t dst_off = (uint64_t)r * WIDTH;
    NormalizeOne(qx, qy, src_off, dst_off, h, f, sq, red, scaleBF, out);
    PipeBarrier<PIPE_ALL>();
    NormalizeOne(kx, ky, src_off, dst_off, h, f, sq, red, scaleBF, out);
}

extern "C" int gdn_l2_pair_launch(void* stream, void* q, void* k,
    void* qo, void* ko, uint32_t rows, uint32_t heads,
    uint32_t group_stride, uint32_t blocks) {
    (void)blocks;
    gdn_l2_pair<<<rows, nullptr, stream>>>((uint8_t*)q, (uint8_t*)k,
        (uint8_t*)qo, (uint8_t*)ko, rows, heads, group_stride);
    return 0;
}
