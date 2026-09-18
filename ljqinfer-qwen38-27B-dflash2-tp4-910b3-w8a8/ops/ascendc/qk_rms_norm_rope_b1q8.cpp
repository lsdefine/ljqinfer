#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t HEAD_DIM = 256;
constexpr uint32_t ROTARY_HALF = 32;
constexpr float RMS_EPS = 1e-6f;
extern "C" __global__ __aicore__ void qk_norm_rope(
    GM_ADDR q, GM_ADDR k, GM_ADDR qdelta, GM_ADDR kdelta,
    GM_ADDR cos, GM_ADDR sin, GM_ADDR qo, GM_ADDR ko,
    uint32_t qrows, uint32_t krows, uint32_t qheads, uint32_t kheads,
    uint32_t q_token_stride, uint32_t q_head_stride,
    uint32_t k_token_stride, uint32_t k_head_stride) {
    uint32_t bi = GetBlockIdx();
    bool is_k = bi >= qrows;
    uint32_t row = is_k ? bi - qrows : bi;
    uint32_t rows = is_k ? krows : qrows;
    uint32_t heads = is_k ? kheads : qheads;
    if (row >= rows) return;
    uint32_t token = row / heads;
    uint32_t head = row % heads;
    uint32_t token_stride = is_k ? k_token_stride : q_token_stride;
    uint32_t head_stride = is_k ? k_head_stride : q_head_stride;
    GlobalTensor<bfloat16_t> gx, gg, gc, gs, gy;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)(is_k ? k : q));
    gg.SetGlobalBuffer((__gm__ bfloat16_t*)(is_k ? kdelta : qdelta), HEAD_DIM);
    gc.SetGlobalBuffer((__gm__ bfloat16_t*)cos);
    gs.SetGlobalBuffer((__gm__ bfloat16_t*)sin);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)(is_k ? ko : qo),
                       (uint64_t)rows * HEAD_DIM);
    TPipe pipe;
    TBuf<TPosition::VECCALC> bx, bg, bxf, bgf, bsq, bred, bn, bc, bs, bt1, bt2;
    pipe.InitBuffer(bx, HEAD_DIM * sizeof(bfloat16_t));
    pipe.InitBuffer(bg, HEAD_DIM * sizeof(bfloat16_t));
    pipe.InitBuffer(bxf, HEAD_DIM * sizeof(float));
    pipe.InitBuffer(bgf, HEAD_DIM * sizeof(float));
    pipe.InitBuffer(bsq, HEAD_DIM * sizeof(float));
    pipe.InitBuffer(bred, HEAD_DIM * sizeof(float));
    pipe.InitBuffer(bn, HEAD_DIM * sizeof(bfloat16_t));
    pipe.InitBuffer(bc, ROTARY_HALF * sizeof(bfloat16_t));
    pipe.InitBuffer(bs, ROTARY_HALF * sizeof(bfloat16_t));
    pipe.InitBuffer(bt1, ROTARY_HALF * sizeof(float));
    pipe.InitBuffer(bt2, ROTARY_HALF * sizeof(float));
    auto x=bx.Get<bfloat16_t>(); auto g=bg.Get<bfloat16_t>();
    auto xf=bxf.Get<float>(); auto gf=bgf.Get<float>();
    auto sq=bsq.Get<float>(); auto red=bred.Get<float>(); auto n=bn.Get<bfloat16_t>();
    auto c=bc.Get<bfloat16_t>(); auto s=bs.Get<bfloat16_t>();
    auto t1=bt1.Get<float>(); auto t2=bt2.Get<float>();
    uint64_t src = (uint64_t)token * token_stride
                 + (uint64_t)head * head_stride;
    DataCopy(x, gx[src], HEAD_DIM);
    DataCopy(g, gg[0], HEAD_DIM);
    DataCopy(c, gc[(uint64_t)token * ROTARY_HALF], ROTARY_HALF);
    DataCopy(s, gs[(uint64_t)token * ROTARY_HALF], ROTARY_HALF);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(xf, x, RoundMode::CAST_NONE, HEAD_DIM);
    Cast(gf, g, RoundMode::CAST_NONE, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Mul(sq, xf, xf, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, sq, red, HEAD_DIM);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float mean = red.GetValue(0) * (1.0f / HEAD_DIM);
    float scale = 1.0f / sqrt(mean + RMS_EPS);
    // The checkpoint stores a zero-centered delta. Round delta + 1 to BF16
    // before multiplication to match torch_npu.npu_rms_norm exactly.
    Adds(gf, gf, 1.0f, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Cast(g, gf, RoundMode::CAST_RINT, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Cast(gf, g, RoundMode::CAST_NONE, HEAD_DIM);
    Muls(xf, xf, scale, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Mul(xf, xf, gf, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Cast(n, xf, RoundMode::CAST_RINT, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Cast(xf, n, RoundMode::CAST_NONE, HEAD_DIM);
    Cast(gf, c, RoundMode::CAST_NONE, ROTARY_HALF);
    Cast(sq, s, RoundMode::CAST_NONE, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    // Preserve the PyTorch expression boundary: each product rounds to BF16
    // before the NeoX add/sub, otherwise the graph is not bitwise equivalent.
    Mul(red, xf, gf, ROTARY_HALF);
    Mul(t2, xf[ROTARY_HALF], sq, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(x, red, RoundMode::CAST_RINT, ROTARY_HALF);
    Cast(g, t2, RoundMode::CAST_RINT, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(red, x, RoundMode::CAST_NONE, ROTARY_HALF);
    Cast(t2, g, RoundMode::CAST_NONE, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Sub(red, red, t2, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(n, red, RoundMode::CAST_RINT, ROTARY_HALF);

    Mul(red, xf[ROTARY_HALF], gf, ROTARY_HALF);
    Mul(t2, xf, sq, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(x, red, RoundMode::CAST_RINT, ROTARY_HALF);
    Cast(g, t2, RoundMode::CAST_RINT, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(red, x, RoundMode::CAST_NONE, ROTARY_HALF);
    Cast(t2, g, RoundMode::CAST_NONE, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Add(red, red, t2, ROTARY_HALF);
    PipeBarrier<PIPE_V>();
    Cast(n[ROTARY_HALF], red, RoundMode::CAST_RINT, ROTARY_HALF);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(gy[(uint64_t)row * HEAD_DIM], n, HEAD_DIM);
}
extern "C" int qk_norm_rope_launch(void* stream, void* q, void* k,
    void* qdelta, void* kdelta, void* cos, void* sin, void* qo, void* ko,
    uint32_t qrows, uint32_t krows, uint32_t qheads, uint32_t kheads,
    uint32_t q_token_stride, uint32_t q_head_stride,
    uint32_t k_token_stride, uint32_t k_head_stride) {
    qk_norm_rope<<<qrows + krows, nullptr, stream>>>(
        (uint8_t*)q, (uint8_t*)k, (uint8_t*)qdelta, (uint8_t*)kdelta,
        (uint8_t*)cos, (uint8_t*)sin, (uint8_t*)qo, (uint8_t*)ko,
        qrows, krows, qheads, kheads, q_token_stride, q_head_stride,
        k_token_stride, k_head_stride);
    return 0;
}
