// Decode-only INT8[N,K] * FP32[N] -> BF16[N,K], row-major.
// Caller owns all storage and passes current stream; no allocation/sync.
// K is 32-aligned and <=5120. Buffers must be aligned and non-overlapping.
#include "kernel_operator.h"
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
constexpr uint32_t MAXK=5120;
// Packed W8 expansion. Queue storage is on-chip; no device allocation.
// Preserve the original active-core guard and all arithmetic/rounding steps.
extern "C" __global__ __aicore__ void decode_w8_expand(GM_ADDR w, GM_ADDR s, GM_ADDR y,
                                                 uint32_t rows, uint32_t k) {
    const uint32_t cores = GetBlockNum();
    const uint32_t me = GetBlockIdx();
    if (me >= cores) return;
    const uint32_t pack = MAXK / k;
    TPipe pipe;
    TQue<TPosition::VECIN, 2> qw, qs;
    TQue<TPosition::VECOUT, 2> qy;
    TBuf<TPosition::VECCALC> bh, bf;
    pipe.InitBuffer(qw, 2, MAXK * sizeof(int8_t));
    pipe.InitBuffer(qs, 2, ((MAXK / 32 * sizeof(float) + 31) / 32) * 32);
    pipe.InitBuffer(qy, 2, MAXK * sizeof(bfloat16_t));
    pipe.InitBuffer(bh, MAXK * sizeof(half));
    pipe.InitBuffer(bf, MAXK * sizeof(float));
    auto lh = bh.Get<half>();
    auto lf = bf.Get<float>();
    GlobalTensor<int8_t> gw; gw.SetGlobalBuffer((__gm__ int8_t*)w);
    GlobalTensor<float> gs; gs.SetGlobalBuffer((__gm__ float*)s);
    GlobalTensor<bfloat16_t> gy; gy.SetGlobalBuffer((__gm__ bfloat16_t*)y);
    const event_t es = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    DataCopyExtParams sp; sp.blockCount = 1; sp.blockLen = 4;
    sp.srcStride = 0; sp.dstStride = 0;
    DataCopyPadExtParams<float> pp; pp.isPad = false; pp.leftPadding = 0;
    pp.rightPadding = 0; pp.paddingValue = 0;
    uint32_t first = me * pack;
    if (first < rows) {
        uint32_t count = rows - first < pack ? rows - first : pack;
        auto input = qw.AllocTensor<int8_t>();
        DataCopy(input, gw[uint64_t(first) * k], count * k);
        qw.EnQue(input);
        auto scales = qs.AllocTensor<float>();
        sp.blockLen = count * sizeof(float);
        DataCopyPad(scales, gs[first], sp, pp);
        qs.EnQue(scales);
    }
    for (uint32_t row = first; row < rows; row += cores * pack) {
        uint32_t count = rows - row < pack ? rows - row : pack;
        int32_t len = int32_t(count * k);
        auto input = qw.DeQue<int8_t>();
        auto scales = qs.DeQue<float>();
        SetFlag<HardEvent::MTE2_S>(es);
        WaitFlag<HardEvent::MTE2_S>(es);
        Cast(lh, input, RoundMode::CAST_NONE, len);
        qw.FreeTensor(input);
        // Next independent tile overlaps with conversion of the current tile.
        uint32_t next = row + cores * pack;
        if (next < rows) {
            uint32_t nc = rows - next < pack ? rows - next : pack;
            auto ni = qw.AllocTensor<int8_t>();
            DataCopy(ni, gw[uint64_t(next) * k], nc * k);
            qw.EnQue(ni);
            auto ns = qs.AllocTensor<float>();
            sp.blockLen = nc * sizeof(float);
            DataCopyPad(ns, gs[next], sp, pp);
            qs.EnQue(ns);
        }
        PipeBarrier<PIPE_V>();
        Cast(lf, lh, RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < count; ++j)
            Muls(lf[j*k], lf[j*k], scales.GetValue(j), int32_t(k));
        qs.FreeTensor(scales);
        PipeBarrier<PIPE_V>();
        auto output = qy.AllocTensor<bfloat16_t>();
        Cast(output, lf, RoundMode::CAST_RINT, len);
        qy.EnQue(output);
        output = qy.DeQue<bfloat16_t>();
        DataCopy(gy[uint64_t(row)*k], output, uint32_t(len));
        qy.FreeTensor(output);
        PipeBarrier<PIPE_V>(); // shared conversion scratch must not be overwritten early
    }
}

extern "C" int dec_w8_expand(void* stream,void* w,void* s,void* y,uint32_t n,uint32_t k) {
    if(!stream||!w||!s||!y||!n||!k||k>MAXK||k%32) return -1;
    if((uintptr_t(w)|uintptr_t(s)|uintptr_t(y))&31) return -1;
    uint64_t wn=uint64_t(n)*k, sn=uint64_t(n)*4, yn=wn*2;
    uintptr_t a=uintptr_t(y), b=uintptr_t(w), c=uintptr_t(s);
    if((a<=b ? b-a<yn : a-b<wn)||(a<=c ? c-a<yn : a-c<sn)) return -1;
    uint32_t blocks=n<40?n:40;
    decode_w8_expand<<<blocks,nullptr,stream>>>((uint8_t*)w,(uint8_t*)s,(uint8_t*)y,n,k);
    return 0;
}
