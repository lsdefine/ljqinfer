#include "kernel_operator.h"
using namespace AscendC;

// W4A8 weight expansion. Both kernels stream rows: read once, write once, so
// the cost is the traffic and nothing else.
//
// Build (the .so is not in the tree; this is the command that makes it):
//   $ASCEND/tools/ccec_compiler/bin/bisheng -x cce --cce-aicore-arch=dav-c220 \
//     -O2 -std=c++17 -shared -fPIC -I$ASCEND/compiler/tikcpp/tikcfw{,/impl,/interface} \
//     -I$ASCEND/include ops/kernels/dequant.cpp -o ops/kernels/libdq.so

constexpr uint32_t MAXK = 5120;             // widest row either kernel serves

// int8, one float scale per output channel.
//   w : int8     [rows, k]
//   s : float32  [rows]
//   y : bfloat16 [rows, k]
extern "C" __global__ __aicore__ void w8_dequant(GM_ADDR w, GM_ADDR s, GM_ADDR y,
                                                 uint32_t rows, uint32_t k) {
    const uint32_t cores = GetBlockNum() * 2;
    const uint32_t me    = GetBlockIdx();

    TPipe pipe;
    TBuf<TPosition::VECCALC> bw, bh, bf, bs;
    pipe.InitBuffer(bw, MAXK * sizeof(int8_t));
    pipe.InitBuffer(bh, MAXK * sizeof(half));
    pipe.InitBuffer(bf, MAXK * sizeof(float));
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
        Muls(lf, lf, ls.GetValue(0), k);
        PipeBarrier<PIPE_ALL>();
        Cast(lb, lf, RoundMode::CAST_RINT, k);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gy[(uint64_t)row * k], lb, k);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void w8_dequant_launch(uint32_t blockDim, void* stream, uint8_t* w,
                                  uint8_t* s, uint8_t* y, uint32_t rows, uint32_t k) {
    w8_dequant<<<blockDim, nullptr, stream>>>(w, s, y, rows, k);
}


// Two int4 output channels per byte, one float scale per output channel.
// Channel 2i is the low nibble, 2i+1 the high one, so a packed row expands
// into the two consecutive rows below it and both writes stay contiguous.
//   w    : int8     [rows, k]        rows = out_rows / 2
//   s    : float32  [2 * rows]
//   tbll : float32  [256]            byte -> value of its low nibble
//   tblh : float32  [256]            byte -> value of its high nibble
//   y    : bfloat16 [2 * rows, k]
template<bool Active>
__global__ __aicore__ void w4_dequant(GM_ADDR w, GM_ADDR s, GM_ADDR tbll,
                                                 GM_ADDR tblh, GM_ADDR y,
                                                 uint32_t rows, uint32_t k, GM_ADDR counts, uint32_t experts) {
    const uint32_t cores = GetBlockNum() * 2;
    const uint32_t me    = GetBlockIdx();

    TPipe pipe;
    TQue<TPosition::VECIN,  2> qw, qs;
    TQue<TPosition::VECOUT, 2> qy;
    pipe.InitBuffer(qw, 2, MAXK * sizeof(uint8_t));
    pipe.InitBuffer(qs, 2, 512 * sizeof(float));
    pipe.InitBuffer(qy, 2, MAXK * sizeof(bfloat16_t));
    TBuf<TPosition::VECCALC> bh, bo, bf, bg, btl, bth;
    pipe.InitBuffer(bh, MAXK * sizeof(half));
    pipe.InitBuffer(bo, MAXK * sizeof(int32_t));
    pipe.InitBuffer(bf, MAXK * sizeof(float));
    pipe.InitBuffer(bg, MAXK * sizeof(float));
    pipe.InitBuffer(btl, 256 * sizeof(float));
    pipe.InitBuffer(bth, 256 * sizeof(float));

    LocalTensor<half>    lh = bh.Get<half>();
    LocalTensor<int32_t> lo = bo.Get<int32_t>();
    LocalTensor<float>   lf = bf.Get<float>();
    LocalTensor<float>   lg = bg.Get<float>();
    LocalTensor<float>   ltl = btl.Get<float>();
    LocalTensor<float>   lth = bth.Get<float>();

    GlobalTensor<float> gtl; gtl.SetGlobalBuffer((__gm__ float*)tbll, 256);
    GlobalTensor<float> gth; gth.SetGlobalBuffer((__gm__ float*)tblh, 256);
    DataCopy(ltl, gtl, 256);
    DataCopy(lth, gth, 256);

    GlobalTensor<uint8_t>    gw; gw.SetGlobalBuffer((__gm__ uint8_t*)w);
    GlobalTensor<float>      gs; gs.SetGlobalBuffer((__gm__ float*)s);
    GlobalTensor<bfloat16_t> gy; gy.SetGlobalBuffer((__gm__ bfloat16_t*)y);

    // A narrow row leaves the vector units idle waiting on the next fetch, so
    // as many rows as the buffer holds travel together; a full-width row makes
    // this one and the loop is exactly what it was before.
    const uint32_t pack = MAXK / k;

    // The queue only orders the fetch against vector work; the scales are
    // read by the scalar unit, which has to be told to wait as well.
    const event_t evs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));


    DataCopyExtParams sp; sp.blockCount = 1; sp.blockLen = 2 * sizeof(float);
    sp.srcStride = 0; sp.dstStride = 0;
    DataCopyPadExtParams<float> pp; pp.isPad = false; pp.leftPadding = 0;
    pp.rightPadding = 0; pp.paddingValue = 0;

    // The two halves of a packed row land one row apart, so each half leaves
    // as a strided burst rather than a write per row.
    DataCopyExtParams yp; yp.blockCount = 1;
    yp.blockLen = k * sizeof(bfloat16_t);
    yp.srcStride = 0; yp.dstStride = k * sizeof(bfloat16_t);

    TBuf<TPosition::VECCALC> bc;
    if constexpr (Active) pipe.InitBuffer(bc, 384 * sizeof(int64_t));
    LocalTensor<int64_t> lcounters;
    if constexpr (Active) {
    lcounters = bc.Get<int64_t>();
    GlobalTensor<int64_t> gc; gc.SetGlobalBuffer((__gm__ int64_t*)counts);
    DataCopy(lcounters, gc, experts);
    SetFlag<HardEvent::MTE2_S>(evs);
    WaitFlag<HardEvent::MTE2_S>(evs);
    }
    const uint32_t expert_rows = rows / experts;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        if constexpr (Active) {
            if (lcounters.GetValue(expert) == 0) continue;
        }
        const uint32_t end = (expert + 1) * expert_rows;
        uint32_t first = expert * expert_rows + me * pack;
        if (first < end) {
            uint32_t count = end - first < pack ? end - first : pack;
            auto input = qw.AllocTensor<uint8_t>();
            DataCopy(input, gw[(uint64_t)first * k], count * k);
            qw.EnQue(input);
            auto scale = qs.AllocTensor<float>();
            sp.blockLen = count * 2 * sizeof(float);
            DataCopyPad(scale, gs[(uint64_t)first * 2], sp, pp);
            qs.EnQue(scale);
        }
        for (uint32_t row = first; row < end; row += cores * pack) {
            const uint32_t cnt = end - row < pack ? end - row : pack;
            const uint32_t len = cnt * k;
            auto lw = qw.DeQue<uint8_t>();
            auto ls = qs.DeQue<float>();
            SetFlag<HardEvent::MTE2_S>(evs);
            WaitFlag<HardEvent::MTE2_S>(evs);

            // byte -> gather offset, the same trick the FP4 bank used
            Cast(lh, lw, RoundMode::CAST_NONE, len);
            qw.FreeTensor(lw);
            // Fetch the next independent tile while vector work consumes this one.
            uint32_t next = row + cores * pack;
            if (next < end) {
                uint32_t count = end - next < pack ? end - next : pack;
                auto input = qw.AllocTensor<uint8_t>();
                DataCopy(input, gw[(uint64_t)next * k], count * k);
                qw.EnQue(input);
                auto scale = qs.AllocTensor<float>();
                sp.blockLen = count * 2 * sizeof(float);
                DataCopyPad(scale, gs[(uint64_t)next * 2], sp, pp);
                qs.EnQue(scale);
            }
            PipeBarrier<PIPE_V>();
            Muls(lh, lh, (half)4.0, len);
            PipeBarrier<PIPE_V>();
            Cast(lo, lh, RoundMode::CAST_RINT, len);
            PipeBarrier<PIPE_V>();
            Gather(lf, ltl, lo.ReinterpretCast<uint32_t>(), (uint32_t)0, len);
            Gather(lg, lth, lo.ReinterpretCast<uint32_t>(), (uint32_t)0, len);
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < cnt; ++r) {
                Muls(lf[r * k], lf[r * k], ls.GetValue(2 * r), k);
                Muls(lg[r * k], lg[r * k], ls.GetValue(2 * r + 1), k);
            }
            PipeBarrier<PIPE_V>();

            qs.FreeTensor(ls);
            yp.blockCount = cnt;

            LocalTensor<bfloat16_t> lb = qy.AllocTensor<bfloat16_t>();
            Cast(lb, lf, RoundMode::CAST_RINT, len);
            qy.EnQue(lb);
            lb = qy.DeQue<bfloat16_t>();
            DataCopyPad(gy[(uint64_t)row * 2 * k], lb, yp);
            qy.FreeTensor(lb);

            LocalTensor<bfloat16_t> lc = qy.AllocTensor<bfloat16_t>();
            Cast(lc, lg, RoundMode::CAST_RINT, len);
            qy.EnQue(lc);
            lc = qy.DeQue<bfloat16_t>();
            DataCopyPad(gy[((uint64_t)row * 2 + 1) * k], lc, yp);
            qy.FreeTensor(lc);
        }
}
}

extern "C" void w4_dequant_launch(uint32_t blockDim, void* stream, uint8_t* w,
 uint8_t* s, uint8_t* lo, uint8_t* hi, uint8_t* y, uint32_t rows, uint32_t k) {
 w4_dequant<false><<<blockDim, nullptr, stream>>>(w,s,lo,hi,y,rows,k,nullptr,1);
}
extern "C" void w4_active_launch(uint32_t blockDim, void* stream, uint8_t* w,
 uint8_t* s, uint8_t* lo, uint8_t* hi, uint8_t* y, uint32_t rows, uint32_t k,
 uint8_t* counts, uint32_t experts) {
 w4_dequant<true><<<blockDim, nullptr, stream>>>(w,s,lo,hi,y,rows,k,counts,experts);
}
