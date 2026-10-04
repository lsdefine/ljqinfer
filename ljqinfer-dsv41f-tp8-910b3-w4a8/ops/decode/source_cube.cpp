// Cube-resident indexer scoring: one GEMM per committed KV page.
//
// The paged indexer bank stores `rpp` contiguous 128-wide BF16 key rows per
// physical page, so a whole page is a contiguous (N=rpp, K=128) transposed B
// operand. The six MTP rows of a batch entry share that operand, and the four
// indexer heads are already laid out contiguously behind each row, so the
// packed queries form a single (M=24, K=128) A operand. One IterateAll per
// page therefore produces every head logit the vector core needs.
//
// Output is blocked per page -- logits[b][page][24][rpp] -- which keeps the C
// row stride equal to rpp and lets the vector pass read each 64-key tile as
// four contiguous runs. Pages are scored in full even when the last one runs
// past `origin`: the surplus columns land inside the same page allocation and
// the vector pass never reads them.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "include/adv_api/matmul/matmul_intf.h"
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// Raw scalar GM reads land in the core data cache; the page table and the
// cursors are rewritten by other kernels every step, so invalidate on entry.
__aicore__ inline void cinv(GM_ADDR p) {
    GlobalTensor<uint8_t> g; g.SetGlobalBuffer(p);
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(g);
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void decode_source_cube_kernel(
    GM_ADDR q, GM_ADDR bank, GM_ADDR table, GM_ADDR slots, GM_ADDR start,
    GM_ADDR active, GM_ADDR logits, GM_ADDR tiling,
    uint32_t batch, uint32_t ratio, uint32_t nslots, uint32_t pages,
    uint32_t maxpages, uint32_t rpp, uint32_t npmax, uint32_t cores) {
    const uint32_t core = GetBlockIdx();
    if (core >= cores) return;
    cinv(table); cinv(slots); cinv(start); cinv(active);
    auto pt = reinterpret_cast<const __gm__ int64_t*>(table);
    auto sl = reinterpret_cast<const __gm__ int64_t*>(slots);
    auto st = reinterpret_cast<const __gm__ int64_t*>(start);
    auto ac = reinterpret_cast<const __gm__ int64_t*>(active);
    TPipe pipe;
    Matmul<AType, BType, CType, CType, CFG_MDL> mm;
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q), uint64_t(batch) * 24 * 128);
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(bank), uint64_t(pages) * rpp * 128);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(logits), uint64_t(batch) * npmax * 24 * rpp);
    AscendC::tiling::TCubeTiling localTiling;
    static_assert(sizeof(localTiling) == 200, "host/device tiling ABI mismatch");
    auto dst = reinterpret_cast<uint32_t*>(&localTiling);
    auto src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    for (uint32_t i = 0; i < sizeof(localTiling) / sizeof(uint32_t); ++i) dst[i] = src[i];
    mm.Init(&localTiling, &pipe);
    mm.SetOrgShape(24, rpp, 128);
    mm.SetSingleShape(24, rpp, 128);
    for (uint32_t job = core; job < batch * npmax; job += cores) {
        const uint32_t bi = job / npmax, lp = job % npmax;
        const int64_t slot = sl[bi], base = st[bi];
        if (!ac[bi] || slot < 0 || slot >= int64_t(nslots) || base < 0) continue;
        if (int64_t(lp) * rpp >= base / int64_t(ratio)) continue;
        const int64_t page = pt[slot * int64_t(maxpages) + int64_t(lp)];
        if (page < 0 || page >= int64_t(pages)) continue;
        mm.SetTensorA(a[uint64_t(bi) * 24 * 128], false);
        mm.SetTensorB(b[uint64_t(page) * rpp * 128], true);
        mm.IterateAll(c[(uint64_t(bi) * npmax + lp) * 24 * rpp], 0, false);
    }
    mm.End();
}

extern "C" int dec_source_cube(void* stream, void* q, void* bank, void* table,
        void* slots, void* start, void* active, void* logits, void* tiling,
        uint32_t batch, uint32_t ratio, uint32_t nslots, uint32_t pages,
        uint32_t maxpages, uint32_t rpp, uint32_t npmax) {
    if (!q || !bank || !table || !slots || !start || !active || !logits || !tiling) return -1;
    if (!batch || batch > 64 || !ratio || !nslots || !pages || !maxpages || !npmax) return -2;
    if (rpp != 1024 && rpp != 2048) return -3;
    const uint32_t cores = 20;
    decode_source_cube_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)q, (uint8_t*)bank, (uint8_t*)table, (uint8_t*)slots,
        (uint8_t*)start, (uint8_t*)active, (uint8_t*)logits, (uint8_t*)tiling,
        batch, ratio, nslots, pages, maxpages, rpp, npmax, cores);
    return 0;
}
