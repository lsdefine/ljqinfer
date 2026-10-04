// Isolated draft shared-down K288 candidate; not yet device-validated.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "include/adv_api/matmul/matmul_intf.h"
#include <cstdint>
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
extern "C" __global__ __aicore__ void decode_draft_down_cube_kernel(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR tiling, uint32_t m) {
    const uint32_t core = GetBlockIdx();
    if (core >= 20) return;
    TPipe pipe;
    Matmul<AType, BType, CType, CType, CFG_MDL> mm;
    GlobalTensor<bfloat16_t> a, b;
    GlobalTensor<float> c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), m * 288);
    b.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w), 5120 * 288);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), m * 5120);
    AscendC::tiling::TCubeTiling localTiling;
    static_assert(sizeof(localTiling) == 200, "host/device tiling ABI mismatch");
    auto dst = reinterpret_cast<uint32_t*>(&localTiling);
    auto src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    for (uint32_t i=0; i<sizeof(localTiling)/sizeof(uint32_t); ++i) dst[i]=src[i];
    mm.Init(&localTiling, &pipe);
    mm.SetOrgShape(m, 5120, 288);
    mm.SetSingleShape(m, 256, 288);
    mm.SetTail(m, 256, 288);
    mm.SetTensorA(a, false);
    mm.SetTensorB(b[core * 256 * 288], true);
    mm.IterateAll(c[core * 256], 0, false);
    mm.End();
}
static bool overlap(uintptr_t a, uint64_t na, uintptr_t b, uint64_t nb) {
    return a <= b ? b-a < na : a-b < nb;
}
extern "C" int dec_draft_down_cube(void* stream, void* x, void* w, void* y,
                             void* tiling, uint32_t m) {
    if (!x || !w || !y || !tiling || (m!=6 && m!=12 && m!=18 && m!=24)) return -1;
    uintptr_t a=reinterpret_cast<uintptr_t>(x),b=reinterpret_cast<uintptr_t>(w),
              c=reinterpret_cast<uintptr_t>(y),t=reinterpret_cast<uintptr_t>(tiling);
    if ((a|b|c|t)&31) return -2;
    if (overlap(c,uint64_t(m)*5120*4,a,uint64_t(m)*288*2) ||
        overlap(c,uint64_t(m)*5120*4,b,uint64_t(5120)*288*2) ||
        overlap(c,uint64_t(m)*5120*4,t,200)) return -3;
    // Caller owns matching tiling and all tensors; current stream passed each call.
    decode_draft_down_cube_kernel<<<20,nullptr,stream>>>(
        (uint8_t*)x,(uint8_t*)w,(uint8_t*)y,(uint8_t*)tiling,m);
    return 0;
}
