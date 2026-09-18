// Graph-safe rank-local special MoE decode leaves. No allocations or dequant cache.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "/mnt/data/kw/llama.cpp/ggml/src/ggml-common.h"

namespace {
constexpr int D = 6144;
constexpr int L = 256;
constexpr int TOPK = 8;
constexpr int NW = 8;

__device__ __forceinline__ float h2f(ggml_half v) { return __half2float(v); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &s, int &m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else {
        s = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float val_iq4xs(const block_iq4_xs &b, int i) {
    int g = i >> 5, j = i & 31;
    int ls = ((b.scales_l[g >> 1] >> (4 * (g & 1))) & 15) |
             (((b.scales_h >> (2 * g)) & 3) << 4);
    uint8_t q = b.qs[g * 16 + (j & 15)];
    int qi = j < 16 ? q & 15 : q >> 4;
    return h2f(b.d) * (ls - 32) * kvalues_iq4nl[qi];
}
__device__ __forceinline__ int q3_scale(const block_q3_K &b, int g) {
    const uint8_t *s = b.scales;
    uint32_t a0 = (uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16) | ((uint32_t)s[3] << 24);
    uint32_t a1 = (uint32_t)s[4] | ((uint32_t)s[5] << 8) | ((uint32_t)s[6] << 16) | ((uint32_t)s[7] << 24);
    uint32_t a2 = (uint32_t)s[8] | ((uint32_t)s[9] << 8) | ((uint32_t)s[10] << 16) | ((uint32_t)s[11] << 24);
    constexpr uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu;
    uint32_t o;
    switch (g >> 2) {
        case 0: o = (a0 & m2) | (((a2 >> 0) & m1) << 4); break;
        case 1: o = (a1 & m2) | (((a2 >> 2) & m1) << 4); break;
        case 2: o = ((a0 >> 4) & m2) | (((a2 >> 4) & m1) << 4); break;
        default: o = ((a1 >> 4) & m2) | (((a2 >> 6) & m1) << 4); break;
    }
    return (int)((o >> (8 * (g & 3))) & 255) - 32;
}
__device__ __forceinline__ float val_q3k(const block_q3_K &b, int i) {
    int n = i >> 7, z = i & 127, g = z >> 5, half = (z >> 4) & 1, l = z & 15;
    int qi = l + 16 * half, shift = 2 * g;
    int q = ((b.qs[n * 32 + qi] >> shift) & 3) -
            ((b.hmask[qi] & (1u << (n * 4 + g))) ? 0 : 4);
    return h2f(b.d) * q3_scale(b, n * 8 + g * 2 + half) * q;
}
__device__ __forceinline__ float val_q4k(const block_q4_K &b, int i) {
    int g = i >> 5, j = i & 31, s, m;
    scale_min_k4(g, b.scales, s, m);
    uint8_t q = b.qs[(g >> 1) * 32 + j];
    int v = (g & 1) ? q >> 4 : q & 15;
    float2 dm = __half22float2(b.dm);
    return dm.x * s * v - dm.y * m;
}
__device__ __forceinline__ float val_q5k(const block_q5_K &b, int i) {
    int g = i >> 5, j = i & 31, s, m;
    scale_min_k4(g, b.scales, s, m);
    uint8_t q = b.qs[(g >> 1) * 32 + j];
    int v = ((g & 1) ? q >> 4 : q & 15) + ((b.qh[j] & (1u << g)) ? 16 : 0);
    float2 dm = __half22float2(b.dm);
    return dm.x * s * v - dm.y * m;
}
__device__ __forceinline__ float val_q6k(const block_q6_K &b, int i) {
    int n = i >> 7, z = i & 127, g = z >> 5, l = z & 31;
    uint8_t ql = b.ql[n * 64 + (g & 1) * 32 + l];
    int lo = g < 2 ? ql & 15 : ql >> 4;
    int hi = (b.qh[n * 32 + l] >> (2 * g)) & 3;
    int sc = n * 8 + 2 * g + (l >> 4);
    return h2f(b.d) * (int)b.scales[sc] * (lo + (hi << 4) - 32);
}

template<int G> __device__ __forceinline__ float gu_dot(const uint8_t *rp, const float *x);
template<> __device__ __forceinline__ float gu_dot<0>(const uint8_t *rp, const float *x) {
    float s = 0; int lane = threadIdx.x & 31;
    for (int kb = 0; kb < D / 256; ++kb) {
        const block_iq4_xs &b = ((const block_iq4_xs *)rp)[kb];
#pragma unroll
        for (int j = 0; j < 8; ++j) { int i = lane + 32 * j; s += val_iq4xs(b, i) * x[kb * 256 + i]; }
    }
    return warp_sum(s);
}
template<> __device__ __forceinline__ float gu_dot<1>(const uint8_t *rp, const float *x) {
    float s = 0; int lane = threadIdx.x & 31, il = lane / 8, ib = lane % 8;
    for (int kb = 0; kb < D / 256; ++kb) {
        const block_iq3_xxs &q = ((const block_iq3_xxs *)rp)[kb];
        const uint8_t *q3 = q.qs + 8 * ib;
        const uint16_t *gas = (const uint16_t *)(q.qs + 64) + 2 * ib;
        const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + q3[2 * il]);
        const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 1]);
        uint32_t a = (uint32_t)gas[0] | ((uint32_t)gas[1] << 16);
        float d = h2f(q.d) * (0.5f + (a >> 28)) * 0.5f;
        uint8_t sign = ksigns_iq2xs[(a >> (7 * il)) & 127];
        int off = kb * 256 + 32 * ib + 8 * il;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            s += d * (g1[j] * ((sign & (1u << j)) ? -1.f : 1.f) * x[off + j] +
                      g2[j] * ((sign & (1u << (j + 4))) ? -1.f : 1.f) * x[off + j + 4]);
    }
    return warp_sum(s);
}
template<> __device__ __forceinline__ float gu_dot<2>(const uint8_t *rp, const float *x) {
    float s = 0; int lane = threadIdx.x & 31;
    for (int kb = 0; kb < D / 256; ++kb) {
        const block_q3_K &b = ((const block_q3_K *)rp)[kb];
#pragma unroll
        for (int j = 0; j < 8; ++j) { int i = lane + 32 * j; s += val_q3k(b, i) * x[kb * 256 + i]; }
    }
    return warp_sum(s);
}

template<int T> __device__ __forceinline__ float down_dot(const uint8_t *rp, const float *h);
template<> __device__ __forceinline__ float down_dot<0>(const uint8_t *rp, const float *h) {
    const block_q5_K &b = *(const block_q5_K *)rp; float s = 0; int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 8; ++j) { int i = lane + 32 * j; s += val_q5k(b, i) * h[i]; }
    return warp_sum(s);
}
template<> __device__ __forceinline__ float down_dot<1>(const uint8_t *rp, const float *h) {
    const block_q6_K &b = *(const block_q6_K *)rp; float s = 0; int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 8; ++j) { int i = lane + 32 * j; s += val_q6k(b, i) * h[i]; }
    return warp_sum(s);
}
template<> __device__ __forceinline__ float down_dot<2>(const uint8_t *rp, const float *h) {
    const block_q4_K &b = *(const block_q4_K *)rp; float s = 0; int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 8; ++j) { int i = lane + 32 * j; s += val_q4k(b, i) * h[i]; }
    return warp_sum(s);
}

constexpr int MAX_Q = 127;
constexpr int GU_TILES = 4;
constexpr int DOWN_TILES = 64;
__device__ float special_h[MAX_Q * TOPK * L];
constexpr int Q3_EXACT_TILES = 16;
__device__ float special_q3_gu_branch[MAX_Q * TOPK * L * 2];

// ---- dp4a path: quantize x to q8 (per-32 scale) once per call ----
__device__ int8_t special_xq[MAX_Q * D];
__device__ float special_xds[MAX_Q * D / 32];
__global__ void special_x_q8_kernel(const float *x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lane = threadIdx.x & 31;
    float v = x[i];
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float scale = a > 0.f ? a / 127.f : 1.f;
    special_xq[i] = (int8_t)__float2int_rn(v / scale);
    if (lane == 0) special_xds[i / 32] = scale;
}

// iq3_xxs gate/up via integer dp4a (mirrors v12 moe_decode_gu_iq3_dp4a_kernel).
__global__ void special_gu_iq3_dp4a_kernel(const uint8_t *g, const uint8_t *u,
                                           const int64_t *eid, int Q, int grb) {
    int task = blockIdx.x, tile = blockIdx.y;
    int q = task / TOPK, k = task - q * TOPK;
    if (q >= Q) return;
    int e = (int)eid[(int64_t)q * TOPK + k];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int il = lane / 8, ib = lane % 8;
    const int8_t *xb = special_xq + (int64_t)q * D;
    const float *xs = special_xds + (int64_t)q * (D / 32);
    float *hh = special_h + ((int64_t)q * TOPK + k) * L;
    for (int row = tile * NW + warp; row < L; row += GU_TILES * NW) {
        const uint8_t *rg = g + ((int64_t)e * L + row) * grb;
        const uint8_t *ru = u + ((int64_t)e * L + row) * grb;
        float sg = 0.f, su = 0.f;
#pragma unroll 2
        for (int kb = 0; kb < D / 256; ++kb) {
            const block_iq3_xxs &qg = ((const block_iq3_xxs *)rg)[kb];
            const block_iq3_xxs &qu = ((const block_iq3_xxs *)ru)[kb];
            const uint8_t *q3g = qg.qs + 8 * ib;
            const uint16_t *gasg = (const uint16_t *)(qg.qs + 64) + 2 * ib;
            const uint8_t *q3u = qu.qs + 8 * ib;
            const uint16_t *gasu = (const uint16_t *)(qu.qs + 64) + 2 * ib;
            uint16_t q3gp = *reinterpret_cast<const uint16_t *>(q3g + 2 * il);
            uint16_t q3up = *reinterpret_cast<const uint16_t *>(q3u + 2 * il);
            uint32_t ag = (uint32_t)gasg[0] | ((uint32_t)gasg[1] << 16);
            uint32_t au = (uint32_t)gasu[0] | ((uint32_t)gasu[1] << 16);
            const uint32_t *sgn_g = (const uint32_t *)(ksigns64 + ((ag >> (7 * il)) & 127));
            const uint32_t *sgn_u = (const uint32_t *)(ksigns64 + ((au >> (7 * il)) & 127));
            int g1 = __vsub4(iq3xxs_grid[q3gp & 255u] ^ sgn_g[0], sgn_g[0]);
            int g2 = __vsub4(iq3xxs_grid[q3gp >> 8] ^ sgn_g[1], sgn_g[1]);
            int u1 = __vsub4(iq3xxs_grid[q3up & 255u] ^ sgn_u[0], sgn_u[0]);
            int u2 = __vsub4(iq3xxs_grid[q3up >> 8] ^ sgn_u[1], sgn_u[1]);
            int off = kb * 256 + 32 * ib + 8 * il;
            int x1 = *(const int *)(xb + off);
            int x2 = *(const int *)(xb + off + 4);
            int dig = __dp4a(g1, x1, __dp4a(g2, x2, 0));
            int diu = __dp4a(u1, x1, __dp4a(u2, x2, 0));
            float xsc = xs[kb * 8 + ib];
            sg += h2f(qg.d) * (0.5f + (ag >> 28)) * 0.5f * xsc * (float)dig;
            su += h2f(qu.d) * (0.5f + (au >> 28)) * 0.5f * xsc * (float)diu;
        }
        sg = warp_sum(sg); su = warp_sum(su);
        if (lane == 0) hh[row] = (sg / (1.f + expf(-sg))) * su;
    }
}

// Exact Q3 Gate/Up parallelization: each output row keeps the legacy 24-K-block
// lane accumulation and warp reduction order. Only independent Gate/Up
// branches and rows are scheduled concurrently.
__global__ void special_q3_gu_branch_kernel(
        const uint8_t *g, const uint8_t *u, const float *x,
        const int64_t *eid, int Q, int grb) {
    int task = blockIdx.x, tile = blockIdx.y, branch = blockIdx.z;
    int q = task / TOPK, k = task - q * TOPK;
    if (q >= Q) return;
    int e = (int)eid[(int64_t)q * TOPK + k];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t *base = branch ? u : g;
    const float *xx = x + (int64_t)q * D;
    for (int row = tile * NW + warp; row < L; row += Q3_EXACT_TILES * NW) {
        const uint8_t *rp = base + ((int64_t)e * L + row) * grb;
        float sum = gu_dot<2>(rp, xx);
        if (lane == 0)
            special_q3_gu_branch[((int64_t)task * L + row) * 2 + branch] = sum;
    }
}

__global__ void special_q3_gu_branch_finalize_kernel(int Q) {
    int task = blockIdx.x, row = threadIdx.x;
    if (task >= Q * TOPK || row >= L) return;
    int64_t off = ((int64_t)task * L + row) * 2;
    float sg = special_q3_gu_branch[off];
    float su = special_q3_gu_branch[off + 1];
    special_h[(int64_t)task * L + row] = (sg / (1.f + expf(-sg))) * su;
}

template<int GT>
__global__ void special_gu_kernel(const uint8_t *g, const uint8_t *u,
                                  const float *x, const int64_t *eid,
                                  int Q, int grb) {
    int task = blockIdx.x, tile = blockIdx.y;
    int q = task / TOPK, k = task - q * TOPK;
    if (q >= Q) return;
    int e = (int)eid[(int64_t)q * TOPK + k];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float *xx = x + (int64_t)q * D;
    float *hh = special_h + ((int64_t)q * TOPK + k) * L;
    for (int row = tile * NW + warp; row < L; row += GU_TILES * NW) {
        const uint8_t *rg = g + ((int64_t)e * L + row) * grb;
        const uint8_t *ru = u + ((int64_t)e * L + row) * grb;
        float sg = gu_dot<GT>(rg, xx), su = gu_dot<GT>(ru, xx);
        if (lane == 0) hh[row] = (sg / (1.f + expf(-sg))) * su;
    }
}

template<int DT>
__global__ void special_down_kernel(const uint8_t *d, const int64_t *eid,
                                    const float *ew, float *out,
                                    int Q, int drb) {
    int q = blockIdx.x, tile = blockIdx.y;
    if (q >= Q) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int row = tile * NW + warp; row < D; row += DOWN_TILES * NW) {
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            int e = (int)eid[(int64_t)q * TOPK + k];
            const uint8_t *rd = d + ((int64_t)e * D + row) * drb;
            const float *hh = special_h + ((int64_t)q * TOPK + k) * L;
            sum += ew[(int64_t)q * TOPK + k] * down_dot<DT>(rd, hh);
        }
        if (lane == 0) out[(int64_t)q * D + row] = sum;
    }
}

void check(torch::Tensor g, torch::Tensor u, torch::Tensor d, torch::Tensor x,
           torch::Tensor ei, torch::Tensor ew, torch::Tensor out, int grb, int drb) {
    TORCH_CHECK(g.is_cuda() && u.is_cuda() && d.is_cuda() && x.is_cuda() && ei.is_cuda() && ew.is_cuda() && out.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(g.scalar_type() == torch::kUInt8 && u.scalar_type() == torch::kUInt8 && d.scalar_type() == torch::kUInt8, "packed weights must be uint8");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && ew.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32 && ei.scalar_type() == torch::kInt64, "x/ew/out fp32; eid int64");
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && d.is_contiguous() && x.is_contiguous() && ei.is_contiguous() && ew.is_contiguous() && out.is_contiguous(), "all tensors contiguous");
    int64_t Q = x.size(0);
    TORCH_CHECK(Q >= 1 && Q <= MAX_Q, "special decode expects flattened Q=1..127");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == D && ei.dim() == 2 && ei.size(0) == Q && ei.size(1) == TOPK && ew.sizes() == ei.sizes() && out.sizes() == x.sizes(), "shape contract violated");
    TORCH_CHECK(g.dim() == 3 && g.size(0) == 256 && g.size(1) == L && g.size(2) == grb && u.sizes() == g.sizes(), "gate/up packed shape");
    TORCH_CHECK(d.dim() == 3 && d.size(0) == 256 && d.size(1) == D && d.size(2) == drb, "down packed shape");
    TORCH_CHECK(g.device() == x.device() && u.device() == x.device() && d.device() == x.device() && ei.device() == x.device() && ew.device() == x.device() && out.device() == x.device(), "same device required");
}
// MTP layer 78 uses Q3_K Gate/Up and Q4_K Down. At B1 draft widths the
// legacy Down kernel assigns one warp per output and serially scans all eight
// selected experts. Spread those independent expert dots across the eight
// warps in a block, then preserve the original k=0..7 reduction order.
constexpr int MTP_DOWN_BLOCKS = 1024;
template<int DT>
__global__ void special_down_expert_parallel(const uint8_t *d,
                                             const int64_t *eid,
                                             const float *ew, float *out,
                                             int Q, int drb) {
    int q = blockIdx.x;
    if (q >= Q) return;
    int k = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    __shared__ float dots[TOPK];
    for (int row = blockIdx.y; row < D; row += MTP_DOWN_BLOCKS) {
        int e = (int)eid[(int64_t)q * TOPK + k];
        const uint8_t *rd = d + ((int64_t)e * D + row) * drb;
        const float *hh = special_h + ((int64_t)q * TOPK + k) * L;
        float dot = down_dot<DT>(rd, hh);
        if (lane == 0) dots[k] = dot;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < TOPK; ++j)
                sum += ew[(int64_t)q * TOPK + j] * dots[j];
            out[(int64_t)q * D + row] = sum;
        }
        __syncthreads();
    }
}

template<int GT, int DT>
void launch(torch::Tensor g, torch::Tensor u, torch::Tensor d, torch::Tensor x,
            torch::Tensor ei, torch::Tensor ew, torch::Tensor out, int grb, int drb) {
    check(g, u, d, x, ei, ew, out, grb, drb);
    c10::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    int Q = x.size(0);
    special_gu_kernel<GT><<<dim3(Q * TOPK, GU_TILES), NW * 32, 0, st>>>(
        g.data_ptr<uint8_t>(), u.data_ptr<uint8_t>(), x.data_ptr<float>(),
        ei.data_ptr<int64_t>(), Q, grb);
    special_down_kernel<DT><<<dim3(Q, DOWN_TILES), NW * 32, 0, st>>>(
        d.data_ptr<uint8_t>(), ei.data_ptr<int64_t>(), ew.data_ptr<float>(),
        out.data_ptr<float>(), Q, drb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void special_iq4xs_q5k(torch::Tensor g, torch::Tensor u, torch::Tensor d, torch::Tensor x, torch::Tensor ei, torch::Tensor ew, torch::Tensor out) { launch<0, 0>(g, u, d, x, ei, ew, out, 3264, 176); }
void special_iq3xxs_q6k(torch::Tensor g, torch::Tensor u, torch::Tensor d, torch::Tensor x, torch::Tensor ei, torch::Tensor ew, torch::Tensor out) {
    constexpr int grb = 2352, drb = 210;
    check(g, u, d, x, ei, ew, out, grb, drb);
    c10::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    int Q = x.size(0);
    int64_t n = (int64_t)Q * D;
    special_x_q8_kernel<<<(int)((n + 255) / 256), 256, 0, st>>>(x.data_ptr<float>(), n);
    special_gu_iq3_dp4a_kernel<<<dim3(Q * TOPK, GU_TILES), NW * 32, 0, st>>>(
        g.data_ptr<uint8_t>(), u.data_ptr<uint8_t>(), ei.data_ptr<int64_t>(), Q, grb);
    special_down_expert_parallel<1><<<dim3(Q, MTP_DOWN_BLOCKS), TOPK * 32, 0, st>>>(
        d.data_ptr<uint8_t>(), ei.data_ptr<int64_t>(), ew.data_ptr<float>(),
        out.data_ptr<float>(), Q, drb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void special_q3k_q4k(torch::Tensor g, torch::Tensor u, torch::Tensor d,
                               torch::Tensor x, torch::Tensor ei, torch::Tensor ew,
                               torch::Tensor out) {
    constexpr int grb = 2640, drb = 144;
    check(g, u, d, x, ei, ew, out, grb, drb);
    c10::cuda::CUDAGuard guard(x.device());
    auto st = at::cuda::getCurrentCUDAStream();
    int Q = x.size(0);
    special_q3_gu_branch_kernel<<<dim3(Q * TOPK, Q3_EXACT_TILES, 2),
                                         NW * 32, 0, st>>>(
        g.data_ptr<uint8_t>(), u.data_ptr<uint8_t>(), x.data_ptr<float>(),
        ei.data_ptr<int64_t>(), Q, grb);
    special_q3_gu_branch_finalize_kernel<<<Q * TOPK, L, 0, st>>>(Q);
    special_down_expert_parallel<2><<<dim3(Q, MTP_DOWN_BLOCKS),
                                       TOPK * 32, 0, st>>>(
        d.data_ptr<uint8_t>(), ei.data_ptr<int64_t>(), ew.data_ptr<float>(),
        out.data_ptr<float>(), Q, drb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("special_iq4xs_q5k", &special_iq4xs_q5k);
    m.def("special_iq3xxs_q6k", &special_iq3xxs_q6k);
    m.def("special_q3k_q4k", &special_q3k_q4k);
}
