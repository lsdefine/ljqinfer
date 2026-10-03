#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <unordered_map>

// GLM-5.2 absorbed MLA attention for SM80.
// One CUDA block owns one (query, head), streams causal K once, and maintains
// the softmax numerator/denominator online. No [H,Q,K] score tensor exists.
constexpr int L = 512;
constexpr int R = 64;
constexpr int THREADS = 256;
constexpr float SCALE = 0.0625f;

__device__ __forceinline__ const half* paged_kv_row(
        const half* pool, const int64_t* table, int page_size, int logical_row) {
    const int logical_page = logical_row / page_size;
    const int offset = logical_row - logical_page * page_size;
    const int64_t physical_page = table[logical_page];
    return pool + ((size_t)physical_page * page_size + offset) * (L + R);
}

__inline__ __device__ float warp_sum(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

__inline__ __device__ float block_sum(float x, float *warp_buf) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) warp_buf[wid] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? warp_buf[lane] : 0.0f;
    if (wid == 0) x = warp_sum(x);
    if (threadIdx.x == 0) warp_buf[0] = x;
    __syncthreads();
    return warp_buf[0];
}

__global__ void flash_mla_sm80_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, half *__restrict__ out,
        int nq, int nk, int q_start, int nh) {
    const int h = blockIdx.x;
    const int tq = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= nh || tq >= nq) return;
    const int kend = min(nk, q_start + tq + 1);
    const half *q_lat = ql + ((size_t)tq * nh + h) * L;
    const half *q_rot = qr + ((size_t)tq * nh + h) * R;

    __shared__ float red[8];
    __shared__ float coeff[2]; // alpha=old numerator scale, beta=new value scale
    float acc0 = 0.0f, acc1 = 0.0f;
    float m = -FLT_MAX, l = 0.0f;

    for (int k = 0; k < kend; ++k) {
        const half *ck = cache + (size_t)k * (L + R);
        float dot = __half2float(q_lat[tid]) * __half2float(ck[tid]);
        dot += __half2float(q_lat[tid + THREADS]) * __half2float(ck[tid + THREADS]);
        if (tid < R) dot += __half2float(q_rot[tid]) * __half2float(ck[L + tid]);
        const float s = block_sum(dot, red) * SCALE;
        if (tid == 0) {
            const float m2 = fmaxf(m, s);
            const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
            const float beta = __expf(s - m2);
            l = l * alpha + beta;
            m = m2;
            coeff[0] = alpha;
            coeff[1] = beta;
        }
        __syncthreads();
        acc0 = acc0 * coeff[0] + coeff[1] * __half2float(ck[tid]);
        acc1 = acc1 * coeff[0] + coeff[1] * __half2float(ck[tid + THREADS]);
        __syncthreads();
    }
    // m/l are block-uniform logically, but only thread 0 updated its private l.
    if (tid == 0) red[0] = l;
    __syncthreads();
    half *dst = out + ((size_t)tq * nh + h) * L;
    dst[tid] = __float2half_rn(acc0 / red[0]);
    dst[tid + THREADS] = __float2half_rn(acc1 / red[0]);
}


// Fast path: eight warps independently scan interleaved keys. Each warp keeps
// a private online-softmax numerator in registers; only final states are merged.
__global__ void flash_mla_sm80_warp_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, half *__restrict__ out,
        int nq, int nk, int q_start, int nh) {
    const int h = blockIdx.x, tq = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int kend = min(nk, q_start + tq + 1);
    const half *q_lat = ql + ((size_t)tq * nh + h) * L;
    const half *q_rot = qr + ((size_t)tq * nh + h) * R;

    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32*j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;

    for (int k = wid; k < kend; k += 8) {
        // C1_FUSE_CK: load latent V once; __ldg for RO cache
        const half *ck = cache + (size_t)k * (L + R);
        float vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            vals[j] = __half2float(__ldg(ck + lane + 32 * j));
        float dot = qr0 * __half2float(__ldg(ck + L + lane)) +
                    qr1 * __half2float(__ldg(ck + L + lane + 32));
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * vals[j];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * vals[j];
    }

    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32*j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        float gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    half *dst = out + ((size_t)tq * nh + h) * L;
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        dst[d] = __float2half_rn(v / total_l);
    }
}


// Decode T=1: one warp per head, no multi-warp merge tax (critical for small K).
__global__ void flash_mla_sm80_t1_swarp_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, half *__restrict__ out,
        int nk, int q_start, int nh, const int* __restrict__ k0_ptr) {
    const int h = blockIdx.x;
    if (h >= nh) return;
    const int lane = threadIdx.x;
    if (k0_ptr) q_start = *k0_ptr;
    const int kend = min(nk, q_start + 1);
    const half *q_lat = ql + (size_t)h * L;
    const half *q_rot = qr + (size_t)h * R;
    float qv[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) qv[j] = __half2float(q_lat[lane + 32 * j]);
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    float m = -FLT_MAX, l = 0.0f;
    for (int k = 0; k < kend; ++k) {
        const half *ck = cache + (size_t)k * (L + R);
        float dot = qr0 * __half2float(ck[L + lane]) + qr1 * __half2float(ck[L + lane + 32]);
#pragma unroll
        for (int j = 0; j < 16; ++j) dot += qv[j] * __half2float(ck[lane + 32 * j]);
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * __half2float(ck[lane + 32 * j]);
    }
    half *dst = out + (size_t)h * L;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < 16; ++j)
        dst[lane + 32 * j] = __float2half_rn(acc[j] * inv);
}

// Decode T=1 multi-warp + optional device k0 (graph-safe).
__global__ void flash_mla_sm80_t1_mwarp_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, const int64_t* __restrict__ page_table, int page_size, half *__restrict__ out,
        int nk, int q_start, int nh, const int* __restrict__ k0_ptr) {
    const int h = blockIdx.x;
    if (h >= nh) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    if (k0_ptr) q_start = *k0_ptr;
    const int kend = min(nk, q_start + 1);
    const half *q_lat = ql + (size_t)h * L;
    const half *q_rot = qr + (size_t)h * R;

    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32 * j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;

    for (int k = wid; k < kend; k += 8) {
        const half *ck = paged_kv_row(cache, page_table, page_size, k);
        float dot = qr0 * __half2float(ck[L + lane]) +
                    qr1 * __half2float(ck[L + lane + 32]);
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * __half2float(ck[lane + 32 * j]);
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * __half2float(ck[lane + 32 * j]);
    }

    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32 * j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        float gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    half *dst = out + (size_t)h * L;
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        dst[d] = __float2half_rn(v / total_l);
    }
}




// Decode T=1 dispatch: single-warp only for short K.
// Measured H=8: K4k t1=2.78ms vs warp8~0.37ms; K32k t1=30ms vs warp8~3.1ms.
#ifndef FLASH_MLA_T1_SWARP_MAX_K
#define FLASH_MLA_T1_SWARP_MAX_K 256
#endif

static inline bool flash_mla_use_t1_swarp(int nq, int q_start) {
    return nq == 1 && (q_start + 1) <= FLASH_MLA_T1_SWARP_MAX_K;
}


// Multi-CTA split-K: each block scans a contiguous K segment; host merges.
// Workspace layout per (q,h): n_split * (2 + L) floats: [m, l, o0..o511]
#ifndef FLASH_MLA_SPLITK_MIN_K
#define FLASH_MLA_SPLITK_MIN_K 2048
#endif
#ifndef FLASH_MLA_SPLITK_TARGET_SPLITS
#define FLASH_MLA_SPLITK_TARGET_SPLITS 32
#endif


// Source-controlled adaptive split count from the A100 single-rank decode sweep.
// Runtime environment overrides are deliberately forbidden.
static inline int flash_mla_pick_nsplit(int kend) {
    // Adaptive from A100 single-rank decode sweep
    // Long-K: nsplit=40 wins at 16k/32k/48k (was stuck at 32 until 49152)
    int n = 8;
    if (kend >= 4096) n = 24;
    if (kend >= 8192) n = 32;
    if (kend >= 16384) n = 40;
    if (kend >= 32768) n = 40;
    if (kend >= 49152) n = 40;
    if (kend >= 65536) n = 48;
    if (n > kend) n = kend;
    if (n < 1) n = 1;
    return n;
}

__global__ void flash_mla_sm80_splitk_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, float *__restrict__ ws,
        int nq, int nk, int q_start, int nh, int n_split, int stride_qh) {
    const int h = blockIdx.x, tq = blockIdx.y, sid = blockIdx.z;
    if (h >= nh || tq >= nq || sid >= n_split) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int kend = min(nk, q_start + tq + 1);
    if (kend <= 0) return;
    const int chunk = (kend + n_split - 1) / n_split;
    const int k0 = sid * chunk;
    const int k1 = min(kend, k0 + chunk);
    if (k0 >= k1) {
        // empty split: m=-inf, l=0, o=0
        float *slot = ws + ((size_t)tq * nh + h) * stride_qh + (size_t)sid * (2 + L);
        if (tid == 0) { slot[0] = -FLT_MAX; slot[1] = 0.0f; }
        for (int d = tid; d < L; d += THREADS) slot[2 + d] = 0.0f;
        return;
    }
    const half *q_lat = ql + ((size_t)tq * nh + h) * L;
    const half *q_rot = qr + ((size_t)tq * nh + h) * R;
    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32 * j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;
    for (int k = k0 + wid; k < k1; k += 8) {
        // C1_FUSE_CK Q>1: load latent V once; __ldg RO
        const half *ck = cache + (size_t)k * (L + R);
        float vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            vals[j] = __half2float(__ldg(ck + lane + 32 * j));
        float dot = qr0 * __half2float(__ldg(ck + L + lane)) +
                    qr1 * __half2float(__ldg(ck + L + lane + 32));
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * vals[j];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * vals[j];
    }
    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l, gm;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32 * j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    float *slot = ws + ((size_t)tq * nh + h) * stride_qh + (size_t)sid * (2 + L);
    if (tid == 0) { slot[0] = gm; slot[1] = total_l; }
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        slot[2 + d] = v; // unnormalized numerator at local m=gm
    }
}

__global__ void flash_mla_sm80_combine_kernel(
        const float *__restrict__ ws, half *__restrict__ out,
        int nq, int nh, int n_split, int stride_qh) {
    const int h = blockIdx.x, tq = blockIdx.y;
    if (h >= nh || tq >= nq) return;
    const int tid = threadIdx.x;
    const float *base = ws + ((size_t)tq * nh + h) * stride_qh;
    // pass1: global m
    float gm = -FLT_MAX;
    for (int s = 0; s < n_split; ++s) {
        const float mi = base[(size_t)s * (2 + L)];
        gm = fmaxf(gm, mi);
    }
    // pass2: total_l and weighted o (tid-parallel over L)
    // each thread accumulates its dims; total_l computed by tid0
    __shared__ float sh_l;
    if (tid == 0) {
        float gl = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float mi = base[(size_t)s * (2 + L)];
            const float li = base[(size_t)s * (2 + L) + 1];
            gl += (li == 0.0f) ? 0.0f : li * __expf(mi - gm);
        }
        sh_l = gl;
    }
    __syncthreads();
    half *dst = out + ((size_t)tq * nh + h) * L;
    const float inv = 1.0f / sh_l;
    for (int d = tid; d < L; d += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float mi = base[(size_t)s * (2 + L)];
            const float li = base[(size_t)s * (2 + L) + 1];
            if (li == 0.0f) continue;
            const float alpha = __expf(mi - gm);
            v += base[(size_t)s * (2 + L) + 2 + d] * alpha;
        }
        dst[d] = __float2half_rn(v * inv);
    }
}


// Persistent split-K workspace (graph-safe: no alloc on hot path after first size).
static torch::Tensor& flash_mla_ws_get(const torch::Device& dev, int64_t n_elem) {
    static std::mutex mu;
    static std::unordered_map<int, torch::Tensor> cache;
    const int di = dev.index();
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(di);
    if (it == cache.end() || it->second.numel() < n_elem) {
        auto t = torch::empty({n_elem}, torch::TensorOptions().dtype(at::kFloat).device(dev));
        cache[di] = t;
        return cache[di];
    }
    return it->second;
}


// Historical filename/API name notwithstanding, flash_mla_sm80* is a
// DECODE-ONLY production implementation.  Model prefill must never call it:
// use tc_mla for identity pages or paged_prefill_mla for non-identity pages.
// This contiguous overload remains only for legacy direct-ABI benchmarks.
torch::Tensor flash_mla_sm80(torch::Tensor q_latent, torch::Tensor q_rope,
                             torch::Tensor cache, int64_t q_start) {
    TORCH_CHECK(q_latent.is_cuda() && q_rope.is_cuda() && cache.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(q_latent.scalar_type() == at::kHalf && q_rope.scalar_type() == at::kHalf && cache.scalar_type() == at::kHalf, "fp16 only");
    TORCH_CHECK(q_latent.is_contiguous() && q_rope.is_contiguous() && cache.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(q_latent.dim() == 3 && q_latent.size(1) > 0 && q_latent.size(1) <= 64 && q_latent.size(2) == L, "q_latent must be [Q,H,512], 1 <= H <= 64");
    TORCH_CHECK(q_rope.sizes() == torch::IntArrayRef({q_latent.size(0), q_latent.size(1), R}), "q_rope must be [Q,H,64]");
    TORCH_CHECK(cache.dim() == 2 && cache.size(1) == L + R, "cache must be [K,576]");
    TORCH_CHECK(q_start >= 0 && q_start < cache.size(0), "invalid q_start");
    TORCH_CHECK(q_start + q_latent.size(0) <= cache.size(0), "queries exceed cache");
    c10::cuda::CUDAGuard guard(q_latent.device());
    auto out = torch::empty_like(q_latent);
    const int nh = (int)q_latent.size(1);
    const int nq = (int)q_latent.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half *ql = reinterpret_cast<half *>(q_latent.data_ptr<at::Half>());
    const half *qrp = reinterpret_cast<half *>(q_rope.data_ptr<at::Half>());
    const half *cp = reinterpret_cast<half *>(cache.data_ptr<at::Half>());
    half *op = reinterpret_cast<half *>(out.data_ptr<at::Half>());
    const int nk = (int)cache.size(0);
    // effective max keys for last query
    const int kend = (int)std::min<int64_t>(nk, q_start + nq);

    if (flash_mla_use_t1_swarp(nq, (int)q_start)) {
      flash_mla_sm80_t1_swarp_kernel<<<nh, 32, 0, stream>>>(
        ql, qrp, cp, op, nk, (int)q_start, nh, nullptr);
    } else if (kend >= FLASH_MLA_SPLITK_MIN_K) {
      int n_split = flash_mla_pick_nsplit(kend);
      const int stride_qh = n_split * (2 + L);
      auto& ws = flash_mla_ws_get(q_latent.device(), (int64_t)nq * nh * stride_qh);
      dim3 grid(nh, nq, n_split);
      flash_mla_sm80_splitk_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, ws.data_ptr<float>(), nq, nk, (int)q_start, nh, n_split, stride_qh);
      dim3 cgrid(nh, nq);
      flash_mla_sm80_combine_kernel<<<cgrid, THREADS, 0, stream>>>(
        ws.data_ptr<float>(), op, nq, nh, n_split, stride_qh);
    } else {
      dim3 grid(nh, nq);
      flash_mla_sm80_warp_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, op, nq, nk, (int)q_start, nh);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor flash_mla_sm80_out(torch::Tensor q_latent, torch::Tensor q_rope,
                             torch::Tensor cache, int64_t q_start, torch::Tensor out) {
    TORCH_CHECK(q_latent.is_cuda() && q_rope.is_cuda() && cache.is_cuda() && out.is_cuda(), "CUDA");
    TORCH_CHECK(q_latent.is_contiguous() && q_rope.is_contiguous() && cache.is_contiguous() && out.is_contiguous(), "contig");
    TORCH_CHECK(out.sizes()==q_latent.sizes(), "out shape");
    c10::cuda::CUDAGuard guard(q_latent.device());
    const int nh = (int)q_latent.size(1);
    const int nq = (int)q_latent.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half *ql = reinterpret_cast<half *>(q_latent.data_ptr<at::Half>());
    const half *qrp = reinterpret_cast<half *>(q_rope.data_ptr<at::Half>());
    const half *cp = reinterpret_cast<half *>(cache.data_ptr<at::Half>());
    half *op = reinterpret_cast<half *>(out.data_ptr<at::Half>());
    const int nk = (int)cache.size(0);
    const int kend = (int)std::min<int64_t>(nk, q_start + nq);

    if (flash_mla_use_t1_swarp(nq, (int)q_start)) {
      flash_mla_sm80_t1_swarp_kernel<<<nh, 32, 0, stream>>>(
        ql, qrp, cp, op, nk, (int)q_start, nh, nullptr);
    } else if (kend >= FLASH_MLA_SPLITK_MIN_K) {
      int n_split = flash_mla_pick_nsplit(kend);
      const int stride_qh = n_split * (2 + L);
      auto& ws = flash_mla_ws_get(q_latent.device(), (int64_t)nq * nh * stride_qh);
      dim3 grid(nh, nq, n_split);
      flash_mla_sm80_splitk_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, ws.data_ptr<float>(), nq, nk, (int)q_start, nh, n_split, stride_qh);
      dim3 cgrid(nh, nq);
      flash_mla_sm80_combine_kernel<<<cgrid, THREADS, 0, stream>>>(
        ws.data_ptr<float>(), op, nq, nh, n_split, stride_qh);
    } else {
      dim3 grid(nh, nq);
      flash_mla_sm80_warp_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, op, nq, nk, (int)q_start, nh);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// Graph-safe split-K: kend = min(nk, *k0_ptr+1). Q=1 only.
__global__ void flash_mla_sm80_splitk_k0_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, const int64_t* __restrict__ page_table, int page_size, float *__restrict__ ws,
        int nk, int nh, int n_split, int stride_qh, const int* __restrict__ k0_ptr) {
    const int h = blockIdx.x, sid = blockIdx.y;
    if (h >= nh || sid >= n_split) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int q_start = k0_ptr ? *k0_ptr : 0;
    const int kend = min(nk, q_start + 1);
    const int chunk = (kend + n_split - 1) / n_split;
    const int k0s = sid * chunk;
    const int k1 = min(kend, k0s + chunk);
    float *slot = ws + (size_t)h * stride_qh + (size_t)sid * (2 + L);
    if (k0s >= k1) {
        if (tid == 0) { slot[0] = -FLT_MAX; slot[1] = 0.0f; }
        for (int d = tid; d < L; d += THREADS) slot[2 + d] = 0.0f;
        return;
    }
    const half *q_lat = ql + (size_t)h * L;
    const half *q_rot = qr + (size_t)h * R;
    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32 * j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;
    for (int k = k0s + wid; k < k1; k += 8) {
        // C1_FUSE_CK: load latent V once (dot + acc share); __ldg for RO cache
        const half *ck = paged_kv_row(cache, page_table, page_size, k);
        float vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            vals[j] = __half2float(__ldg(ck + lane + 32 * j));
        float dot = qr0 * __half2float(__ldg(ck + L + lane)) +
                    qr1 * __half2float(__ldg(ck + L + lane + 32));
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * vals[j];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * vals[j];
    }
    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l, gm;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32 * j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    if (tid == 0) { slot[0] = gm; slot[1] = total_l; }
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        slot[2 + d] = v;
    }
}


// Multi-Q warp path with device k0: kend = min(nk, *k0 + tq + 1). Graph-safe.
__global__ void flash_mla_sm80_warp_k0_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, const int64_t* __restrict__ page_table, int page_size, half *__restrict__ out,
        int nq, int nk, int nh, const int* __restrict__ k0_ptr) {
    const int h = blockIdx.x, tq = blockIdx.y;
    if (h >= nh || tq >= nq) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int q_start = k0_ptr ? *k0_ptr : 0;
    const int kend = min(nk, q_start + tq + 1);
    const half *q_lat = ql + ((size_t)tq * nh + h) * L;
    const half *q_rot = qr + ((size_t)tq * nh + h) * R;
    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32 * j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;
    for (int k = wid; k < kend; k += 8) {
        const half *ck = paged_kv_row(cache, page_table, page_size, k);
        float vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            vals[j] = __half2float(__ldg(ck + lane + 32 * j));
        float dot = qr0 * __half2float(__ldg(ck + L + lane)) +
                    qr1 * __half2float(__ldg(ck + L + lane + 32));
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * vals[j];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * vals[j];
    }
    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32 * j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        float gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    half *dst = out + ((size_t)tq * nh + h) * L;
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        dst[d] = __float2half_rn(v / total_l);
    }
}

// Validated fixed single-graph Q2 production choice: shared scan, 16 warps,
// 13 effective-length splits.  The existing combine kernel consumes the same
// [query, head, split, m/l/o] workspace layout.
__device__ __forceinline__ void flash_mla_q2_write_empty(float* slot, int tid) {
    if (tid == 0) {
        slot[0] = -FLT_MAX;
        slot[1] = 0.0f;
    }
    for (int d = tid; d < L; d += blockDim.x) slot[2 + d] = 0.0f;
}

// Q=2 graph decode: one CTA shares each K scan across both causal queries.
// Work is partitioned by device-visible effective kend, so one captured graph
// remains efficient as k0 advances inside a fixed-capacity KV cache.
template <int SWARPS>
__global__ void flash_mla_sm80_splitk_k0_q2_shared_eff_kernel(
    const half* __restrict__ ql,
    const half* __restrict__ qr,
    const half* __restrict__ cache, const int64_t* __restrict__ page_table, int page_size,
    float* __restrict__ ws,
    int nk,
    int nh,
    int n_split,
    const int* __restrict__ k0_ptr) {
  const int h = blockIdx.x;
  const int sid = blockIdx.y;
  if (h >= nh || sid >= n_split) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int q_start = k0_ptr ? *k0_ptr : 0;
  const int kend0 = min(nk, q_start + 1);
  const int kend1 = min(nk, q_start + 2);
  const int chunk = (kend1 + n_split - 1) / n_split;
  const int kb = sid * chunk;
  const int ke = min(kend1, min(nk, kb + chunk));
  float* slot0 = ws + (((size_t)0 * nh + h) * n_split + sid) * (2 + L);
  float* slot1 = ws + (((size_t)1 * nh + h) * n_split + sid) * (2 + L);
  if (kb >= ke) {
    flash_mla_q2_write_empty(slot0, tid);
    flash_mla_q2_write_empty(slot1, tid);
    return;
  }

  const half* ql0 = ql + ((size_t)0 * nh + h) * L;
  const half* ql1 = ql + ((size_t)1 * nh + h) * L;
  const half* qr0p = qr + ((size_t)0 * nh + h) * R;
  const half* qr1p = qr + ((size_t)1 * nh + h) * R;
  float qv0[16], qv1[16], acc0[16], acc1[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    qv0[j] = __half2float(ql0[lane + 32 * j]);
    qv1[j] = __half2float(ql1[lane + 32 * j]);
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  const float qrr00 = __half2float(qr0p[lane]);
  const float qrr01 = __half2float(qr0p[lane + 32]);
  const float qrr10 = __half2float(qr1p[lane]);
  const float qrr11 = __half2float(qr1p[lane + 32]);
  float m0 = -FLT_MAX, l0 = 0.0f;
  float m1 = -FLT_MAX, l1 = 0.0f;

  for (int k = kb + wid; k < ke; k += SWARPS) {
    const half* ck = paged_kv_row(cache, page_table, page_size, k);
    float vals[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      vals[j] = __half2float(__ldg(ck + lane + 32 * j));
    const float cr0 = __half2float(__ldg(ck + L + lane));
    const float cr1 = __half2float(__ldg(ck + L + lane + 32));

    if (k < kend0) {
      // Interleave independent query chains while preserving each query's
      // accumulation and reduction order.
      float dot0 = qrr00 * cr0 + qrr01 * cr1;
      float dot1 = qrr10 * cr0 + qrr11 * cr1;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        dot0 += qv0[j] * vals[j];
        dot1 += qv1[j] * vals[j];
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float peer0 = __shfl_down_sync(0xffffffff, dot0, off);
        const float peer1 = __shfl_down_sync(0xffffffff, dot1, off);
        dot0 += peer0;
        dot1 += peer1;
      }
      const float score0 = __shfl_sync(0xffffffff, dot0, 0) * SCALE;
      const float score1 = __shfl_sync(0xffffffff, dot1, 0) * SCALE;
      const float m20 = fmaxf(m0, score0);
      const float alpha0 = (m0 == -FLT_MAX) ? 0.0f : __expf(m0 - m20);
      const float beta0 = __expf(score0 - m20);
      l0 = l0 * alpha0 + beta0;
      m0 = m20;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        acc0[j] = acc0[j] * alpha0 + beta0 * vals[j];
      const float m21 = fmaxf(m1, score1);
      const float alpha1 = (m1 == -FLT_MAX) ? 0.0f : __expf(m1 - m21);
      const float beta1 = __expf(score1 - m21);
      l1 = l1 * alpha1 + beta1;
      m1 = m21;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        acc1[j] = acc1[j] * alpha1 + beta1 * vals[j];
    } else {
      // Query 1 alone attends the final newly appended key.
      float dot1 = qrr10 * cr0 + qrr11 * cr1;
#pragma unroll
      for (int j = 0; j < 16; ++j) dot1 += qv1[j] * vals[j];
      dot1 = warp_sum(dot1);
      const float score1 = __shfl_sync(0xffffffff, dot1, 0) * SCALE;
      const float m21 = fmaxf(m1, score1);
      const float alpha1 = (m1 == -FLT_MAX) ? 0.0f : __expf(m1 - m21);
      const float beta1 = __expf(score1 - m21);
      l1 = l1 * alpha1 + beta1;
      m1 = m21;
#pragma unroll
      for (int j = 0; j < 16; ++j)
        acc1[j] = acc1[j] * alpha1 + beta1 * vals[j];
    }
  }

  extern __shared__ float partial[];
  float* partial0 = partial;
  float* partial1 = partial + SWARPS * L;
  __shared__ float wm0[SWARPS], wl0[SWARPS], fac0[SWARPS];
  __shared__ float wm1[SWARPS], wl1[SWARPS], fac1[SWARPS];
  __shared__ float gm0, gm1, total0, total1;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    partial0[wid * L + lane + 32 * j] = acc0[j];
    partial1[wid * L + lane + 32 * j] = acc1[j];
  }
  if (lane == 0) {
    wm0[wid] = m0;
    wl0[wid] = l0;
    wm1[wid] = m1;
    wl1[wid] = l1;
  }
  __syncthreads();
  if (tid == 0) {
    float block_m0 = -FLT_MAX, block_m1 = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < SWARPS; ++w) {
      block_m0 = fmaxf(block_m0, wm0[w]);
      block_m1 = fmaxf(block_m1, wm1[w]);
    }
    float block_l0 = 0.0f, block_l1 = 0.0f;
#pragma unroll
    for (int w = 0; w < SWARPS; ++w) {
      fac0[w] = wl0[w] == 0.0f ? 0.0f : __expf(wm0[w] - block_m0);
      fac1[w] = wl1[w] == 0.0f ? 0.0f : __expf(wm1[w] - block_m1);
      block_l0 += wl0[w] * fac0[w];
      block_l1 += wl1[w] * fac1[w];
    }
    gm0 = block_m0;
    gm1 = block_m1;
    total0 = block_l0;
    total1 = block_l1;
  }
  __syncthreads();
  if (tid == 0) {
    slot0[0] = gm0;
    slot0[1] = total0;
    slot1[0] = gm1;
    slot1[1] = total1;
  }
  for (int d = tid; d < L; d += SWARPS * 32) {
    float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
    for (int w = 0; w < SWARPS; ++w) {
      v0 += partial0[w * L + d] * fac0[w];
      v1 += partial1[w * L + d] * fac1[w];
    }
    slot0[2 + d] = v0;
    slot1[2 + d] = v1;
  }
}


// Multi-Q split-K with device k0. Split by capacity nk (empty tails ok; graph-safe).
// out_k0 multi-Q
__global__ void flash_mla_sm80_splitk_k0_mq_kernel(
        const half *__restrict__ ql, const half *__restrict__ qr,
        const half *__restrict__ cache, const int64_t* __restrict__ page_table, int page_size, float *__restrict__ ws,
        int nq, int nk, int nh, int n_split, int stride_qh,
        const int* __restrict__ k0_ptr) {
    const int h = blockIdx.x, tq = blockIdx.y, sid = blockIdx.z;
    if (h >= nh || tq >= nq || sid >= n_split) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int q_start = k0_ptr ? *k0_ptr : 0;
    const int kend = min(nk, q_start + tq + 1);
    // chunk by capacity so host need not know k0
    const int chunk = (nk + n_split - 1) / n_split;
    const int k0s = sid * chunk;
    const int k1 = min(kend, min(nk, k0s + chunk));
    float *slot = ws + ((size_t)tq * nh + h) * stride_qh + (size_t)sid * (2 + L);
    if (k0s >= k1 || k0s >= kend) {
        if (tid == 0) { slot[0] = -FLT_MAX; slot[1] = 0.0f; }
        for (int d = tid; d < L; d += THREADS) slot[2 + d] = 0.0f;
        return;
    }
    const half *q_lat = ql + ((size_t)tq * nh + h) * L;
    const half *q_rot = qr + ((size_t)tq * nh + h) * R;
    float qv[16], acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        qv[j] = __half2float(q_lat[lane + 32 * j]);
        acc[j] = 0.0f;
    }
    const float qr0 = __half2float(q_rot[lane]);
    const float qr1 = __half2float(q_rot[lane + 32]);
    float m = -FLT_MAX, l = 0.0f;
    for (int k = k0s + wid; k < k1; k += 8) {
        const half *ck = paged_kv_row(cache, page_table, page_size, k);
        float vals[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
            vals[j] = __half2float(__ldg(ck + lane + 32 * j));
        float dot = qr0 * __half2float(__ldg(ck + L + lane)) +
                    qr1 * __half2float(__ldg(ck + L + lane + 32));
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * vals[j];
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * vals[j];
    }
    __shared__ float partial[8][L];
    __shared__ float wm[8], wl[8], factor[8], total_l, gm;
#pragma unroll
    for (int j = 0; j < 16; ++j) partial[wid][lane + 32 * j] = acc[j];
    if (lane == 0) { wm[wid] = m; wl[wid] = l; }
    __syncthreads();
    if (tid == 0) {
        gm = -FLT_MAX;
#pragma unroll
        for (int w = 0; w < 8; ++w) gm = fmaxf(gm, wm[w]);
        float gl = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            factor[w] = wl[w] == 0.0f ? 0.0f : __expf(wm[w] - gm);
            gl += wl[w] * factor[w];
        }
        total_l = gl;
    }
    __syncthreads();
    if (tid == 0) { slot[0] = gm; slot[1] = total_l; }
    for (int d = tid; d < L; d += THREADS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += partial[w][d] * factor[w];
        slot[2 + d] = v;
    }
}

// Paged KV primitives for prefill. The physical owner is [P,S,576], while
// page_table maps each logical page to a physical page. Gather output is a
// reusable per-device workspace, so shuffled pages do not create per-layer
// allocator traffic. These primitives deliberately keep paging independent of
// the attention math; a direct paged MLA kernel can replace only the gather.
constexpr int PAGED_KV_D = L + R;

__global__ void paged_kv_scatter_kernel(
    const half* __restrict__ src, half* __restrict__ pool,
    const int64_t* __restrict__ table, int64_t logical_start,
    int64_t n_token, int page_size, int n_page) {
  const int64_t n = n_token * PAGED_KV_D;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (int64_t)blockDim.x * gridDim.x) {
    const int64_t t = i / PAGED_KV_D;
    const int d = (int)(i - t * PAGED_KV_D);
    const int64_t logical = logical_start + t;
    const int logical_page = (int)(logical / page_size);
    const int offset = (int)(logical - (int64_t)logical_page * page_size);
    const int64_t physical_page = table[logical_page];
    if (physical_page >= 0 && physical_page < n_page)
      pool[((physical_page * page_size + offset) * PAGED_KV_D) + d] = src[t * PAGED_KV_D + d];
  }
}

__global__ void paged_kv_gather_kernel(
    const half* __restrict__ pool, half* __restrict__ dst,
    const int64_t* __restrict__ table, int64_t logical_len,
    int page_size, int n_page) {
  const int64_t n = logical_len * PAGED_KV_D;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += (int64_t)blockDim.x * gridDim.x) {
    const int64_t logical = i / PAGED_KV_D;
    const int d = (int)(i - logical * PAGED_KV_D);
    const int logical_page = (int)(logical / page_size);
    const int offset = (int)(logical - (int64_t)logical_page * page_size);
    const int64_t physical_page = table[logical_page];
    if (physical_page >= 0 && physical_page < n_page)
      dst[i] = pool[((physical_page * page_size + offset) * PAGED_KV_D) + d];
  }
}

static std::mutex paged_ws_mu;
static std::unordered_map<int, torch::Tensor> paged_ws;

void paged_kv_scatter_cuda(torch::Tensor pool, torch::Tensor table,
                          int64_t logical_start, torch::Tensor src) {
  const int64_t n_token = src.size(0);
  const int page_size = (int)pool.size(1), n_page = (int)pool.size(0);
  const int64_t n = n_token * PAGED_KV_D;
  const int blocks = (int)std::min<int64_t>(65535, (n + THREADS - 1) / THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  paged_kv_scatter_kernel<<<blocks, THREADS, 0, stream>>>(
    reinterpret_cast<const half*>(src.data_ptr<at::Half>()),
    reinterpret_cast<half*>(pool.data_ptr<at::Half>()), table.data_ptr<int64_t>(),
    logical_start, n_token, page_size, n_page);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor paged_kv_gather_cuda(torch::Tensor pool, torch::Tensor table,
                                   int64_t logical_len) {
  const int dev = pool.get_device();
  torch::Tensor ws;
  {
    std::lock_guard<std::mutex> lock(paged_ws_mu);
    auto it = paged_ws.find(dev);
    const int64_t capacity = pool.size(0) * pool.size(1);
    if (it == paged_ws.end() || it->second.size(0) < capacity) {
      paged_ws[dev] = torch::empty({capacity, PAGED_KV_D}, pool.options());
    }
    ws = paged_ws.at(dev);
  }
  const int page_size = (int)pool.size(1), n_page = (int)pool.size(0);
  const int64_t n = logical_len * PAGED_KV_D;
  const int blocks = (int)std::min<int64_t>(65535, (n + THREADS - 1) / THREADS);
  auto stream = at::cuda::getCurrentCUDAStream();
  paged_kv_gather_kernel<<<blocks, THREADS, 0, stream>>>(
    reinterpret_cast<const half*>(pool.data_ptr<at::Half>()),
    reinterpret_cast<half*>(ws.data_ptr<at::Half>()), table.data_ptr<int64_t>(),
    logical_len, page_size, n_page);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ws.narrow(0, 0, logical_len);
}

// Production decode/batched-decode API.  T here is decode/MTP query count,
// not a prefill chunk; never reuse this API for prefill or batched prefill.
torch::Tensor flash_mla_sm80_out_k0(torch::Tensor q_latent, torch::Tensor q_rope,
                             torch::Tensor pool, torch::Tensor page_table,
                             torch::Tensor k0, torch::Tensor out) {
    // out_k0 multi-Q decode: T=1 keeps legacy kernels; T>1 uses device-k0 warp/splitk.
    TORCH_CHECK(q_latent.is_cuda() && q_rope.is_cuda() && pool.is_cuda() && page_table.is_cuda() && out.is_cuda() && k0.is_cuda(), "CUDA");
    TORCH_CHECK(k0.scalar_type()==at::kInt && k0.numel()==1, "k0 must be int32[1] on device");
    TORCH_CHECK(q_latent.is_contiguous() && q_rope.is_contiguous() && pool.is_contiguous() && page_table.is_contiguous() && out.is_contiguous() && k0.is_contiguous(), "contig");
    TORCH_CHECK(out.sizes()==q_latent.sizes(), "out shape");
    TORCH_CHECK(pool.dim()==3 && pool.size(2)==L+R, "pool must be [pages,page_size,576]");
    TORCH_CHECK(page_table.dim()==1 && page_table.scalar_type()==at::kLong, "page table must be int64[pages]");
    TORCH_CHECK(page_table.numel()==pool.size(0), "page table length mismatch");
    c10::cuda::CUDAGuard guard(q_latent.device());
    const int nh = (int)q_latent.size(1);
    const int nq = (int)q_latent.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(nq >= 1 && nq <= 4, "out_k0 supports T=1..4");
    const int page_size = (int)pool.size(1);
    const int nk = (int)(pool.size(0) * pool.size(1));
    const half *ql = reinterpret_cast<half *>(q_latent.data_ptr<at::Half>());
    const half *qrp = reinterpret_cast<half *>(q_rope.data_ptr<at::Half>());
    const half *cp = reinterpret_cast<half *>(pool.data_ptr<at::Half>());
    const int64_t *pt = page_table.data_ptr<int64_t>();
    half *op = reinterpret_cast<half *>(out.data_ptr<at::Half>());
    int *k0p = k0.data_ptr<int>();
    // Graph path: kend unknown on host; split by capacity; empty tails ok.
    if (nq == 1) {
      if (nk >= FLASH_MLA_SPLITK_MIN_K) {
        int n_split = flash_mla_pick_nsplit(nk);
        const int stride_qh = n_split * (2 + L);
        auto& ws = flash_mla_ws_get(q_latent.device(), (int64_t)nh * stride_qh);
        dim3 grid(nh, n_split);
        flash_mla_sm80_splitk_k0_kernel<<<grid, THREADS, 0, stream>>>(
          ql, qrp, cp, pt, page_size, ws.data_ptr<float>(), nk, nh, n_split, stride_qh, k0p);
        dim3 cgrid(nh, 1);
        flash_mla_sm80_combine_kernel<<<cgrid, THREADS, 0, stream>>>(
          ws.data_ptr<float>(), op, 1, nh, n_split, stride_qh);
      } else {
        flash_mla_sm80_t1_mwarp_kernel<<<nh, THREADS, 0, stream>>>(
          ql, qrp, cp, pt, page_size, op, nk, 0, nh, k0p);
      }
    } else if (nq == 2 && nk >= FLASH_MLA_SPLITK_MIN_K) {
      constexpr int n_split = 13;
      constexpr int q2_swarps = 16;
      const int stride_qh = n_split * (2 + L);
      auto& ws = flash_mla_ws_get(q_latent.device(), (int64_t)2 * nh * stride_qh);
      constexpr int smem = 2 * q2_swarps * L * (int)sizeof(float);
      C10_CUDA_CHECK(cudaFuncSetAttribute(
        flash_mla_sm80_splitk_k0_q2_shared_eff_kernel<q2_swarps>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      dim3 grid(nh, n_split);
      flash_mla_sm80_splitk_k0_q2_shared_eff_kernel<q2_swarps>
        <<<grid, q2_swarps * 32, smem, stream>>>(
          ql, qrp, cp, pt, page_size, ws.data_ptr<float>(), nk, nh, n_split, k0p);
      dim3 cgrid(nh, 2);
      flash_mla_sm80_combine_kernel<<<cgrid, THREADS, 0, stream>>>(
        ws.data_ptr<float>(), op, 2, nh, n_split, stride_qh);
    } else if (nk >= FLASH_MLA_SPLITK_MIN_K) {
      int n_split = flash_mla_pick_nsplit(nk);
      const int stride_qh = n_split * (2 + L);
      auto& ws = flash_mla_ws_get(q_latent.device(), (int64_t)nq * nh * stride_qh);
      dim3 grid(nh, nq, n_split);
      flash_mla_sm80_splitk_k0_mq_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, pt, page_size, ws.data_ptr<float>(), nq, nk, nh, n_split, stride_qh, k0p);
      dim3 cgrid(nh, nq);
      flash_mla_sm80_combine_kernel<<<cgrid, THREADS, 0, stream>>>(
        ws.data_ptr<float>(), op, nq, nh, n_split, stride_qh);
    } else {
      dim3 grid(nh, nq);
      flash_mla_sm80_warp_k0_kernel<<<grid, THREADS, 0, stream>>>(
        ql, qrp, cp, pt, page_size, op, nq, nk, nh, k0p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



