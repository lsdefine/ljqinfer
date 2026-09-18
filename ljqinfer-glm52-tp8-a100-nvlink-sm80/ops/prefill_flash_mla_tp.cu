#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

// GLM-5.2 absorbed MLA attention for SM80.
// One CUDA block owns one (query, head), streams causal K once, and maintains
// the softmax numerator/denominator online. No [H,Q,K] score tensor exists.
constexpr int L = 512;
constexpr int R = 64;
constexpr int THREADS = 256;
constexpr float SCALE = 0.0625f;

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
        const half *ck = cache + (size_t)k * (L + R);
        float dot = qr0 * __half2float(ck[L + lane]) +
                    qr1 * __half2float(ck[L + lane + 32]);
#pragma unroll
        for (int j = 0; j < 16; ++j)
            dot += qv[j] * __half2float(ck[lane + 32*j]);
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffff, dot, 0) * SCALE;
        const float m2 = fmaxf(m, score);
        const float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m2);
        const float beta = __expf(score - m2);
        l = l * alpha + beta;
        m = m2;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] = acc[j] * alpha + beta * __half2float(ck[lane + 32*j]);
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
    const int nh = q_latent.size(1);
    dim3 grid(nh, q_latent.size(0));
    flash_mla_sm80_warp_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half *>(q_latent.data_ptr<at::Half>()),
        reinterpret_cast<half *>(q_rope.data_ptr<at::Half>()),
        reinterpret_cast<half *>(cache.data_ptr<at::Half>()),
        reinterpret_cast<half *>(out.data_ptr<at::Half>()),
        q_latent.size(0), cache.size(0), q_start, nh);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_mla_sm80, "SM80 absorbed FlashMLA forward", pybind11::call_guard<pybind11::gil_scoped_release>());
}
