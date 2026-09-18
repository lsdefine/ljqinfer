#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

// out[tok[n], d] += (float)sbuf[n,d] * weight[n]
// vectorized float2 path when D even; one block per row
__global__ void weighted_scatter_row_kernel(
    const __half* __restrict__ sbuf,
    const float* __restrict__ weight,
    const int64_t* __restrict__ tok,
    float* __restrict__ out,
    int64_t N, int64_t D) {
  int64_t n = blockIdx.x;
  if (n >= N) return;
  int64_t t = tok[n];
  float w = weight[n];
  const __half* s = sbuf + n * D;
  float* o = out + t * D;
  // process 2 floats at a time when possible
  int64_t d = threadIdx.x * 2;
  int64_t stride = blockDim.x * 2;
  for (; d + 1 < D; d += stride) {
    float v0 = __half2float(s[d]) * w;
    float v1 = __half2float(s[d + 1]) * w;
    atomicAdd(o + d, v0);
    atomicAdd(o + d + 1, v1);
  }
  if (d < D) {
    atomicAdd(o + d, __half2float(s[d]) * w);
  }
}

// flat grid-stride over N*D — better occupancy when N large
__global__ void weighted_scatter_flat_kernel(
    const __half* __restrict__ sbuf,
    const float* __restrict__ weight,
    const int64_t* __restrict__ tok,
    float* __restrict__ out,
    int64_t N, int64_t D) {
  int64_t ND = N * D;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < ND; i += (int64_t)blockDim.x * gridDim.x) {
    int64_t n = i / D;
    int64_t d = i - n * D;
    float v = __half2float(sbuf[i]) * weight[n];
    atomicAdd(out + tok[n] * D + d, v);
  }
}

// Build token -> dispatched-row mapping, then sort row indices. Dispatch rows
// are expert-bucket ordered, so row-index order is the fixed expert order.
__global__ void build_token_rows_kernel(
    const int64_t* __restrict__ tok, int64_t* __restrict__ rows,
    int* __restrict__ counts, int64_t N, int64_t B) {
  int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int64_t t = tok[n];
  if ((uint64_t)t >= (uint64_t)B) return;
  int slot = atomicAdd(counts + t, 1);
  if (slot < 8) rows[t * 8 + slot] = n;
}

__global__ void sort_token_rows_kernel(int64_t* rows, int64_t B) {
  int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B) return;
  int64_t* a = rows + t * 8;
  #pragma unroll
  for (int i = 1; i < 8; ++i) {
    int64_t v = a[i];
    int j = i - 1;
    while (j >= 0 && (a[j] < 0 || (v >= 0 && a[j] > v))) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = v;
  }
}

__global__ void deterministic_scatter_kernel(
    const __half* __restrict__ sbuf, const float* __restrict__ weight,
    const int64_t* __restrict__ rows, float* __restrict__ out,
    int64_t B, int64_t D) {
  int64_t total = B * D;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total; i += (int64_t)blockDim.x * gridDim.x) {
    int64_t t = i / D;
    int64_t d = i - t * D;
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int64_t n = rows[t * 8 + j];
      if (n >= 0) acc += __half2float(sbuf[n * D + d]) * weight[n];
    }
    out[i] = acc;
  }
}

// half2 vectorized flat: D must be even
__global__ void weighted_scatter_flat_h2_kernel(
    const __half* __restrict__ sbuf,
    const float* __restrict__ weight,
    const int64_t* __restrict__ tok,
    float* __restrict__ out,
    int64_t N, int64_t D) {
  int64_t ND2 = N * (D >> 1);
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < ND2; i += (int64_t)blockDim.x * gridDim.x) {
    int64_t n = i / (D >> 1);
    int64_t d2 = i - n * (D >> 1);
    int64_t d = d2 << 1;
    float w = weight[n];
    __half2 h2 = *reinterpret_cast<const __half2*>(sbuf + n * D + d);
    float2 f2 = __half22float2(h2);
    float* o = out + tok[n] * D + d;
    atomicAdd(o, f2.x * w);
    atomicAdd(o + 1, f2.y * w);
  }
}

torch::Tensor weighted_scatter_fp16(
    torch::Tensor sbuf,   // [N,D] half
    torch::Tensor weight, // [N,1] or [N] float
    torch::Tensor tok,    // [N] int64
    int64_t B,
    int64_t mode = 2) {   // 0=row, 1=flat, 2=flat_h2
  TORCH_CHECK(sbuf.is_cuda() && sbuf.scalar_type()==torch::kFloat16 && sbuf.dim()==2 && sbuf.is_contiguous());
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kLong && tok.is_contiguous());
  TORCH_CHECK(weight.is_cuda());
  auto w = weight.contiguous().view({-1}).to(torch::kFloat32);
  TORCH_CHECK(w.size(0)==sbuf.size(0) && tok.size(0)==sbuf.size(0));
  const int64_t N = sbuf.size(0), D = sbuf.size(1);
  c10::cuda::CUDAGuard guard(sbuf.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = torch::zeros({B, D}, sbuf.options().dtype(torch::kFloat32));
  if (N == 0) return out;
  const __half* sp = reinterpret_cast<const __half*>(sbuf.data_ptr<at::Half>());
  const float* wp = w.data_ptr<float>();
  const int64_t* tp = tok.data_ptr<int64_t>();
  float* op = out.data_ptr<float>();
  if (mode == 0) {
    int threads = 256;
    weighted_scatter_row_kernel<<<(unsigned)N, threads, 0, stream.stream()>>>(sp, wp, tp, op, N, D);
  } else if (mode == 1) {
    TORCH_CHECK(N <= B * 8, "deterministic scatter requires top-k <= 8");
    int threads = 256;
    auto rows = torch::full({B, 8}, -1, tok.options());
    auto counts = torch::zeros({B}, tok.options().dtype(torch::kInt32));
    int map_blocks = (int)((N + threads - 1) / threads);
    build_token_rows_kernel<<<map_blocks, threads, 0, stream.stream()>>>(
        tp, rows.data_ptr<int64_t>(), counts.data_ptr<int>(), N, B);
    sort_token_rows_kernel<<<(unsigned)((B + threads - 1) / threads), threads,
                             0, stream.stream()>>>(rows.data_ptr<int64_t>(), B);
    int blocks = (int)std::min<int64_t>((B * D + threads - 1) / threads, 2048);
    deterministic_scatter_kernel<<<blocks, threads, 0, stream.stream()>>>(
        sp, wp, rows.data_ptr<int64_t>(), op, B, D);
  } else {
    TORCH_CHECK((D & 1) == 0, "flat_h2 needs even D");
    int threads = 256;
    int64_t nwork = N * (D >> 1);
    int blocks = (int)std::min<int64_t>((nwork + threads - 1) / threads, 2048);
    weighted_scatter_flat_h2_kernel<<<blocks, threads, 0, stream.stream()>>>(sp, wp, tp, op, N, D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

