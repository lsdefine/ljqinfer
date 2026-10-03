#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// One independent block per token.  The routing semantics are identical for
// every small-path T<128: sigmoid -> top-2 group score -> top-4 groups -> top-8 experts,
// then renormalize the un-biased sigmoid weights to sum to 2.5.
__global__ void decode_moe_route_q13_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int64_t* __restrict__ ei,
    float* __restrict__ ew,
    const bool input_is_probs) {
  const int b = blockIdx.x;
  input += b * 256;
  ei += b * 8;
  ew += b * 8;

  __shared__ float probs[256];
  __shared__ float sel[256];
  __shared__ float gscore[8];
  __shared__ int gorder[8];
  __shared__ float cand[128];
  __shared__ int topi[8];
  __shared__ int has_nonfinite;

  const int tid = threadIdx.x;
  if (tid == 0) has_nonfinite = 0;
  __syncthreads();
  const float p = input_is_probs
      ? input[tid]
      : (1.0f / (1.0f + expf(-input[tid])));
  probs[tid] = p;
  sel[tid] = p + bias[tid];
  if (!isfinite(p) || !isfinite(sel[tid])) atomicExch(&has_nonfinite, 1);
  __syncthreads();

  if (tid < 8) {
    const float* row = sel + tid * 32;
    float m1 = -FLT_MAX, m2 = -FLT_MAX;
    for (int j = 0; j < 32; ++j) {
      const float v = row[j];
      if (v > m1) { m2 = m1; m1 = v; }
      else if (v > m2) { m2 = v; }
    }
    gscore[tid] = m1 + m2;
    gorder[tid] = tid;
    if (!isfinite(gscore[tid])) atomicExch(&has_nonfinite, 1);
  }
  __syncthreads();

  // Finite values use the parallel exact-rank fast path. NaN/Inf values must
  // retain the previous serial comparison semantics: unordered comparisons can
  // otherwise assign several items the same rank and leave topi uninitialized.
  if (has_nonfinite) {
    if (tid == 0) {
      for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 8; ++j) {
          if (gscore[j] > gscore[i] ||
              (gscore[j] == gscore[i] && gorder[j] < gorder[i])) {
            const float ts = gscore[i]; gscore[i] = gscore[j]; gscore[j] = ts;
            const int ti = gorder[i]; gorder[i] = gorder[j]; gorder[j] = ti;
          }
        }
      }
    }
  } else if (tid < 8) {
    const float v = gscore[tid];
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      rank += (gscore[j] > v || (gscore[j] == v && j < tid));
    gorder[rank] = tid;
  }
  __syncthreads();

  if (tid < 128) {
    const int gi = gorder[tid >> 5];
    cand[tid] = sel[gi * 32 + (tid & 31)];
  }
  __syncthreads();

  // Every finite candidate computes its exact rank. The non-finite fallback
  // below is the old serial insertion policy.
  if (!has_nonfinite && tid < 128) {
    const float v = cand[tid];
    int rank = 0;
#pragma unroll 4
    for (int j = 0; j < 128; ++j)
      rank += (cand[j] > v || (cand[j] == v && j < tid));
    if (rank < 8) topi[rank] = tid;
  }
  __syncthreads();

  if (has_nonfinite && tid == 0) {
    float topv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { topv[k] = -FLT_MAX; topi[k] = 0; }
    for (int i = 0; i < 128; ++i) {
      const float v = cand[i];
      if (v <= topv[7]) continue;
      int pos = 7;
      while (pos > 0 && v > topv[pos - 1]) {
        topv[pos] = topv[pos - 1];
        topi[pos] = topi[pos - 1];
        --pos;
      }
      topv[pos] = v;
      topi[pos] = i;
    }
  }
  __syncthreads();

  if (tid == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = topi[k];
      const int expert = gorder[i >> 5] * 32 + (i & 31);
      ei[k] = static_cast<int64_t>(expert);
      ew[k] = probs[expert];
      sum += ew[k];
    }
    const float inv = (sum > 6.103515625e-5f) ? (2.5f / sum) : 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) ew[k] *= inv;
  }
}


void decode_moe_route_probs_q13_cuda(torch::Tensor probs, torch::Tensor bias,
                                     torch::Tensor ei, torch::Tensor ew) {
  TORCH_CHECK(probs.is_cuda() && bias.is_cuda() && ei.is_cuda() && ew.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(probs.is_contiguous() && bias.is_contiguous() &&
              ei.is_contiguous() && ew.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(probs.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat &&
              ei.scalar_type() == at::kLong && ew.scalar_type() == at::kFloat,
              "expected float probs/bias/weights and int64 indices");
  TORCH_CHECK(bias.numel() == 256 && probs.numel() % 256 == 0,
              "expected probs [Q,256] and bias [256]");
  const int64_t q = probs.numel() / 256;
  TORCH_CHECK(q >= 1 && q < 128, "small route expects T=1..127");
  TORCH_CHECK(ei.numel() == q * 8 && ew.numel() == q * 8,
              "expected indices/weights [Q,8]");
  TORCH_CHECK(probs.device() == bias.device() && probs.device() == ei.device() &&
              probs.device() == ew.device(), "all tensors must be on same device");
  c10::cuda::CUDAGuard guard(probs.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_moe_route_q13_kernel<<<static_cast<unsigned>(q), 256, 0, stream>>>(
      probs.data_ptr<float>(), bias.data_ptr<float>(),
      ei.data_ptr<int64_t>(), ew.data_ptr<float>(), true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void decode_moe_route_q13_cuda(torch::Tensor logits, torch::Tensor bias,
                               torch::Tensor ei, torch::Tensor ew) {
  TORCH_CHECK(logits.is_cuda() && bias.is_cuda() && ei.is_cuda() && ew.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(logits.is_contiguous() && bias.is_contiguous() &&
              ei.is_contiguous() && ew.is_contiguous(), "all tensors must be contiguous");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat &&
              ei.scalar_type() == at::kLong && ew.scalar_type() == at::kFloat,
              "expected float logits/bias/weights and int64 indices");
  TORCH_CHECK(bias.numel() == 256 && logits.numel() % 256 == 0,
              "expected logits [Q,256] and bias [256]");
  const int64_t q = logits.numel() / 256;
  TORCH_CHECK(q >= 1 && q < 128, "small route expects T=1..127");
  TORCH_CHECK(ei.numel() == q * 8 && ew.numel() == q * 8,
              "expected indices/weights [Q,8]");
  TORCH_CHECK(logits.device() == bias.device() && logits.device() == ei.device() &&
              logits.device() == ew.device(), "all tensors must be on same device");
  c10::cuda::CUDAGuard guard(logits.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_moe_route_q13_kernel<<<static_cast<unsigned>(q), 256, 0, stream>>>(
      logits.data_ptr<float>(), bias.data_ptr<float>(),
      ei.data_ptr<int64_t>(), ew.data_ptr<float>(), false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Keep the old Python symbols and behavior unchanged.
  m.def("moe_route_t1_cuda", &decode_moe_route_q13_cuda);
  m.def("decode_moe_route_probs_q13_cuda", &decode_moe_route_probs_q13_cuda);
}
