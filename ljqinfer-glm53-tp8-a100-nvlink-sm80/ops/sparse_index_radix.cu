#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// Extracted from DSV4 index_score.cu, as used by DSV41f decode.
// ---- single-kernel exact top-K select fused with mask+offset post ----
// Replaces at::topk (mbtopk multi-kernel chain) + topk_post[_positions].
// Output set is exact top-K; emission order is deterministic for stable FP reduction.
__device__ __forceinline__ unsigned f2key(float v) {
  unsigned u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ __launch_bounds__(1024) void topk_select_post_kernel(
    const float* __restrict__ score,        // [S, N]
    int* __restrict__ out,                  // [S, K]
    const int64_t* __restrict__ offsets,    // per-row, or null -> offset0
    const int64_t* __restrict__ positions,  // per-row, or null -> pos0 + s
    int N, int K, int ratio, long offset0, long pos0) {
  const long s = blockIdx.x;
  const float* row = score + s * (long)N;
  const int lim = (int)(((positions ? positions[s] : pos0 + s) + 1) / ratio);
  // Live prefix: the tail [lim, N) is -INF by construction, so skipping it is
  // exact.  This is what keeps one static graph cheap on a 1M-capacity pool.
  const int NL = (lim <= 0) ? 0 : ((lim < N) ? lim : N);
  const long off = offsets ? offsets[s] : offset0;

  if (NL <= K) {
    for (int t = threadIdx.x; t < K; t += blockDim.x)
      out[s * (long)K + t] = (t < NL) ? (int)(t + off) : -1;
    return;
  }

  __shared__ int hist[256];
  __shared__ unsigned s_prefix;
  __shared__ int s_k;
  constexpr int EMIT_ITEMS = 8;
  __shared__ int warp_gt[32 * EMIT_ITEMS], warp_eq[32 * EMIT_ITEMS];
  __shared__ int base_gt, base_eq, chunk_gt, chunk_eq;
  if (threadIdx.x == 0) { s_prefix = 0u; s_k = K; }
  __syncthreads();

  for (int pass = 3; pass >= 0; --pass) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const unsigned mask_hi = (pass == 3) ? 0u : (0xFFFFFFFFu << ((pass + 1) * 8));
    const unsigned prefix = s_prefix;
    const int N4 = ((N & 3) == 0) ? (NL >> 2) : 0;  // row offset s*N must stay 16B-aligned for float4
    const float4* row4 = reinterpret_cast<const float4*>(row);
    for (int t = threadIdx.x; t < N4; t += blockDim.x) {
      const float4 v = row4[t];
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned u = f2key(vv[j]);
        if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> (pass * 8)) & 0xFF], 1);
      }
    }
    for (int t = (N4 << 2) + threadIdx.x; t < NL; t += blockDim.x) {
      const unsigned u = f2key(row[t]);
      if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> (pass * 8)) & 0xFF], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int kk = s_k, b = 255;
      for (; b > 0; --b) { if (hist[b] >= kk) break; kk -= hist[b]; }
      s_k = kk;
      s_prefix = s_prefix | ((unsigned)b << (pass * 8));
    }
    __syncthreads();
  }

  // Emit the exact same top-K set in a deterministic order.  The previous
  // atomic slot allocator made the output permutation depend on warp timing;
  // sparse attention is set-equivalent but its floating-point reduction is not
  // permutation invariant.  Integer radix histograms above remain deterministic.
  const unsigned thr = s_prefix;
  const int need_tie = s_k;
  const int n_gt = K - need_tie;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
  if (threadIdx.x == 0) { base_gt = 0; base_eq = 0; }
  __syncthreads();
  for (int begin = 0; begin < NL; begin += blockDim.x * EMIT_ITEMS) {
    unsigned mask_gt[EMIT_ITEMS], mask_eq[EMIT_ITEMS];
#pragma unroll
    for (int item = 0; item < EMIT_ITEMS; ++item) {
      const int t = begin + item * blockDim.x + threadIdx.x;
      const unsigned u = (t < NL) ? f2key(row[t]) : 0u;
      mask_gt[item] = __ballot_sync(0xffffffffu, t < NL && u > thr);
      mask_eq[item] = __ballot_sync(0xffffffffu, t < NL && u == thr);
      if (lane == 0) {
        const int segment = item * 32 + warp;
        warp_gt[segment] = __popc(mask_gt[item]);
        warp_eq[segment] = __popc(mask_eq[item]);
      }
    }
    __syncthreads();
    // Eight warps scan the 256 deterministic emission segments in parallel.
    __shared__ int scan_gt[8], scan_eq[8];
    if (threadIdx.x < 256) {
      const int i = threadIdx.x, w = i >> 5, l = i & 31;
      const int cg = warp_gt[i], ce = warp_eq[i];
      int pg = cg, pe = ce;
#pragma unroll
      for (int delta=1; delta<32; delta*=2) {
        int g=__shfl_up_sync(0xffffffffu,pg,delta);
        int e=__shfl_up_sync(0xffffffffu,pe,delta);
        if(l>=delta){pg+=g;pe+=e;}
      }
      warp_gt[i]=pg-cg;warp_eq[i]=pe-ce;
      if(l==31){scan_gt[w]=pg;scan_eq[w]=pe;}
    }
    __syncthreads();
    if(threadIdx.x<32){
      int g=threadIdx.x<8?scan_gt[threadIdx.x]:0;
      int e=threadIdx.x<8?scan_eq[threadIdx.x]:0;
      const int cg=g,ce=e;
#pragma unroll
      for(int delta=1;delta<8;delta*=2){
        int ng=__shfl_up_sync(0xffffffffu,g,delta);
        int ne=__shfl_up_sync(0xffffffffu,e,delta);
        if(threadIdx.x>=delta){g+=ng;e+=ne;}
      }
      if(threadIdx.x<8){scan_gt[threadIdx.x]=g-cg;scan_eq[threadIdx.x]=e-ce;}
      if(threadIdx.x==7){chunk_gt=g;chunk_eq=e;}
    }
    __syncthreads();
    if(threadIdx.x<256){warp_gt[threadIdx.x]+=scan_gt[threadIdx.x>>5];warp_eq[threadIdx.x]+=scan_eq[threadIdx.x>>5];}
    __syncthreads();
#pragma unroll
    for (int item = 0; item < EMIT_ITEMS; ++item) {
      const int t = begin + item * blockDim.x + threadIdx.x;
      const int segment = item * 32 + warp;
      if (mask_gt[item] & (1u << lane)) {
        const int rank = base_gt + warp_gt[segment] + __popc(mask_gt[item] & lane_mask);
        if (rank < n_gt)
          out[s * (long)K + rank] = (t >= lim) ? -1 : (int)(t + off);
      }
      if (mask_eq[item] & (1u << lane)) {
        const int rank = base_eq + warp_eq[segment] + __popc(mask_eq[item] & lane_mask);
        if (rank < need_tie)
          out[s * (long)K + n_gt + rank] = (t >= lim) ? -1 : (int)(t + off);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      base_gt += chunk_gt;
      base_eq += chunk_eq;
    }
    __syncthreads();
  }
}


void select_out(at::Tensor scores, at::Tensor positions, at::Tensor out) {
 c10::cuda::CUDAGuard guard(scores.device());
 TORCH_CHECK(scores.dim()==2 && out.dim()==2 && positions.dim()==1,"ranks");
 for(const auto& x:{scores,positions,out}) TORCH_CHECK(x.is_cuda() && x.device()==scores.device() && x.is_contiguous(),"device/contiguous");
 TORCH_CHECK(scores.scalar_type()==at::kFloat && out.scalar_type()==at::kInt && positions.scalar_type()==at::kLong,"dtypes");
 TORCH_CHECK(scores.size(0)==out.size(0) && positions.numel()==out.size(0) && out.size(1)>0 && out.size(1)<=scores.size(1),"shapes");
 if(!out.size(0))return;
 topk_select_post_kernel<<<out.size(0),1024,0,at::cuda::getCurrentCUDAStream()>>>(scores.data_ptr<float>(),out.data_ptr<int>(),nullptr,positions.data_ptr<int64_t>(),scores.size(1),out.size(1),1,0L,0L);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("select_out",&select_out);}
