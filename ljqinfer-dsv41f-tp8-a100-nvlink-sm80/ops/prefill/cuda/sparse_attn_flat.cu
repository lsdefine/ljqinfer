// v4.1 fused sparse latent attention (prefill).  Ported from the v4 paged
// tensor-core kernel: 1 block = 1 token, 8 warps, TILE=32 keys staged in
// shared, V == K (latent MLA), online softmax with a sink denominator.
// v4.1 differences: flat bank (no paging), explicit `bad` mask instead of
// id<0, prefill only (T >> SM count, so no split-K), fp32 softmax.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#define DIM 512
#define NWARP 8
#define TILE 32
#define LDK (DIM + 8)
#define LDP (TILE + 8)

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                         const unsigned* b, const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned pk(const bf16* p) {
  return *(const unsigned*)p;  // two contiguous bf16
}

__global__ __launch_bounds__(NWARP * 32) void sparse_attn_flat_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ bank,
    const int64_t* __restrict__ idxs, const bool* __restrict__ bad,
    const float* __restrict__ sink, bf16* __restrict__ out, int K, float scale,
    int nhead) {
  extern __shared__ char smem[];
  bf16* kvs = (bf16*)smem;                  // [TILE][LDK]
  bf16* ps = kvs + TILE * LDK;              // [8][LDP]
  float* spart = (float*)(ps + 8 * LDP);    // [2][8][TILE]
  float* sh_corr = spart + 2 * 8 * TILE;    // [8]
  float* sh_m = sh_corr + 8;                // [8]
  float* sh_l = sh_m + 8;                   // [8]
  int* ok = (int*)(sh_l + 8);               // [TILE]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7  -> head
  const int tig = lane & 3;    // 0..3

  const int kg = w >> 2;             // contraction half (0/1)
  const int keybase = (w & 3) * 8;   // this warp's 8 keys for QK^T
  const int dimbase = w * 64;        // this warp's 64 dims for PV

  unsigned qa[16][2];
  {
    const bf16* qp = q + ((size_t)token * nhead + gid) * DIM + kg * 256;
    const bool live = (gid < nhead);
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      qa[ks][0] = live ? pk(qp + ks * 16 + tig * 2) : 0u;
      qa[ks][1] = live ? pk(qp + ks * 16 + tig * 2 + 8) : 0u;
    }
  }

  float acc[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.f;
  if (tid < 8) { sh_m[tid] = -INFINITY; sh_l[tid] = 0.f; }

  const int64_t* myidx = idxs + (size_t)token * K;
  const bool* mybad = bad + (size_t)token * K;

  for (int kb = 0; kb < K; kb += TILE) {
    __syncthreads();
    for (int slot = w; slot < TILE; slot += NWARP) {
      const int kk = kb + slot;
      int good = 0;
      const bf16* src = nullptr;
      if (kk < K && !mybad[kk]) {
        src = bank + myidx[kk] * (size_t)DIM;
        good = 1;
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (good) v = *(const uint4*)(src + j * 256 + lane * 8);
        *(uint4*)(kvs + slot * LDK + j * 256 + lane * 8) = v;
      }
      if (lane == 0) ok[slot] = good;
    }
    __syncthreads();

    {
      int any = 0;
#pragma unroll
      for (int i = 0; i < TILE; ++i) any |= ok[i];
      if (!any) continue;
    }

    // ---- QK^T : 16(head) x 8(key) x 256(dim half) ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const bf16* kp = kvs + (keybase + gid) * LDK + kg * 256;
#pragma unroll 4
      for (int ks = 0; ks < 16; ++ks) {
        unsigned a[4] = {qa[ks][0], 0u, qa[ks][1], 0u};
        unsigned b[2];
        b[0] = pk(kp + ks * 16 + tig * 2);
        b[1] = pk(kp + ks * 16 + tig * 2 + 8);
        mma16816(c, a, b, c);
      }
      spart[(kg * 8 + gid) * TILE + keybase + tig * 2] = c[0];
      spart[(kg * 8 + gid) * TILE + keybase + tig * 2 + 1] = c[1];
    }
    __syncthreads();

    // ---- online softmax: warp w owns head w, lane = key slot ----
    if (w < 8) {
      float s = -INFINITY;
      if (ok[lane] && (kb + lane) < K)
        s = (spart[w * TILE + lane] + spart[(8 + w) * TILE + lane]) * scale;
      float mx = s;
#pragma unroll
      for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mold = sh_m[w];
      const float mnew = fmaxf(mold, mx);
      const float corr = (mold == -INFINITY) ? 0.f : __expf(mold - mnew);
      float p = (s == -INFINITY) ? 0.f : __expf(s - mnew);
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      ps[w * LDP + lane] = __float2bfloat16(p);
      if (lane == 0) {
        sh_corr[w] = corr;
        sh_m[w] = mnew;
        sh_l[w] = sh_l[w] * corr + sum;
      }
    }
    __syncthreads();

    {
      const float corr = sh_corr[gid];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        acc[t][0] *= corr;
        acc[t][1] *= corr;
      }
    }

    // ---- PV : 16(head) x 64(dim) x 32(key), V == K ----
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int dcol = dimbase + t * 8 + gid;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int k0 = ks * 16;
        unsigned a[4], b[2];
        a[0] = pk(ps + gid * LDP + k0 + tig * 2);
        a[1] = 0u;
        a[2] = pk(ps + gid * LDP + k0 + 8 + tig * 2);
        a[3] = 0u;
        const bf16* vp = kvs + (k0 + tig * 2) * LDK + dcol;
        b[0] = *(unsigned*)&__halves2bfloat162(vp[0], vp[LDK]);
        const bf16* vq = vp + 8 * LDK;
        b[1] = *(unsigned*)&__halves2bfloat162(vq[0], vq[LDK]);
        mma16816(acc[t], a, b, acc[t]);
      }
    }
  }

  __syncthreads();
  if (gid < nhead) {
    const float l = sh_l[gid] + __expf(sink[gid] - sh_m[gid]);
    const float inv = 1.f / l;
    bf16* op = out + ((size_t)token * nhead + gid) * DIM;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int d = dimbase + t * 8 + tig * 2;
      op[d] = __float2bfloat16(acc[t][0] * inv);
      op[d + 1] = __float2bfloat16(acc[t][1] * inv);
    }
  }
}

void sparse_attn_flat(at::Tensor q, at::Tensor bank, at::Tensor idxs,
                      at::Tensor bad, at::Tensor sink, at::Tensor out,
                      double scale) {
  const int h = q.size(1), K = idxs.size(-1), T = q.size(0);
  TORCH_CHECK(q.size(2) == DIM && bank.size(1) == DIM, "latent dim must be 512");
  TORCH_CHECK(h <= 8, "at most 8 heads per rank");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  bank.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16,
              "q/bank/out must be bf16");
  TORCH_CHECK(idxs.scalar_type() == at::kLong && bad.scalar_type() == at::kBool,
              "idxs int64 / bad bool");
  TORCH_CHECK(K % TILE == 0, "K must be a multiple of 32");
  TORCH_CHECK(out.size(0) == T && idxs.size(0) == T && bad.size(0) == T,
              "token count mismatch");
  size_t sm = (size_t)TILE * LDK * 2 + (size_t)8 * LDP * 2 +
              (2 * 8 * TILE + 24) * 4 + TILE * 4;
  auto st = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(sparse_attn_flat_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  sparse_attn_flat_kernel<<<T, NWARP * 32, sm, st>>>(
      (const bf16*)q.data_ptr(), (const bf16*)bank.data_ptr(),
      (const int64_t*)idxs.data_ptr(), (const bool*)bad.data_ptr(),
      (const float*)sink.data_ptr(), (bf16*)out.data_ptr(), K, (float)scale, h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifndef DSV41_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_attn_flat", &sparse_attn_flat, "fused sparse latent attention",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
#endif
