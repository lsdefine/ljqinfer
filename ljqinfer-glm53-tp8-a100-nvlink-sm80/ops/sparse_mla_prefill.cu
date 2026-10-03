// Adapted from dsv41f sparse_attn_flat.cu; GLM FP16 paged QK576/PV512.
// Tensor-core warp mapping retained; compression, SWA and sink removed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>

#define DIM 512
#define NWARP 8
#define TILE 32
#define QDIM 576
#define LDK (QDIM + 8)
#define LDP (TILE + 8)
#define LDSCORE (TILE + 8)

typedef __half bf16; // GLM target cache ABI is FP16

__device__ __forceinline__ void mma16816(float* d, const unsigned* a,
                                         const unsigned* b, const float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned pk(const bf16* p) {
  return *(const unsigned*)p;  // two contiguous bf16
}


// Register look-ahead only: no additional shared-memory staging.
__device__ __forceinline__ void load_qk_fragment(
    unsigned* a, unsigned* b, unsigned qa, unsigned ka) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(qa));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      : "=r"(b[0]), "=r"(b[1]) : "r"(ka));
}

template<int FIRST>
__device__ __forceinline__ void qk_register_pipeline(
    float* c, const bf16* qs, const bf16* kvs, int lane, int kg, int keybase) {
  const unsigned qa = __cvta_generic_to_shared(
      qs + (lane & 15)*LDK + kg*288 + FIRST*16 + (lane >> 4)*8);
  const unsigned ka = __cvta_generic_to_shared(
      kvs + (keybase + (lane & 7))*LDK + kg*288 + FIRST*16 + (lane & 8));
  unsigned a[2][4], b[2][2];
  load_qk_fragment(a[0], b[0], qa, ka);
#pragma unroll
  for (int step=0; step<9; ++step) {
    const int current = step & 1;
    if (step < 8)
      load_qk_fragment(a[current ^ 1], b[current ^ 1],
                       qa + (step+1)*32, ka + (step+1)*32);
    mma16816(c, a[current], b[current], c);
  }
}

template<int PAGE>
__global__ __launch_bounds__(NWARP * 32) void sparse_attn_flat_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ bank,
    const int64_t* __restrict__ idxs, const int64_t* __restrict__ pt,
    const int64_t* __restrict__ pos, const int64_t* __restrict__ ctx,
    bf16* __restrict__ out, int K, float scale, int nhead, int runtime_page, int capacity) {
  const int page = PAGE ? PAGE : runtime_page;
  extern __shared__ char smem[];
  bf16* kvs = (bf16*)smem;                  // [TILE][LDK]
  bf16* ps = kvs + TILE * LDK;              // [16][LDP]
  float* spart = (float*)(ps + 16 * LDP);    // [2][16][LDSCORE]
  float* sh_corr = spart + 2 * 16 * LDSCORE;    // [16]
  float* sh_m = sh_corr + 16;                // [16]
  float* sh_l = sh_m + 16;                   // [16]
  int* ok = (int*)(sh_l + 16);               // [TILE]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7  -> head
  const int tig = lane & 3;    // 0..3

  const int kg = w >> 2;             // contraction half (0/1)
  const int keybase = (w & 3) * 8;   // this warp's 8 keys for QK^T
  const int dimbase = w * 64;        // this warp's 64 dims for PV

  bf16* qs = (bf16*)(ok + TILE);
  for (int j=tid; j<16*QDIM; j+=NWARP*32) {
    int h=j/QDIM, d=j%QDIM;
    qs[h*LDK+d] = h<nhead ? q[(((size_t)(h/8)*gridDim.x+token)*8+h%8)*QDIM+d] : __float2half_rn(0.f);
  }
  __syncthreads();

  unsigned qcache[9][4];
#pragma unroll
  for (int step=0; step<9; ++step) {
    const unsigned qa=__cvta_generic_to_shared(qs+(lane&15)*LDK+kg*288+step*16+(lane>>4)*8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(qcache[step][0]), "=r"(qcache[step][1]), "=r"(qcache[step][2]), "=r"(qcache[step][3]) : "r"(qa));
  }
  float acc[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.f;
  float reg_m[2] = {-INFINITY, -INFINITY};
  float reg_l[2] = {0.f, 0.f};

  const int64_t* myidx = idxs + (size_t)token * K;
  const int64_t limit = min(ctx[0], min(pos[token] + 1, (int64_t)capacity));

  // Carry only metadata across tiles; KV shared storage is unchanged.
  const int slot = tid >> 3;
  const int sub = tid & 7;
  const int64_t first_id = slot < K ? myidx[slot] : -1;
  int carry_good = first_id >= 0 && first_id < limit;
  const bf16* carry_src = carry_good
      ? bank + (pt[first_id/page]*page + first_id%page)*(size_t)QDIM : bank;
  for (int kb = 0; kb < K; kb += TILE) {
    __syncthreads();
    // GLM52 df0e3cc mapping: eight lanes/row, nine 16-byte copies/lane.
    {
      const int good = carry_good;
      const bf16* src = carry_src;
      const int bytes = good ? 16 : 0;
#pragma unroll
      for (int group=0; group<2; ++group) {
#pragma unroll
        for (int v=0; v<9; ++v) {
          const int d=(sub+v*8)*8;
          if ((d%288 < 144) == (group == 0)) {
            unsigned dst=__cvta_generic_to_shared(kvs+slot*LDK+d);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src+d),"r"(bytes));
          }
        }
        // All lanes commit both groups, including zero-fill lanes.
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
      if (sub == 0) ok[slot] = good;
    }
    // Issue the next index load before current QK; consume it after QK.
    int64_t next_id = -1;
    if (kb + TILE + slot < K) {
      asm volatile("ld.global.nc.u64 %0, [%1];"
          : "=l"(next_id) : "l"(myidx + kb + TILE + slot));
    }
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();

    {
      // CTA barrier above makes all validity flags visible.
      const int any = __any_sync(0xffffffffu, ok[lane] != 0);
      if (!any) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
      carry_good = next_id >= 0 && next_id < limit;
      carry_src = carry_good
          ? bank + (pt[next_id/page]*page + next_id%page)*(size_t)QDIM : bank;
        continue;
      }
    }

    // ---- QK^T : 16(head) x 8(key) x 288(dim half) ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      unsigned bk[2][2];
      const unsigned ka=__cvta_generic_to_shared(kvs+(keybase+(lane&7))*LDK+kg*288+(lane&8));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(bk[0][0]), "=r"(bk[0][1]) : "r"(ka));
#pragma unroll
      for (int step=0; step<9; ++step) {
        const int current=step&1;
        if (step<8) {
          const unsigned addr=ka+(step+1)*32;
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(bk[current^1][0]), "=r"(bk[current^1][1]) : "r"(addr));
        }
        mma16816(c,qcache[step],bk[current],c);
      }
      // The second copy group is not read until every producer completes.
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      qk_register_pipeline<9>(c, qs, kvs, lane, kg, keybase);
      spart[(kg * 16 + gid) * LDSCORE + keybase + tig * 2] = c[0];
      spart[(kg * 16 + gid) * LDSCORE + keybase + tig * 2 + 1] = c[1];
      spart[(kg*16+gid+8)*LDSCORE+keybase+tig*2]=c[2];
      spart[(kg*16+gid+8)*LDSCORE+keybase+tig*2+1]=c[3];
    }
    __syncthreads();

    // Resolve next page before current softmax/PV.
      carry_good = next_id >= 0 && next_id < limit;
      carry_src = carry_good
          ? bank + (pt[next_id/page]*page + next_id%page)*(size_t)QDIM : bank;

    // ---- online softmax: warp w owns heads w and w+8, lane = key slot ----
    #pragma unroll
    for (int hi=0;hi<2;++hi) {
      const int h=w+hi*8;
      float s = -INFINITY;
      if (ok[lane] && (kb + lane) < K)
        s = (spart[h * LDSCORE + lane] + spart[(16 + h) * LDSCORE + lane]) * scale;
      // Ordered signed encoding preserves finite FP32 comparison and infinities.
      // fmaxf ignores NaNs; canonicalize them to -infinity for the maximum.
      int ordered = __float_as_int(isnan(s) ? -INFINITY : s);
      ordered = ordered < 0 ? (ordered ^ 0x7fffffff) : ordered;
      int maximum = __reduce_max_sync(0xffffffffu, ordered);
      maximum = maximum < 0 ? (maximum ^ 0x7fffffff) : maximum;
      const float mx = __int_as_float(maximum);
      const float mold = reg_m[hi];
      const float mnew = fmaxf(mold, mx);
      const float corr = (mold == -INFINITY) ? 0.f : __expf(mold - mnew);
      float p = (s == -INFINITY) ? 0.f : __expf(s - mnew);
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      reg_m[hi] = mnew;
      ps[h * LDP + lane] = __float2half_rn(p);
      if (lane == 0) {
        sh_corr[h] = corr;
        reg_l[hi] = reg_l[hi] * corr + sum;
      }
    }
    __syncthreads();

    {
      const float corr = sh_corr[gid];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        acc[t][0] *= corr;
        acc[t][1] *= corr;
        acc[t][2] *= sh_corr[gid+8];
        acc[t][3] *= sh_corr[gid+8];
      }
    }

    // ---- PV : 16(head) x 64(dim) x 32(key), V == K ----
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int k0 = ks * 16;
        unsigned a[4], b[2];
        a[0] = pk(ps + gid * LDP + k0 + tig * 2);
        a[1] = pk(ps + (gid+8)*LDP + k0 + tig*2);
        a[2] = pk(ps + gid * LDP + k0 + 8 + tig * 2);
        a[3] = pk(ps + (gid+8)*LDP + k0 + 8 + tig*2);
        // SM80: transpose two 8x8 FP16 tiles directly into the MMA B registers.
        // Lanes 0..15 supply rows k0..k0+15; duplicate valid addresses above lane 15.
        unsigned vaddr = __cvta_generic_to_shared(kvs + (k0 + (lane & 15))*LDK + dimbase + t*8);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b[0]), "=r"(b[1]) : "r"(vaddr));
        mma16816(acc[t], a, b, acc[t]);
      }
    }
  }

  if (lane == 0) { sh_l[w] = reg_l[0]; sh_l[w+8] = reg_l[1]; }
  __syncthreads();
  if (gid < nhead) {
    const float l = sh_l[gid];
    const float inv = l > 0.f ? 1.f / l : 0.f;
    bf16* op = out + ((size_t)token * 8 + gid) * DIM;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int d = dimbase + t * 8 + tig * 2;
      op[d] = __float2half_rn(acc[t][0] * inv);
      op[d + 1] = __float2half_rn(acc[t][1] * inv);
      if(gid+8<nhead) {
        float l2=sh_l[gid+8]; float inv2=l2>0.f?1.f/l2:0.f;
        op[(size_t)gridDim.x*8*DIM+d]=__float2half_rn(acc[t][2]*inv2);
        op[(size_t)gridDim.x*8*DIM+d+1]=__float2half_rn(acc[t][3]*inv2);
      }
    }
  }
}


void forward(at::Tensor q, at::Tensor pool, at::Tensor ids, at::Tensor pt,
             at::Tensor pos, at::Tensor ctx, at::Tensor out, double scale) {
  for (const auto& x : {q,pool,ids,pt,pos,ctx,out}) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "contiguous CUDA tensors required");
    TORCH_CHECK(x.device()==q.device(), "device mismatch");
  }
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.dim()==3 && q.size(2)==576 && q.size(1)==8 && q.size(0)%2==0,"q [2*T,8,576] paired receive layout");
  int T=q.size(0)/2, H=16;
  TORCH_CHECK(pool.dim()==3 && pool.size(2)==576 && pool.size(1)>0,"pool [P,S,576]");
  TORCH_CHECK(ids.dim()==2 && ids.size(0)==T,"ids [T,K]");
  TORCH_CHECK(out.dim()==3 && out.size(0)==2*T && out.size(1)==8 && out.size(2)==512,"out [2*T,8,512] paired send layout");
  TORCH_CHECK(pt.dim()==1 && pos.dim()==1 && pos.numel()==T && ctx.numel()==1,"page/position/context shape");
  TORCH_CHECK(q.scalar_type()==at::kHalf && pool.scalar_type()==at::kHalf && out.scalar_type()==at::kHalf,"FP16 cache ABI");
  TORCH_CHECK(ids.scalar_type()==at::kLong && pt.scalar_type()==at::kLong && pos.scalar_type()==at::kLong && ctx.scalar_type()==at::kLong,"int64 metadata");
  size_t sm=TILE*LDK*2+16*LDP*2+(2*16*LDSCORE+48)*4+TILE*4+16*LDK*2;
  if (!T) return;
  if (pool.size(1)==64) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_attn_flat_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  sparse_attn_flat_kernel<64><<<T,256,sm,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)pool.data_ptr(),ids.data_ptr<int64_t>(),
    pt.data_ptr<int64_t>(),pos.data_ptr<int64_t>(),ctx.data_ptr<int64_t>(),
    (bf16*)out.data_ptr(),ids.size(1),scale,H,pool.size(1),pt.numel()*pool.size(1));
  } else {
  C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_attn_flat_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  sparse_attn_flat_kernel<0><<<T,256,sm,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)pool.data_ptr(),ids.data_ptr<int64_t>(),
    pt.data_ptr<int64_t>(),pos.data_ptr<int64_t>(),ctx.data_ptr<int64_t>(),
    (bf16*)out.data_ptr(),ids.size(1),scale,H,pool.size(1),pt.numel()*pool.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward", &forward);}
