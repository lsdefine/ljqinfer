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

template<int HEADS,int PAGE>
__global__ __launch_bounds__(NWARP * 32) void sparse_attn_flat_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ bank,
    const int64_t* __restrict__ idxs, const int64_t* __restrict__ pt,
    const int64_t* __restrict__ pos, const int64_t* __restrict__ ctx,
    bf16* __restrict__ out, int K, float scale, int runtime_heads, int runtime_page, int capacity) {
  const int nhead=HEADS ? HEADS : runtime_heads;
  const int page=PAGE ? PAGE : runtime_page;
  extern __shared__ char smem[];
  bf16* kvs = (bf16*)smem;                  // [TILE][LDK]
  bf16* ps = kvs + TILE * LDK;              // [16][LDP]
  float* spart = (float*)(ps + 16 * LDP);    // [2][16][TILE]
  float* sh_corr = spart + 2 * 16 * TILE;    // [16]
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

  unsigned qa[18][4];
  {
    const bf16* qp = q + ((size_t)token * nhead + gid) * QDIM + kg * 288;
    const bool live = (gid < nhead);
#pragma unroll
    for (int ks = 0; ks < 18; ++ks) {
      qa[ks][0] = live ? pk(qp + ks * 16 + tig * 2) : 0u;
      qa[ks][1] = live ? pk(qp + ks * 16 + tig * 2 + 8) : 0u;
      qa[ks][2] = gid+8<nhead ? pk(qp + 8*QDIM + ks*16 + tig*2) : 0u;
      qa[ks][3] = gid+8<nhead ? pk(qp + 8*QDIM + ks*16 + tig*2+8) : 0u;
    }
  }

  float acc[8][4];
#pragma unroll
  for (int t = 0; t < 8; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[t][j] = 0.f;
  if (tid < 16) { sh_m[tid] = -INFINITY; sh_l[tid] = 0.f; }

  const int64_t* myidx = idxs + (size_t)token * K;
  const int64_t limit = min(ctx[0], min(pos[token] + 1, (int64_t)capacity));

  __shared__ unsigned live_tiles[64];
  if(K<=2048){
    for(int tile=w;tile<(K+31)/32;tile+=NWARP){
      int j=tile*32+lane;
      int64_t id=j<K?myidx[j]:-1;
      unsigned mask=__ballot_sync(0xffffffff,id>=0 && id<limit);
      if(lane==0)live_tiles[tile]=mask;
    }
    __syncthreads();
  }
  for (int kb = 0; kb < K; kb += TILE) {
    // IDs need not be sorted or compact: inspect every tile, never break.
    if(K<=2048){
      if(!live_tiles[kb/32])continue;
      __syncthreads(); // previous PV must finish before overwriting shared KV
    }else{
      const int64_t probe = tid < TILE && kb + tid < K ? myidx[kb + tid] : -1;
      if (!__syncthreads_or(probe >= 0 && probe < limit)) continue;
    }
    for (int slot = w; slot < TILE; slot += NWARP) {
      const int kk = kb + slot;
      int good = 0;
      const bf16* src = nullptr;
      const int64_t id = kk < K ? myidx[kk] : -1;
      if (id >= 0 && id < limit) {
        src = bank + (pt[id/page] * page + id%page) * (size_t)QDIM;
        good = 1;
      }
      {
        const bf16* safe = good ? src : bank;
        const int bytes = good ? 16 : 0;
#pragma unroll
        for(int j=0;j<2;++j) {
          unsigned dst=__cvta_generic_to_shared(kvs+slot*LDK+j*256+lane*8);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(safe+j*256+lane*8),"r"(bytes));
        }
        if(lane<8) {
          unsigned dst=__cvta_generic_to_shared(kvs+slot*LDK+512+lane*8);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(safe+512+lane*8),"r"(bytes));
        }
      }
      if (lane == 0) ok[slot] = good;
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
    __syncthreads();

    // ---- QK^T : 16(head) x 8(key) x 288(dim half) ----
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const bf16* kp = kvs + (keybase + gid) * LDK + kg * 288;
#pragma unroll 4
      for (int ks = 0; ks < 18; ++ks) {
        unsigned a[4] = {qa[ks][0], qa[ks][2], qa[ks][1], qa[ks][3]};
        unsigned b[2];
        b[0] = pk(kp + ks * 16 + tig * 2);
        b[1] = pk(kp + ks * 16 + tig * 2 + 8);
        mma16816(c, a, b, c);
      }
      spart[(kg * 16 + gid) * TILE + keybase + tig * 2] = c[0];
      spart[(kg * 16 + gid) * TILE + keybase + tig * 2 + 1] = c[1];
      spart[(kg*16+gid+8)*TILE+keybase+tig*2]=c[2];
      spart[(kg*16+gid+8)*TILE+keybase+tig*2+1]=c[3];
    }
    __syncthreads();

    // ---- online softmax: warp w owns heads w and w+8, lane = key slot ----
    for (int h=w;h<16;h+=8) {
      float s = -INFINITY;
      if (ok[lane] && (kb + lane) < K)
        s = (spart[h * TILE + lane] + spart[(16 + h) * TILE + lane]) * scale;
      float mx = s;
#pragma unroll
      for (int o = 16; o; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mold = sh_m[h];
      const float mnew = fmaxf(mold, mx);
      const float corr = (mold == -INFINITY) ? 0.f : __expf(mold - mnew);
      float p = (s == -INFINITY) ? 0.f : __expf(s - mnew);
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      ps[h * LDP + lane] = __float2half_rn(p);
      if (lane == 0) {
        sh_corr[h] = corr;
        sh_m[h] = mnew;
        sh_l[h] = sh_l[h] * corr + sum;
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
      const int dcol = dimbase + t * 8 + gid;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int k0 = ks * 16;
        unsigned a[4], b[2];
        a[0] = pk(ps + gid * LDP + k0 + tig * 2);
        a[1] = pk(ps + (gid+8)*LDP + k0 + tig*2);
        a[2] = pk(ps + gid * LDP + k0 + 8 + tig * 2);
        a[3] = pk(ps + (gid+8)*LDP + k0 + 8 + tig*2);
        const bf16* vp = kvs + (k0 + tig * 2) * LDK + dcol;
        b[0] = *(unsigned*)&__halves2half2(vp[0], vp[LDK]);
        const bf16* vq = vp + 8 * LDK;
        b[1] = *(unsigned*)&__halves2half2(vq[0], vq[LDK]);
        mma16816(acc[t], a, b, acc[t]);
      }
    }
  }

  __syncthreads();
  if (gid < nhead) {
    const float l = sh_l[gid];
    const float inv = l > 0.f ? 1.f / l : 0.f;
    bf16* op = out + ((size_t)token * nhead + gid) * DIM;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int d = dimbase + t * 8 + tig * 2;
      op[d] = __float2half_rn(acc[t][0] * inv);
      op[d + 1] = __float2half_rn(acc[t][1] * inv);
      if(gid+8<nhead) {
        float l2=sh_l[gid+8]; float inv2=l2>0.f?1.f/l2:0.f;
        op[8*DIM+d]=__float2half_rn(acc[t][2]*inv2);
        op[8*DIM+d+1]=__float2half_rn(acc[t][3]*inv2);
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
  TORCH_CHECK(q.dim()==3 && q.size(2)==576 && q.size(1)>0 && q.size(1)<=16,"q [T,H<=16,576]");
  int T=q.size(0), H=q.size(1);
  TORCH_CHECK(pool.dim()==3 && pool.size(2)==576 && pool.size(1)>0,"pool [P,S,576]");
  TORCH_CHECK(ids.dim()==2 && ids.size(0)==T,"ids [T,K]");
  TORCH_CHECK(out.dim()==3 && out.size(0)==T && out.size(1)==H && out.size(2)==512,"out [T,H,512]");
  TORCH_CHECK(pt.dim()==1 && pos.dim()==1 && pos.numel()==T && ctx.numel()==1,"page/position/context shape");
  TORCH_CHECK(q.scalar_type()==at::kHalf && pool.scalar_type()==at::kHalf && out.scalar_type()==at::kHalf,"FP16 cache ABI");
  TORCH_CHECK(ids.scalar_type()==at::kLong && pt.scalar_type()==at::kLong && pos.scalar_type()==at::kLong && ctx.scalar_type()==at::kLong,"int64 metadata");
  size_t sm=TILE*LDK*2+16*LDP*2+(2*16*TILE+48)*4+TILE*4;
  if (!T) return;
  if(H==8 && pool.size(1)==64){
  sparse_attn_flat_kernel<8,64><<<T,256,sm,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)pool.data_ptr(),ids.data_ptr<int64_t>(),
    pt.data_ptr<int64_t>(),pos.data_ptr<int64_t>(),ctx.data_ptr<int64_t>(),
    (bf16*)out.data_ptr(),ids.size(1),scale,H,pool.size(1),pt.numel()*pool.size(1));
  }else{
  sparse_attn_flat_kernel<0,0><<<T,256,sm,at::cuda::getCurrentCUDAStream()>>>(
    (const bf16*)q.data_ptr(),(const bf16*)pool.data_ptr(),ids.data_ptr<int64_t>(),
    pt.data_ptr<int64_t>(),pos.data_ptr<int64_t>(),ctx.data_ptr<int64_t>(),
    (bf16*)out.data_ptr(),ids.size(1),scale,H,pool.size(1),pt.numel()*pool.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward", &forward);}
