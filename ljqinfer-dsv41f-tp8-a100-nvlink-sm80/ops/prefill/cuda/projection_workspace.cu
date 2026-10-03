#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ float to_float(float x) {return x;}
__device__ float to_float(__nv_bfloat16 x) {return __bfloat162float(x);}

template<class T> __global__ void quant(const T* x,float* y,int64_t count) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
        float v=to_float(x[i]),mx=fabsf(v);
        for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
        float scale=exp2f(ceilf(log2f(fmaxf(mx,1.e-4f)/448.f)));
        __nv_fp8_e4m3 q(fminf(448.f,fmaxf(-448.f,v/scale)));
        y[i]=float(q)*scale;
    }
}
torch::Tensor activation(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(),"contiguous CUDA input required");
    TORCH_CHECK(x.dim()>0 && x.size(-1)>0 && x.size(-1)%32==0,"K32 geometry");
    TORCH_CHECK(x.scalar_type()==torch::kBFloat16 || x.scalar_type()==torch::kFloat32,"BF16/FP32 input required");
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(y.device()==x.device() && y.scalar_type()==torch::kFloat32 && y.is_contiguous() && y.sizes()==x.sizes(), "FP32 output geometry/device/contiguity");
    TORCH_CHECK(!y.is_alias_of(x), "output must not alias input");
    if(!x.numel())return y;
    int blocks=int(std::min<int64_t>((x.numel()+255)/256,65535));
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    if(x.scalar_type()==torch::kBFloat16)
        quant<<<blocks,256,0,stream>>>((const __nv_bfloat16*)x.data_ptr(),y.data_ptr<float>(),x.numel());
    else quant<<<blocks,256,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}

// Preserve the reference M,Ntile,K GEMM geometry and final BF16 rounding.
// Only preparation/unpack is fused; the existing ATen FP32 GEMM is retained.
template<bool FP4> __global__ void unpack_tile(const unsigned char* w,const unsigned char* scales,float* out,int K,int start,int rows) {
 const float lut[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
 int64_t count=int64_t(rows)*K;
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x){
  int row=start+i/K,col=i%K;float v;
  if(FP4){int code=(w[int64_t(row)*(K/2)+col/2]>>((col&1)*4))&15;v=lut[code&7];if(code&8)v=-v;}
  else {__nv_fp8_e4m3 t;t.__x=w[int64_t(row)*K+col];v=float(t);}
  int sr=FP4?row:row/32;
  float scale=exp2f(float(scales[int64_t(sr)*(K/32)+col/32])-127.f);
  out[i]=v*scale;
 }
}
torch::Tensor projection(torch::Tensor x,torch::Tensor w,torch::Tensor s,torch::Tensor abuf,torch::Tensor wbuf,torch::Tensor pbuf,torch::Tensor output){
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kBFloat16 && x.dim()==2 && x.is_contiguous(),"contiguous CUDA BF16 matrix required");
 TORCH_CHECK(w.device()==x.device() && s.device()==x.device() && w.dim()==2 && s.dim()==2 && w.is_contiguous() && s.is_contiguous(),"rank-local contiguous matrices required");
 bool fp4=w.scalar_type()==torch::kInt8;
 TORCH_CHECK(fp4 || w.scalar_type()==torch::kFloat8_e4m3fn,"FP4/FP8 weight required");
 int64_t K=x.size(1),N=w.size(0),M=x.size(0);
 TORCH_CHECK(K>0 && K%32==0 && K<2147483647 && N>0 && N<2147483647 && w.size(1)*(fp4?2:1)==K,"geometry");
 TORCH_CHECK(s.scalar_type()==torch::kUInt8 && s.size(0)==(fp4?N:(N+31)/32) && s.size(1)==K/32,"scale ABI");
 c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream().stream();
 int64_t tile=std::min<int64_t>(256,N);
 for(auto buf : {abuf,wbuf,pbuf}) {
  TORCH_CHECK(buf.device()==x.device() && buf.scalar_type()==torch::kFloat32 && buf.is_contiguous() && buf.dim()==1,"flat FP32 workspace required");
  TORCH_CHECK(!buf.is_alias_of(x) && !buf.is_alias_of(w) && !buf.is_alias_of(s) && !buf.is_alias_of(output),"workspace alias");
 }
 TORCH_CHECK(!abuf.is_alias_of(wbuf) && !abuf.is_alias_of(pbuf) && !wbuf.is_alias_of(pbuf),"workspace overlap");
 TORCH_CHECK(abuf.numel()>=M*K && wbuf.numel()>=tile*K && pbuf.numel()>=M*tile,"workspace capacity");
 TORCH_CHECK(output.device()==x.device() && output.scalar_type()==torch::kBFloat16 && output.is_contiguous() && output.dim()==2 && output.size(0)==M && output.size(1)==N,"output geometry");
 TORCH_CHECK(!output.is_alias_of(x) && !output.is_alias_of(w) && !output.is_alias_of(s),"output alias");
 auto a=abuf.narrow(0,0,M*K).view({M,K});activation(x,a);
 auto scratch=wbuf.narrow(0,0,tile*K).view({tile,K});
 for(int start=0;start<N;start+=256){
  int n=std::min<int64_t>(256,N-start);auto wt=scratch.narrow(0,0,n);
  int blocks=int(std::min<int64_t>((int64_t(n)*K+255)/256,65535));
  if(fp4)unpack_tile<true><<<blocks,256,0,stream>>>((const unsigned char*)w.data_ptr(),s.data_ptr<unsigned char>(),wt.data_ptr<float>(),K,start,n);
  else unpack_tile<false><<<blocks,256,0,stream>>>((const unsigned char*)w.data_ptr(),s.data_ptr<unsigned char>(),wt.data_ptr<float>(),K,start,n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto partial=pbuf.narrow(0,0,M*n).view({M,n});
  at::mm_out(partial,a,wt.t());
  output.narrow(1,start,n).copy_(partial);
 }
 return output;
}
// fp8(e4m3 + e8m0 32x32 block scale) x bf16 GEMM on A100 (sm80) via tensor cores.
// y[M,N] = x[M,K] @ w[N,K]^T   (w stored row-major as [N,K] e4m3)
// A single code path for every M so the numerics never depend on the batch width.

#define KS 128            // K chunk per iteration
#define NT 1             // n8 tiles per warp  -> 8 n per warp
#define NW 4             // warps per block
#define NPB (NT * 8 * NW)  // 64 n per block
#define LDW 136           // shared row stride in bf16 elements (pad to dodge bank conflicts)
#define FIXC 1.32922799578e36f  // 2^120, folds the e4m3->bf16 exponent bias gap

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// 16 e4m3 bytes -> 16 bf16 (8 packed pairs), scaled by f.
__device__ __forceinline__ void unpack16(uint4 v, __nv_bfloat162 *out, __nv_bfloat162 f) {
  const uint32_t *p = reinterpret_cast<const uint32_t *>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t a = prmt(p[i], 0u, 0x4140u);
    uint32_t b = prmt(p[i], 0u, 0x4342u);
    a = ((a & 0x007f007fu) << 4) | ((a & 0x00800080u) << 8);
    b = ((b & 0x007f007fu) << 4) | ((b & 0x00800080u) << 8);
    out[2 * i] = __hmul2(*reinterpret_cast<__nv_bfloat162 *>(&a), f);
    out[2 * i + 1] = __hmul2(*reinterpret_cast<__nv_bfloat162 *>(&b), f);
  }
}

template <int MT>
__global__ void gemm_fp8_tc(const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w,
                            const uint8_t *__restrict__ sc, float *__restrict__ part,
                            int M, int N, int K, int sK, int BPS) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16 *xs = smem;                     // MT*16 rows x LDW
  __nv_bfloat16 *ws = smem + MT * 16 * LDW;     // NW*8 rows x LDW

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wn0 = blockIdx.x * NPB + warp * 8;
  __nv_bfloat16 *wsw = ws + warp * 8 * LDW;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  const int nr = lane >> 2, half = lane & 3;  // weight loader mapping: 16B per lane

  // Registers holding the *next* K chunk: the global loads are issued before the
  // mma work of the current chunk so DRAM latency hides behind the tensor cores.
  uint4 wpre, wpre2;
  float spre;
  uint4 xpre[2 * MT];
  const uint8_t *wrow = w + (size_t)(wn0 + nr) * K + half * 32;
  const uint8_t *srow = sc + (size_t)((wn0 + nr) >> 5) * sK;

#define LOADW(kk)                                                                        \
  {                                                                                      \
    wpre = *reinterpret_cast<const uint4 *>(wrow + (kk));                                \
    wpre2 = *reinterpret_cast<const uint4 *>(wrow + (kk) + 16);                          \
    spre = __int_as_float((uint32_t)srow[((kk) + half * 32) >> 5] << 23);                \
  }
#define LOADX(kk)                                                                        \
  {                                                                                      \
    _Pragma("unroll") for (int u = 0; u < 2 * MT; ++u) {                                     \
      int r = (tid + u * NW * 32) / (KS / 8);                                            \
      int c = ((tid + u * NW * 32) % (KS / 8)) * 8;                                      \
      xpre[u] = make_uint4(0u, 0u, 0u, 0u);                                              \
      if (r < M) xpre[u] = *reinterpret_cast<const uint4 *>(x + (size_t)r * K + (kk) + c); \
    }                                                                                    \
  }

  const int kblk = K / KS;
  const int sb = blockIdx.y * BPS, se = min(sb + BPS, kblk);
  LOADW(sb * KS) LOADX(sb * KS)

  for (int kb = sb; kb < se; ++kb) {
    const int k0 = kb * KS;
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 2 * MT; ++u) {
      int r = (tid + u * NW * 32) / (KS / 8);
      int c = ((tid + u * NW * 32) % (KS / 8)) * 8;
      *reinterpret_cast<uint4 *>(xs + r * LDW + c) = xpre[u];
    }
    {
      __nv_bfloat162 f = __float2bfloat162_rn(FIXC * spre);
      __nv_bfloat162 o[16];
      unpack16(wpre, o, f);
      unpack16(wpre2, o + 8, f);
      __nv_bfloat16 *dst = wsw + nr * LDW + half * 32;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        *reinterpret_cast<uint4 *>(dst + q * 8) = *(reinterpret_cast<uint4 *>(o) + q);
    }
    __syncthreads();
    if (kb + 1 < se) { LOADW((kb + 1) * KS) LOADX((kb + 1) * KS) }

#pragma unroll
    for (int ks = 0; ks < KS; ks += 16) {
      uint32_t b[NT][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const __nv_bfloat16 *bp = wsw + (j * 8 + g) * LDW + ks + 2 * t;
        b[j][0] = *reinterpret_cast<const uint32_t *>(bp);
        b[j][1] = *reinterpret_cast<const uint32_t *>(bp + 8);
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const __nv_bfloat16 *ap = xs + (i * 16 + g) * LDW + ks + 2 * t;
        uint32_t a0 = *reinterpret_cast<const uint32_t *>(ap);
        uint32_t a1 = *reinterpret_cast<const uint32_t *>(ap + 8 * LDW);
        uint32_t a2 = *reinterpret_cast<const uint32_t *>(ap + 8);
        uint32_t a3 = *reinterpret_cast<const uint32_t *>(ap + 8 * LDW + 8);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[j][0]), "r"(b[j][1]));
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      int nb = wn0 + j * 8 + 2 * t;
      int r0 = i * 16 + g, r1 = r0 + 8;
      float *po = part + (size_t)blockIdx.y * M * N;
      if (r0 < M) {
        po[(size_t)r0 * N + nb] = acc[i][j][0];
        po[(size_t)r0 * N + nb + 1] = acc[i][j][1];
      }
      if (r1 < M) {
        po[(size_t)r1 * N + nb] = acc[i][j][2];
        po[(size_t)r1 * N + nb + 1] = acc[i][j][3];
      }
    }
}

__global__ void reduce_k(const float *__restrict__ part, __nv_bfloat16 *__restrict__ y,
                         int total, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float s = 0.f;
  for (int q = 0; q < S; ++q) s += part[(size_t)q * total + i];
  y[i] = __float2bfloat16(s);
}

torch::Tensor fp8_dense(torch::Tensor x, torch::Tensor w, torch::Tensor sc) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sc.is_cuda(), "cuda only");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(N % NPB == 0 && K % KS == 0, "shape not tiled");
  const int sK = sc.size(1);
  auto y = torch::empty({M, N}, x.options());

  // K-split factor depends only on the shape (never on M), so the reduction
  // order -- and therefore the numerics -- is identical for every batch width.
  const int kblk = K / KS;
  int S = 1;
  for (int c = 8; c >= 1; --c)
    if (kblk % c == 0) { S = c; break; }
  const int BPS = kblk / S;
  auto part = torch::empty({S, M, N}, x.options().dtype(torch::kFloat32));
  const int MT = (M + 15) / 16;
  TORCH_CHECK(MT <= 2, "M too large for this tile");
  const int smem = (MT * 16 + NW * 8) * LDW * (int)sizeof(__nv_bfloat16);
  dim3 blocks(N / NPB, S), threads(NW * 32);
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr());
  auto wp = reinterpret_cast<const uint8_t *>(w.data_ptr());
  auto sp = reinterpret_cast<const uint8_t *>(sc.data_ptr());
  auto yp = reinterpret_cast<__nv_bfloat16 *>(y.data_ptr());
  auto pp = part.data_ptr<float>();
  if (MT == 1)
    gemm_fp8_tc<1><<<blocks, threads, smem, st>>>(xp, wp, sp, pp, M, N, K, sK, BPS);
  else
    gemm_fp8_tc<2><<<blocks, threads, smem, st>>>(xp, wp, sp, pp, M, N, K, sK, BPS);
  const int total = M * N;
  reduce_k<<<(total + 255) / 256, 256, 0, st>>>(pp, yp, total, S);
  return y;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("projection",&projection);m.def("activation",&activation);m.def("fp8_dense",&fp8_dense);}
