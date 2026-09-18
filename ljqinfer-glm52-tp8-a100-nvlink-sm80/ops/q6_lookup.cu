#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <mutex>

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

// Eight warps compute eight vocabulary rows per block.  Each lane decodes eight
// values from every 256-wide Q6_K super-block.
__global__ void q6_matvec_kernel(const uint8_t *w, const __half *x, float *y,
                                 int nb, int rows) {
    const int t = blockIdx.y;
    x += (size_t)t * nb * 256;
    y += (size_t)t * rows;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q = w + ((size_t)row * nb + b) * 210;
        const float d = __half2float(*reinterpret_cast<const __half *>(q + 208));
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            #pragma unroll
            for (int ip = 0; ip < 2; ++ip) {
                const int j = ip * 128 + part * 32 + lane;
                const uint8_t lo = q[ip * 64 + lane + ((part & 1) ? 32 : 0)];
                const uint8_t hi = q[128 + ip * 32 + lane];
                const int v = ((part < 2 ? lo & 15 : lo >> 4) |
                               (((hi >> (2 * part)) & 3) << 4)) - 32;
                const int8_t s = (int8_t)q[192 + ip * 8 + lane / 16 + 2 * part];
                sum += d * (float)s * (float)v * __half2float(x[b * 256 + j]);
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void q6_dequant_fp16_kernel(const uint8_t *w, __half *out, int nb) {
    const int row = blockIdx.x, b = blockIdx.y, j = threadIdx.x;
    const uint8_t *q = w + ((size_t)row * nb + b) * 210;
    const int ip = j >> 7, p = j & 127, part = p >> 5, lane = p & 31;
    const uint8_t lo = q[ip * 64 + lane + ((part & 1) ? 32 : 0)];
    const uint8_t hi = q[128 + ip * 32 + lane];
    const int v = ((part < 2 ? lo & 15 : lo >> 4) |
                   (((hi >> (2 * part)) & 3) << 4)) - 32;
    const int8_t s = (int8_t)q[192 + ip * 8 + lane / 16 + 2 * part];
    const float d = __half2float(*reinterpret_cast<const __half *>(q + 208));
    out[((size_t)row * nb + b) * 256 + j] = __float2half(d * (float)s * (float)v);
}

void q6_dequant_fp16_out_cuda(torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK(w.is_cuda() && out.is_cuda() && w.device() == out.device(),
                "packed and out must be on the same CUDA device");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8 && out.scalar_type() == torch::kFloat16,
                "expected uint8 packed and float16 out");
    TORCH_CHECK(w.is_contiguous() && out.is_contiguous() && w.dim() == 3 &&
                w.size(2) == 210 && out.dim() == 2 && out.size(0) == w.size(0) &&
                out.size(1) == w.size(1) * 256, "bad Q6_K layout/output shape");
    c10::cuda::CUDAGuard g(w.device());
    q6_dequant_fp16_kernel<<<dim3(w.size(0), w.size(1)), 256, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        w.data_ptr<uint8_t>(), reinterpret_cast<__half *>(out.data_ptr<at::Half>()),
        w.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Dedicated per-device cuBLAS handle with a permanently resident workspace.
// Rationale: the shared torch handle gets its workspace cleared on every
// CUDA graph capture_begin, so a capture-time allocation from the shared
// private graph pool gets baked into the graph, then freed and reused by
// later captures for resident buffers -> replay-time memory corruption.
// A private handle + cudaMalloc'd workspace has a fixed address outside
// any allocator pool for the process lifetime.
static cublasHandle_t q6_gemm_handle(int device) {
    constexpr size_t kWorkspaceBytes = 32u * 1024u * 1024u;
    static cublasHandle_t handles[64] = {};
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!handles[device]) {
        cublasHandle_t h = nullptr;
        TORCH_CUDABLAS_CHECK(cublasCreate(&h));
        void *ws = nullptr;
        C10_CUDA_CHECK(cudaMalloc(&ws, kWorkspaceBytes));
        TORCH_CUDABLAS_CHECK(cublasSetWorkspace(h, ws, kWorkspaceBytes));
        handles[device] = h;
    }
    return handles[device];
}

void q6_gemm_fp16_out_cuda(torch::Tensor w, torch::Tensor x,
                              torch::Tensor out) {
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && out.is_cuda() &&
                w.device() == x.device() && w.device() == out.device(),
                "w, x, and out must be on the same CUDA device");
    TORCH_CHECK(w.scalar_type() == torch::kFloat16 &&
                x.scalar_type() == torch::kFloat16 &&
                out.scalar_type() == torch::kFloat32,
                "w/x must be FP16 and out must be FP32");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.is_contiguous(),
                "w, x, and out must be contiguous");
    TORCH_CHECK(w.dim() == 2 && x.dim() == 2 && out.dim() == 2 &&
                x.size(1) == w.size(1) && out.size(0) == x.size(0) &&
                out.size(1) == w.size(0), "invalid GEMM shapes");
    c10::cuda::CUDAGuard guard(w.device());
    const int m = static_cast<int>(x.size(0));
    const int n = static_cast<int>(w.size(0));
    const int k = static_cast<int>(w.size(1));
    const float alpha = 1.0f, beta = 0.0f;
    auto handle = q6_gemm_handle(w.get_device());
    TORCH_CUDABLAS_CHECK(cublasSetStream(
        handle, at::cuda::getCurrentCUDAStream()));
    // Row-major out[m,n] = x[m,k] * w[n,k]^T. The same storage is
    // column-major out[n,m] = opT(w[k,n]) * x[k,m] to cuBLAS.
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, w.data_ptr<at::Half>(), CUDA_R_16F, k,
        x.data_ptr<at::Half>(), CUDA_R_16F, k,
        &beta, out.data_ptr<float>(), CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

torch::Tensor q6_matvec_cuda(torch::Tensor w, torch::Tensor x) {
    TORCH_CHECK(w.is_cuda() && x.is_cuda() && w.device() == x.device(),
                "packed and x must be on the same CUDA device");
    TORCH_CHECK(w.scalar_type() == torch::kUInt8 && x.scalar_type() == torch::kFloat16,
                "expected uint8 packed and float16 x");
    TORCH_CHECK(w.is_contiguous() && x.is_contiguous() &&
                w.dim() == 3 && w.size(2) == 210 && (x.dim() == 1 || x.dim() == 2),
                "bad Q6_K layout/input rank");
    const int64_t K = w.size(1) * 256;
    TORCH_CHECK(x.size(-1) == K, "x width must equal packed K");
    const int64_t T = x.dim() == 1 ? 1 : x.size(0);
    TORCH_CHECK(T >= 1 && T <= 65535, "unsupported token count");
    c10::cuda::CUDAGuard g(w.device());
    auto y = x.dim() == 1
        ? torch::empty({w.size(0)}, w.options().dtype(torch::kFloat32))
        : torch::empty({T, w.size(0)}, w.options().dtype(torch::kFloat32));
    q6_matvec_kernel<<<dim3((w.size(0) + 7) / 8, T), 256, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        w.data_ptr<uint8_t>(), reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        y.data_ptr<float>(), w.size(1), w.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Grid is (nb, min(nt, 65535)).  gridDim.y is capped by the hardware limit of
// 65535, so the y dimension strides over tokens.  For nt <= 65535 the stride
// loop runs exactly one iteration per block and the block-to-token mapping is
// identical to a plain blockIdx.y indexing, preserving the original behaviour.
__global__ void kernel(const uint8_t*w,const int*ids,__nv_bfloat16*y,long long nt,int nb,int vocab){
    const int b=blockIdx.x,j=threadIdx.x;
    if(j>=256)return;
    for(long long t=blockIdx.y;t<nt;t+=gridDim.y){
        const int row=ids[t];
        if((unsigned)row>=(unsigned)vocab)continue;
        const uint8_t*q=w+((size_t)row*nb+b)*210;
        int ip=j>>7,p=j&127,part=p>>5,l=p&31;
        uint8_t lo=q[ip*64+l+((part&1)?32:0)],hi=q[128+ip*32+l];
        int v=((part<2?lo&15:lo>>4)|(((hi>>(2*part))&3)<<4))-32;
        int8_t s=(int8_t)q[192+ip*8+l/16+2*part];
        float d=__half2float(*reinterpret_cast<const __half*>(q+208));
        y[((size_t)t*nb+b)*256+j]=__float2bfloat16(d*s*v);
    }
}
torch::Tensor q6_lookup_cuda(torch::Tensor w,torch::Tensor ids){TORCH_CHECK(w.is_cuda()&&ids.is_cuda()&&w.scalar_type()==torch::kUInt8&&ids.scalar_type()==torch::kInt32,"bad device/dtype");TORCH_CHECK(w.is_contiguous()&&ids.is_contiguous()&&w.dim()==3&&w.size(2)==210,"bad Q6_K layout");c10::cuda::CUDAGuard g(w.device());auto y=torch::empty({ids.numel(),w.size(1)*256},w.options().dtype(torch::kBFloat16));const long long nt=ids.numel();if(nt==0)return y;const unsigned int gy=(unsigned int)std::min<long long>(nt,65535LL);kernel<<<dim3((unsigned int)w.size(1),gy),256,0,at::cuda::getCurrentCUDAStream()>>>(w.data_ptr<uint8_t>(),ids.data_ptr<int>(),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),nt,w.size(1),w.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
