#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__inline__ __device__ float warp_sum(float v) {
#pragma unroll
    for (int d=16; d; d>>=1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

__device__ __forceinline__ float q8_row_dot(const half *x, const unsigned char *row, int K) {
    float sum=0.f; const int nb=K>>5;
    for (int b=threadIdx.x; b<nb; b+=blockDim.x) {
        const unsigned char *p=row+(size_t)b*34;
        const float d=__half2float(*reinterpret_cast<const half*>(p));
        const signed char *q=reinterpret_cast<const signed char*>(p+2);
        const int k0=b<<5;
#pragma unroll
        for (int j=0;j<32;++j) sum=fmaf(__half2float(x[k0+j]),d*(float)q[j],sum);
    }
    return sum;
}

__device__ __forceinline__ void reduce_store(float sum, half *dst) {
    sum=warp_sum(sum); __shared__ float ws[8];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    if(lane==0) ws[warp]=sum; __syncthreads();
    if(warp==0) {
        float v=lane<(blockDim.x>>5)?ws[lane]:0.f; v=warp_sum(v);
        if(lane==0) *dst=__float2half_rn(v);
    }
}

__global__ void q8_linear_kernel(const half *x,const unsigned char *w,half *y,
                                  int K,int N,int rb) {
    const int n=blockIdx.x,t=blockIdx.y;
    reduce_store(q8_row_dot(x+(size_t)t*K,w+(size_t)n*rb,K),y+(size_t)t*N+n);
}

__global__ void q8_grouped_kernel(const half *x,const unsigned char *w,half *y,
                                   int K,int H,int N,int rb) {
    const int n=blockIdx.x,t=blockIdx.y,h=blockIdx.z,th=t*H+h;
    reduce_store(q8_row_dot(x+(size_t)th*K,w+((size_t)h*N+n)*rb,K),y+(size_t)th*N+n);
}

// Prefill micro-batch, coalesced Q8_0: one warp handles one quant block.
__global__ void q8_linear_outtile8(const half *x,const unsigned char *w,half *y,
                                      int T,int K,int N,int rb) {
    constexpr int ROWS=16, KT=256;
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int n=(int)blockIdx.x*ROWS+warp;
    __shared__ half sx[8][KT];
    float a[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const unsigned char *row=n<N ? w+(size_t)n*rb : w;
    for(int k0=0;k0<K;k0+=KT) {
        const int valid=min(K-k0,KT);
        for(int i=tid;i<valid;i+=blockDim.x) {
#pragma unroll
            for(int t=0;t<8;t++) if(t<T) sx[t][i]=x[(size_t)t*K+k0+i];
        }
        __syncthreads();
        if(n<N) {
            const int nb=(valid+31)>>5;
            for(int b=0;b<nb;b++) {
                const unsigned char *p=row+(size_t)((k0>>5)+b)*34;
                float d=lane==0 ? __half2float(*reinterpret_cast<const half*>(p)) : 0.f;
                d=__shfl_sync(0xffffffff,d,0);
                const float wf=d*(float)reinterpret_cast<const signed char*>(p+2)[lane];
#pragma unroll
                for(int t=0;t<8;t++) if(t<T)
                    a[t]=fmaf(__half2float(sx[t][(b<<5)+lane]),wf,a[t]);
            }
        }
        __syncthreads();
    }
    if(n<N) {
#pragma unroll
        for(int t=0;t<8;t++) if(t<T) {
            float v=warp_sum(a[t]);
            if(lane==0) y[(size_t)t*N+n]=__float2half_rn(v);
        }
    }
}

__global__ void q8_grouped_outtile8(const half *x,const unsigned char *w,half *y,
                                     int T,int K,int H,int N,int rb) {
    constexpr int ROWS=16, KT=256;
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int h=(int)blockIdx.y, n=(int)blockIdx.x*ROWS+warp;
    __shared__ half sx[8][KT];
    float a[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const unsigned char *row=n<N ? w+((size_t)h*N+n)*rb : w;
    for(int k0=0;k0<K;k0+=KT) {
        const int valid=min(K-k0,KT);
        for(int i=tid;i<valid;i+=blockDim.x) {
#pragma unroll
            for(int t=0;t<8;t++) if(t<T)
                sx[t][i]=x[((size_t)t*H+h)*K+k0+i];
        }
        __syncthreads();
        if(n<N) {
            const int nb=(valid+31)>>5;
            for(int b=0;b<nb;b++) {
                const unsigned char *p=row+(size_t)((k0>>5)+b)*34;
                float d=lane==0 ? __half2float(*reinterpret_cast<const half*>(p)) : 0.f;
                d=__shfl_sync(0xffffffff,d,0);
                const float wf=d*(float)reinterpret_cast<const signed char*>(p+2)[lane];
#pragma unroll
                for(int t=0;t<8;t++) if(t<T)
                    a[t]=fmaf(__half2float(sx[t][(b<<5)+lane]),wf,a[t]);
            }
        }
        __syncthreads();
    }
    if(n<N) {
#pragma unroll
        for(int t=0;t<8;t++) if(t<T) {
            float v=warp_sum(a[t]);
            if(lane==0) y[((size_t)t*H+h)*N+n]=__float2half_rn(v);
        }
    }
}


// Experimental large-prefill tile: 8 output rows x 4 tokens, reusing packed weights and input.
__global__ void q8_linear_prefill_t8(const half *x,const unsigned char *w,half *y,
                                     int T,int K,int N,int rb) {
    constexpr int ROWS=8, TT=8, KT=256;
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int n=(int)blockIdx.x*ROWS+warp, t0=(int)blockIdx.y*TT;
    __shared__ half sx[TT][KT];
    float a[TT]={0};
    const unsigned char *row=n<N ? w+(size_t)n*rb : w;
    for(int k0=0;k0<K;k0+=KT) {
        const int valid=min(K-k0,KT);
        for(int i=tid;i<TT*valid;i+=blockDim.x) {
            const int tt=i/valid, kk=i-tt*valid;
            if(t0+tt<T) sx[tt][kk]=x[(size_t)(t0+tt)*K+k0+kk];
        }
        __syncthreads();
        if(n<N) {
            const int nb=(valid+31)>>5;
            for(int b=0;b<nb;b++) {
                const unsigned char *p=row+(size_t)((k0>>5)+b)*34;
                float d=lane==0 ? __half2float(*reinterpret_cast<const half*>(p)) : 0.f;
                d=__shfl_sync(0xffffffff,d,0);
                const float wf=d*(float)reinterpret_cast<const signed char*>(p+2)[lane];
#pragma unroll
                for(int tt=0;tt<TT;tt++) if(t0+tt<T)
                    a[tt]=fmaf(__half2float(sx[tt][(b<<5)+lane]),wf,a[tt]);
            }
        }
        __syncthreads();
    }
    if(n<N) {
#pragma unroll
        for(int tt=0;tt<TT;tt++) if(t0+tt<T) {
            float v=warp_sum(a[tt]);
            if(lane==0) y[(size_t)(t0+tt)*N+n]=__float2half_rn(v);
        }
    }
}

torch::Tensor q8_linear_cuda(torch::Tensor x,torch::Tensor packed,int64_t K64) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int T=x.size(0),K=(int)K64,N=packed.size(0),rb=packed.size(1);
    auto y=torch::empty({T,N},x.options()); auto st=at::cuda::getCurrentCUDAStream();
    if(T>8) q8_linear_prefill_t8<<<dim3((N+7)/8,(T+7)/8),256,0,st>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),packed.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),T,K,N,rb);
    else if(T>1) q8_linear_outtile8<<<(N+15)/16,512,0,st>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),packed.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),T,K,N,rb);
    else q8_linear_kernel<<<dim3(N,T),256,0,st>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),packed.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),K,N,rb);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

torch::Tensor q8_linear_grouped_cuda(torch::Tensor x,torch::Tensor packed,int64_t K64) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int T=x.size(0),H=x.size(1),K=(int)K64,N=packed.size(1),rb=packed.size(2);
    TORCH_CHECK(T<=65535,"T exceeds CUDA grid.y");
    TORCH_CHECK(H<=65535,"H exceeds CUDA grid.z");
    auto y=torch::empty({T,H,N},x.options()); auto st=at::cuda::getCurrentCUDAStream();
    if(T>1&&T<=8) q8_grouped_outtile8<<<dim3((N+15)/16,H),512,0,st>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),packed.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),T,K,H,N,rb);
    else q8_grouped_kernel<<<dim3(N,T,H),256,0,st>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),packed.data_ptr<unsigned char>(),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),K,H,N,rb);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
