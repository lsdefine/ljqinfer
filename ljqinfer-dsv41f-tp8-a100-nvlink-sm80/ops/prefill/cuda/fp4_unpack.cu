#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "prefill_moe_cutlass_gemm.h"

__global__ void unpack_kernel(const unsigned char* w, const unsigned char* s,
                              __nv_bfloat16* out, int64_t n) {
  const float table[8] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  for (int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=(int64_t)blockDim.x*gridDim.x) {
    int code = (w[i/2] >> ((i&1)*4)) & 15;
    float v = table[code&7];
    if (code&8) v = -v;
    out[i] = __float2bfloat16_rn(ldexpf(v, (int)s[i/32]-127));
  }
}

torch::Tensor unpack_fp4(torch::Tensor w, torch::Tensor s) {
  TORCH_CHECK(w.is_cuda() && s.is_cuda() && w.device()==s.device(), "same CUDA device required");
  TORCH_CHECK(w.scalar_type()==torch::kInt8 && s.scalar_type()==torch::kUInt8, "FP4 int8 / E8M0 uint8 required");
  TORCH_CHECK(w.dim()==3 && s.dim()==3 && w.is_contiguous() && s.is_contiguous(), "contiguous [E,N,K/2] and [E,N,K/32] required");
  TORCH_CHECK(w.size(2)%16==0 && s.size(0)==w.size(0) && s.size(1)==w.size(1) && s.size(2)*16==w.size(2), "scale geometry");
  c10::cuda::CUDAGuard guard(w.device());
  auto out = torch::empty({w.size(0),w.size(1),w.size(2)*2},w.options().dtype(torch::kBFloat16));
  if (out.numel()) {
    int blocks = (int)std::min<int64_t>((out.numel()+255)/256,65535);
    unpack_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
      (const unsigned char*)w.data_ptr(),s.data_ptr<unsigned char>(),(__nv_bfloat16*)out.data_ptr(),out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("unpack_fp4",&unpack_fp4);
  m.def("grouped_gemm_sm80",&grouped_gemm_sm80);
}
