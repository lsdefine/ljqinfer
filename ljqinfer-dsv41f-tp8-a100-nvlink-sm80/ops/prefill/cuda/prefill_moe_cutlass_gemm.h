
#pragma once
#include <torch/extension.h>
// x: [Ntot,K] bf16 row-major
// w: [na,Nout,K] bf16 (column-major B)
// sizes: [na] int64, sum==Ntot
// threadblock_count: 0 = auto
torch::Tensor grouped_gemm_sm80(torch::Tensor x, torch::Tensor w, torch::Tensor sizes, int64_t threadblock_count=0);
int64_t grouped_gemm_sufficient();
