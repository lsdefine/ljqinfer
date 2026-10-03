
// Fused host-side Engram row collector: gather + FP8 dequant + bf16 store in
// one pass. Replaces numpy advanced indexing plus six torch CPU ops (0.82ms
// per decode round for 21 rows) that dominated the exposed host tail.
#include <torch/extension.h>
#include <cmath>

void engram_rows(const at::Tensor& wt, const at::Tensor& sc,
                 const at::Tensor& idx, const at::Tensor& lut, at::Tensor out) {
  TORCH_CHECK(wt.scalar_type() == at::kByte && sc.scalar_type() == at::kByte,
              "table bytes must be uint8 views");
  TORCH_CHECK(idx.scalar_type() == at::kLong && idx.is_contiguous(), "idx int64 contiguous");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(), "out bf16 contiguous");
  TORCH_CHECK(wt.size(1) == 256 && sc.size(1) == 8, "expected [rows,256] and [rows,8]");
  const int64_t n = idx.numel(), rows = wt.size(0);
  TORCH_CHECK(out.numel() >= n * 256, "out too small");
  const uint8_t* W = wt.data_ptr<uint8_t>();
  const uint8_t* S = sc.data_ptr<uint8_t>();
  const int64_t* I = idx.data_ptr<int64_t>();
  const float* L = lut.data_ptr<float>();
  at::BFloat16* O = out.data_ptr<at::BFloat16>();
  for (int64_t r = 0; r < n; ++r)
    TORCH_CHECK(I[r] >= 0 && I[r] < rows, "hash row out of range");
  pybind11::gil_scoped_release no_gil;
  // Each row sits at a random offset in a ~100GB mmap'd table, so the core
  // mostly waits on a TLB/page miss (measured ~48us per row, far above any
  // arithmetic cost). Threads exist to overlap those stalls, not to add FLOPs.
#pragma omp parallel for schedule(static) num_threads(4) if (n > 1)
  for (int64_t r = 0; r < n; ++r) {
    const int64_t id = I[r];
    const uint8_t* wr = W + id * 256;
    const uint8_t* sr = S + id * 8;
    at::BFloat16* o = O + r * 256;
    for (int g = 0; g < 8; ++g) {
      const float m = std::ldexp(1.0f, (int)sr[g] - 127);
      const uint8_t* wg = wr + g * 32;
      at::BFloat16* og = o + g * 32;
      for (int j = 0; j < 32; ++j) og[j] = at::BFloat16(L[wg[j]] * m);
    }
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("engram_rows", &engram_rows); }
