// cutlass_grouped_gemm.cu - SM80 fp16 variable-M grouped GEMM (no pad)
// Optimized: persistent problem buffers, pinned host, single packed H2D for descriptors
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/gemm.h>
#include <vector>
#include <string>

#include "prefill_moe_cutlass_gemm.h"

using Element = cutlass::bfloat16_t;
using Acc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Default production tile (keep semantics identical)
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
  Element, LayoutA, cutlass::ComplexTransform::kNone, 8,
  Element, LayoutB, cutlass::ComplexTransform::kNone, 8,
  Element, LayoutC,
  Acc,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombination<Element, 8, Acc, Acc>,
  cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
  4
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// Per-call descriptor block: one pinned host tensor (torch caching host allocator,
// no cudaMallocHost per call) copied once, non-blocking, into one device tensor
// (torch caching allocator).  copy_ records the stream event, so both buffers
// stay alive until the H2D copy completes.  No process-global state.
// layout (each section 16B-aligned; GemmCoord is 12B so problems goes last):
//   A[na] | B[na] | C[na] | lda[na] | ldb[na] | ldc[na] | problems[na]
struct DescLayout {
  int64_t na, off_problems, off_A, off_B, off_C, off_lda, off_ldb, off_ldc, bytes;
  static int64_t al16(int64_t v) { return (v + 15) & ~(int64_t)15; }
  explicit DescLayout(int64_t n) : na(n) {
    int64_t o = 0;
    off_A = o; o = al16(o + n * (int64_t)sizeof(Element*));
    off_B = o; o = al16(o + n * (int64_t)sizeof(Element*));
    off_C = o; o = al16(o + n * (int64_t)sizeof(Element*));
    off_lda = o; o = al16(o + n * (int64_t)sizeof(int64_t));
    off_ldb = o; o = al16(o + n * (int64_t)sizeof(int64_t));
    off_ldc = o; o = al16(o + n * (int64_t)sizeof(int64_t));
    off_problems = o; o = al16(o + n * (int64_t)sizeof(cutlass::gemm::GemmCoord));
    bytes = o;
  }
};

torch::Tensor grouped_gemm_sm80(
    torch::Tensor x,          // [Ntot, K] fp16 contiguous row-major
    torch::Tensor w,          // [na, Nout, K] fp16  (column-major B storage)
    torch::Tensor sizes,      // [na] int64 M_i, sum=Ntot
    int64_t threadblock_count // 0 = auto sufficient()
){
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sizes.is_cuda(), "cuda tensors");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && w.scalar_type()==torch::kBFloat16, "bf16");
  TORCH_CHECK(x.dim()==2 && w.dim()==3 && sizes.dim()==1, "shapes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && sizes.is_contiguous(), "contiguous");

  const int device = x.get_device();
  c10::cuda::CUDAGuard guard(device);
  const int64_t na = sizes.size(0);
  const int64_t K = x.size(1);
  const int64_t Nout = w.size(1);
  TORCH_CHECK(w.size(2)==K, "K mismatch");
  TORCH_CHECK(w.size(0)==na, "na mismatch");
  const int64_t Ntot = x.size(0);

  auto y = torch::empty({Ntot, Nout}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(device).stream();

  // sizes on CPU for M_i (small na~50)
  auto sizes_cpu = sizes.to(torch::kCPU).contiguous();
  auto sp = sizes_cpu.data_ptr<int64_t>();

  DescLayout lay(na);
  auto h = torch::empty({lay.bytes},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
  char* hp = (char*)h.data_ptr();
  auto* h_problems = (cutlass::gemm::GemmCoord*)(hp + lay.off_problems);
  auto* h_A = (Element**)(hp + lay.off_A);
  auto* h_B = (Element**)(hp + lay.off_B);
  auto* h_C = (Element**)(hp + lay.off_C);
  auto* h_lda = (int64_t*)(hp + lay.off_lda);
  auto* h_ldb = (int64_t*)(hp + lay.off_ldb);
  auto* h_ldc = (int64_t*)(hp + lay.off_ldc);

  Element* x_base = reinterpret_cast<Element*>(x.data_ptr());
  Element* w_base = reinterpret_cast<Element*>(w.data_ptr());
  Element* y_base = reinterpret_cast<Element*>(y.data_ptr());

  int64_t off = 0;
  for (int64_t i = 0; i < na; ++i) {
    int64_t m = sp[i];
    TORCH_CHECK(m >= 0, "neg size");
    h_problems[i] = cutlass::gemm::GemmCoord((int)m, (int)Nout, (int)K);
    h_A[i] = x_base + off * K;
    // w is row-major [Nout, K] per expert == ColumnMajor (K, Nout) with ld = K
    h_B[i] = w_base + i * Nout * K;
    h_C[i] = y_base + off * Nout;
    h_lda[i] = K;
    h_ldb[i] = K;
    h_ldc[i] = Nout;
    off += m;
  }
  TORCH_CHECK(off == Ntot, "sizes sum != Ntot");

  // single non-blocking H2D on the current stream
  auto d = h.to(torch::TensorOptions().device(torch::kCUDA, device), /*non_blocking=*/true);
  char* dp = (char*)d.data_ptr();

  using EpilogueOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOp::Params epilogue_op(1.0f, 0.0f);

  int tb = (int)threadblock_count;
  if (tb <= 0) {
    tb = GemmGrouped::sufficient();
    if (tb <= 0) tb = 216;
  }

  typename GemmGrouped::Arguments args(
    reinterpret_cast<cutlass::gemm::GemmCoord*>(dp + lay.off_problems),
    (int)na,
    tb,
    epilogue_op,
    reinterpret_cast<Element**>(dp + lay.off_A),
    reinterpret_cast<Element**>(dp + lay.off_B),
    reinterpret_cast<Element**>(dp + lay.off_C),
    reinterpret_cast<Element**>(dp + lay.off_C),
    reinterpret_cast<int64_t*>(dp + lay.off_lda),
    reinterpret_cast<int64_t*>(dp + lay.off_ldb),
    reinterpret_cast<int64_t*>(dp + lay.off_ldc),
    reinterpret_cast<int64_t*>(dp + lay.off_ldc)
  );

  GemmGrouped gemm;
  size_t ws_bytes = GemmGrouped::get_workspace_size(args);
  void* ws_ptr = nullptr;
  torch::Tensor workspace;
  if (ws_bytes > 0) {
    workspace = torch::empty({(int64_t)ws_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
    ws_ptr = workspace.data_ptr();
  }

  auto status = gemm.initialize(args, ws_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
    std::string("cutlass initialize failed: ")+cutlassGetStatusString(status)+
    " tb="+std::to_string(tb));

  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
    std::string("cutlass run failed: ")+cutlassGetStatusString(status));

  return y;
}

int64_t grouped_gemm_sufficient(){
  return (int64_t)GemmGrouped::sufficient();
}

#ifdef CUTLASS_GROUPED_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("grouped_gemm_sm80", &grouped_gemm_sm80, "CUTLASS SM80 fp16 grouped GEMM variable M");
  m.def("grouped_gemm_sufficient", &grouped_gemm_sufficient, "recommended threadblock_count");
}
#endif
