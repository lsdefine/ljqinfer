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
#include <mutex>
#include <unordered_map>

#include "prefill_moe_cutlass_gemm.h"

using Element = cutlass::half_t;
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

struct PersistBuf {
  int64_t cap = 0;
  // pinned host
  cutlass::gemm::GemmCoord* h_problems = nullptr;
  Element** h_A = nullptr;
  Element** h_B = nullptr;
  Element** h_C = nullptr;
  int64_t* h_lda = nullptr;
  int64_t* h_ldb = nullptr;
  int64_t* h_ldc = nullptr;
  // device
  torch::Tensor d_problems;
  torch::Tensor d_ptr_A, d_ptr_B, d_ptr_C;
  torch::Tensor d_lda, d_ldb, d_ldc;
  torch::Tensor workspace;
  size_t ws_cap = 0;
};

static std::mutex g_mu;
static std::unordered_map<int, PersistBuf> g_bufs; // key = device index

static PersistBuf& get_buf(int device, int64_t na) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& b = g_bufs[device];
  if (b.cap >= na) return b;
  // free old pinned
  if (b.h_problems) { cudaFreeHost(b.h_problems); b.h_problems=nullptr; }
  // allocate one contiguous pinned block for all host arrays
  // layout: problems[na] | A[na] | B[na] | C[na] | lda[na] | ldb[na] | ldc[na]
  size_t bytes =
    (size_t)na * sizeof(cutlass::gemm::GemmCoord) +
    (size_t)na * sizeof(Element*) * 3 +
    (size_t)na * sizeof(int64_t) * 3;
  void* raw = nullptr;
  AT_CUDA_CHECK(cudaMallocHost(&raw, bytes));
  char* p = (char*)raw;
  b.h_problems = (cutlass::gemm::GemmCoord*)p; p += na * sizeof(cutlass::gemm::GemmCoord);
  b.h_A = (Element**)p; p += na * sizeof(Element*);
  b.h_B = (Element**)p; p += na * sizeof(Element*);
  b.h_C = (Element**)p; p += na * sizeof(Element*);
  b.h_lda = (int64_t*)p; p += na * sizeof(int64_t);
  b.h_ldb = (int64_t*)p; p += na * sizeof(int64_t);
  b.h_ldc = (int64_t*)p;
  auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device);
  // device buffers (byte views for problems/ptrs)
  b.d_problems = torch::empty({na * (int64_t)sizeof(cutlass::gemm::GemmCoord)}, opts_u8);
  b.d_ptr_A = torch::empty({na * (int64_t)sizeof(Element*)}, opts_u8);
  b.d_ptr_B = torch::empty({na * (int64_t)sizeof(Element*)}, opts_u8);
  b.d_ptr_C = torch::empty({na * (int64_t)sizeof(Element*)}, opts_u8);
  b.d_lda = torch::empty({na}, opts_i64);
  b.d_ldb = torch::empty({na}, opts_i64);
  b.d_ldc = torch::empty({na}, opts_i64);
  b.cap = na;
  b.ws_cap = 0;
  return b;
}

torch::Tensor grouped_gemm_sm80(
    torch::Tensor x,          // [Ntot, K] fp16 contiguous row-major
    torch::Tensor w,          // [na, Nout, K] fp16  (column-major B storage)
    torch::Tensor sizes,      // [na] int64 M_i, sum=Ntot
    int64_t threadblock_count // 0 = auto sufficient()
){
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sizes.is_cuda(), "cuda tensors");
  TORCH_CHECK(x.scalar_type()==torch::kFloat16 && w.scalar_type()==torch::kFloat16, "fp16");
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

  PersistBuf& buf = get_buf(device, na);

  Element* x_base = reinterpret_cast<Element*>(x.data_ptr<at::Half>());
  Element* w_base = reinterpret_cast<Element*>(w.data_ptr<at::Half>());
  Element* y_base = reinterpret_cast<Element*>(y.data_ptr<at::Half>());

  int64_t off = 0;
  for (int64_t i = 0; i < na; ++i) {
    int64_t m = sp[i];
    TORCH_CHECK(m >= 0, "neg size");
    buf.h_problems[i] = cutlass::gemm::GemmCoord((int)m, (int)Nout, (int)K);
    buf.h_A[i] = x_base + off * K;
    // B is ColumnMajor [K, Nout] stored as contiguous [Nout, K] per expert
    buf.h_B[i] = w_base + i * Nout * K;
    buf.h_C[i] = y_base + off * Nout;
    buf.h_lda[i] = K;
    buf.h_ldb[i] = Nout; // ColumnMajor ld = rows = Nout when viewing as K x Nout? 
    // Original used: h_ldb[i] = K; wait check original...
    // Original: h_ldb[i] = K; for ColumnMajor B with shape (K, Nout) ld is K.
    // But w is [Nout, K] contiguous. ColumnMajor (K,Nout) means element (k,n) at k + n*ld with ld>=K.
    // If stored as row-major [Nout,K], element (n,k) at n*K+k, which equals ColumnMajor (k,n) at k + n*K, so ld=K.
    buf.h_ldb[i] = K;
    buf.h_ldc[i] = Nout;
    off += m;
  }
  TORCH_CHECK(off == Ntot, "sizes sum != Ntot");

  // H2D descriptors
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_problems.data_ptr(), buf.h_problems, na*sizeof(cutlass::gemm::GemmCoord), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_ptr_A.data_ptr(), buf.h_A, na*sizeof(Element*), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_ptr_B.data_ptr(), buf.h_B, na*sizeof(Element*), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_ptr_C.data_ptr(), buf.h_C, na*sizeof(Element*), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_lda.data_ptr(), buf.h_lda, na*sizeof(int64_t), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_ldb.data_ptr(), buf.h_ldb, na*sizeof(int64_t), cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(buf.d_ldc.data_ptr(), buf.h_ldc, na*sizeof(int64_t), cudaMemcpyHostToDevice, stream));

  using EpilogueOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOp::Params epilogue_op(1.0f, 0.0f);

  int tb = (int)threadblock_count;
  if (tb <= 0) {
    tb = GemmGrouped::sufficient();
    if (tb <= 0) tb = 216;
  }

  typename GemmGrouped::Arguments args(
    reinterpret_cast<cutlass::gemm::GemmCoord*>(buf.d_problems.data_ptr()),
    (int)na,
    tb,
    epilogue_op,
    reinterpret_cast<Element**>(buf.d_ptr_A.data_ptr()),
    reinterpret_cast<Element**>(buf.d_ptr_B.data_ptr()),
    reinterpret_cast<Element**>(buf.d_ptr_C.data_ptr()),
    reinterpret_cast<Element**>(buf.d_ptr_C.data_ptr()),
    reinterpret_cast<int64_t*>(buf.d_lda.data_ptr()),
    reinterpret_cast<int64_t*>(buf.d_ldb.data_ptr()),
    reinterpret_cast<int64_t*>(buf.d_ldc.data_ptr()),
    reinterpret_cast<int64_t*>(buf.d_ldc.data_ptr())
  );

  GemmGrouped gemm;
  size_t ws_bytes = GemmGrouped::get_workspace_size(args);
  void* ws_ptr = nullptr;
  if (ws_bytes > 0) {
    if (buf.ws_cap < ws_bytes) {
      auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
      buf.workspace = torch::empty({(int64_t)ws_bytes}, opts_u8);
      buf.ws_cap = ws_bytes;
    }
    ws_ptr = buf.workspace.data_ptr();
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
