// cutlass_grouped_gemm.cu - SM80 bf16 variable-M grouped GEMM (no pad)
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

#include <c10/cuda/CUDAException.h>

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
    torch::Tensor x,          // [Ntot, K] bf16 contiguous row-major
    torch::Tensor w,          // [na, Nout, K] bf16  (column-major B storage)
    torch::Tensor sizes,      // [na] int64 M_i, sum=Ntot
    int64_t threadblock_count,
    torch::Tensor y, torch::Tensor h, torch::Tensor d, torch::Tensor workspace
    // Caller owns buffers. Before host descriptor overwrite, wait for its
    // preceding copy-completion event. Device buffers require stream ordering.
){
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && sizes.device().is_cpu(), "cuda tensors");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && w.scalar_type()==torch::kBFloat16, "bf16");
  TORCH_CHECK(x.dim()==2 && w.dim()==3 && sizes.dim()==1, "shapes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && sizes.is_contiguous(), "contiguous");

  TORCH_CHECK(x.device()==w.device(), "same CUDA device required");
  TORCH_CHECK(sizes.scalar_type()==torch::kInt64, "sizes must be int64");
  TORCH_CHECK(x.size(1)>0 && x.size(1)%8==0 && w.size(1)>0 && w.size(1)%8==0, "positive aligned K/N required");
  TORCH_CHECK(threadblock_count>=0 && threadblock_count<=2147483647LL, "invalid threadblock count");
  const int device = x.get_device();
  c10::cuda::CUDAGuard guard(device);
  const int64_t na = sizes.size(0);
  const int64_t K = x.size(1);
  const int64_t Nout = w.size(1);
  TORCH_CHECK(w.size(2)==K, "K mismatch");
  TORCH_CHECK(w.size(0)==na, "na mismatch");
  const int64_t Ntot = x.size(0);
  TORCH_CHECK(K<=2147483647LL && Nout<=2147483647LL && na<=2147483647LL,
      "CUTLASS coordinate overflow");

  TORCH_CHECK(y.device()==x.device() && y.scalar_type()==x.scalar_type() &&
      y.is_contiguous() && y.dim()==2 && y.size(0)==Ntot && y.size(1)==Nout,
      "output geometry/device/dtype");
  TORCH_CHECK(!y.is_alias_of(x) && !y.is_alias_of(w), "output alias");
  auto stream = at::cuda::getCurrentCUDAStream(device).stream();

  // sizes on CPU for M_i (small na~50)
  auto sizes_cpu = sizes;
  auto sp = sizes_cpu.data_ptr<int64_t>();

  DescLayout lay(na);
  TORCH_CHECK(h.device().is_cpu() && h.is_pinned() && h.is_contiguous() &&
      h.scalar_type()==torch::kUInt8 && h.numel()>=lay.bytes,
      "pinned host descriptor capacity");
  TORCH_CHECK(d.device()==x.device() && d.is_contiguous() &&
      d.scalar_type()==torch::kUInt8 && d.numel()>=lay.bytes,
      "device descriptor capacity");
  TORCH_CHECK(workspace.device()==x.device() && workspace.is_contiguous() &&
      workspace.scalar_type()==torch::kUInt8, "workspace device/dtype");
  TORCH_CHECK(!d.is_alias_of(workspace) && !d.is_alias_of(x) &&
      !d.is_alias_of(w) && !d.is_alias_of(y) && !workspace.is_alias_of(x) &&
      !workspace.is_alias_of(w) && !workspace.is_alias_of(y), "scratch alias");
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
    TORCH_CHECK(m >= 0 && m<=2147483647LL, "invalid expert row count");
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

  if (Ntot == 0) return y;

  // single non-blocking H2D on the current stream
  C10_CUDA_CHECK(cudaMemcpyAsync(d.data_ptr(), h.data_ptr(), lay.bytes,
      cudaMemcpyHostToDevice, stream));
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
  TORCH_CHECK(workspace.numel() >= (int64_t)ws_bytes, "CUTLASS workspace capacity");
  if (ws_bytes > 0) ws_ptr = workspace.data_ptr();

  auto status = gemm.initialize(args, ws_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
    std::string("cutlass initialize failed: ")+cutlassGetStatusString(status)+
    " tb="+std::to_string(tb));

  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
    std::string("cutlass run failed: ")+cutlassGetStatusString(status));

  return y;
}


// ---- device-side descriptor path (graph-capturable; no host readback) ----
__global__ void fill_desc_kernel(const int64_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    cutlass::gemm::GemmCoord* problems, Element** A, Element** B, Element** C,
    int64_t* lda, int64_t* ldb, int64_t* ldc,
    Element* x, Element* w, Element* y, int na, int K, int Nout){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= na) return;
  int64_t m = counts[i], off = offsets[i];
  problems[i] = cutlass::gemm::GemmCoord((int)m, Nout, K);
  A[i] = x + off*(int64_t)K;
  B[i] = w + (int64_t)i*(int64_t)Nout*(int64_t)K;
  C[i] = y + off*(int64_t)Nout;
  lda[i] = K; ldb[i] = K; ldc[i] = Nout;
}

torch::Tensor grouped_gemm_sm80_device(
    torch::Tensor x, torch::Tensor w,
    torch::Tensor counts, torch::Tensor offsets,
    int64_t threadblock_count,
    torch::Tensor y, torch::Tensor d, torch::Tensor workspace){
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && counts.is_cuda() && offsets.is_cuda(), "cuda tensors");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && w.scalar_type()==torch::kBFloat16, "bf16");
  TORCH_CHECK(counts.scalar_type()==torch::kInt64 && offsets.scalar_type()==torch::kInt64, "int64 counts/offsets");
  TORCH_CHECK(x.dim()==2 && w.dim()==3 && counts.dim()==1 && offsets.dim()==1, "shapes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && counts.is_contiguous() && offsets.is_contiguous(), "contiguous");
  const int device = x.get_device();
  c10::cuda::CUDAGuard guard(device);
  const int64_t na = counts.size(0);
  TORCH_CHECK(offsets.size(0)==na && w.size(0)==na, "na mismatch");
  const int64_t K = x.size(1), Nout = w.size(1);
  TORCH_CHECK(w.size(2)==K, "K mismatch");
  TORCH_CHECK(K>0 && K%8==0 && Nout>0 && Nout%8==0, "positive aligned K/N required");
  TORCH_CHECK(y.device()==x.device() && y.scalar_type()==x.scalar_type() &&
      y.is_contiguous() && y.dim()==2 && y.size(1)==Nout && y.size(0)==x.size(0),
      "output geometry/device/dtype");
  TORCH_CHECK(!y.is_alias_of(x) && !y.is_alias_of(w), "output alias");
  DescLayout lay(na);
  TORCH_CHECK(d.device()==x.device() && d.is_contiguous() &&
      d.scalar_type()==torch::kUInt8 && d.numel()>=lay.bytes, "device descriptor capacity");
  TORCH_CHECK(workspace.device()==x.device() && workspace.is_contiguous() &&
      workspace.scalar_type()==torch::kUInt8, "workspace device/dtype");
  auto stream = at::cuda::getCurrentCUDAStream(device).stream();
  char* dp = (char*)d.data_ptr();
  fill_desc_kernel<<<(unsigned)((na+63)/64), 64, 0, stream>>>(
      counts.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      (cutlass::gemm::GemmCoord*)(dp + lay.off_problems),
      (Element**)(dp + lay.off_A), (Element**)(dp + lay.off_B), (Element**)(dp + lay.off_C),
      (int64_t*)(dp + lay.off_lda), (int64_t*)(dp + lay.off_ldb), (int64_t*)(dp + lay.off_ldc),
      reinterpret_cast<Element*>(x.data_ptr()), reinterpret_cast<Element*>(w.data_ptr()),
      reinterpret_cast<Element*>(y.data_ptr()), (int)na, (int)K, (int)Nout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  using EpilogueOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOp::Params epilogue_op(1.0f, 0.0f);
  int tb = (int)threadblock_count;
  if (tb <= 0) { tb = GemmGrouped::sufficient(); if (tb <= 0) tb = 216; }
  typename GemmGrouped::Arguments args(
    reinterpret_cast<cutlass::gemm::GemmCoord*>(dp + lay.off_problems), (int)na, tb, epilogue_op,
    reinterpret_cast<Element**>(dp + lay.off_A),
    reinterpret_cast<Element**>(dp + lay.off_B),
    reinterpret_cast<Element**>(dp + lay.off_C),
    reinterpret_cast<Element**>(dp + lay.off_C),
    reinterpret_cast<int64_t*>(dp + lay.off_lda),
    reinterpret_cast<int64_t*>(dp + lay.off_ldb),
    reinterpret_cast<int64_t*>(dp + lay.off_ldc),
    reinterpret_cast<int64_t*>(dp + lay.off_ldc));
  GemmGrouped gemm;
  size_t ws_bytes = GemmGrouped::get_workspace_size(args);
  void* ws_ptr = nullptr;
  TORCH_CHECK(workspace.numel() >= (int64_t)ws_bytes, "CUTLASS workspace capacity");
  if (ws_bytes > 0) ws_ptr = workspace.data_ptr();
  auto status = gemm.initialize(args, ws_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
    std::string("cutlass initialize failed: ")+cutlassGetStatusString(status));
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
  m.def("grouped_gemm_sm80", &grouped_gemm_sm80, "CUTLASS SM80 bf16 grouped GEMM variable M");
  m.def("grouped_gemm_sm80_device", &grouped_gemm_sm80_device, "device-side descriptors, graph-capturable");
  m.def("grouped_gemm_sufficient", &grouped_gemm_sufficient, "recommended threadblock_count");
}
#endif
