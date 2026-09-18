#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cstdlib>
#include <algorithm>

torch::Tensor dequant_k_cuda(torch::Tensor p, int64_t K, int64_t qt);
torch::Tensor mmq_iq3_tile_cuda(torch::Tensor packed, torch::Tensor x);
torch::Tensor mmq_iq4_tile_cuda(torch::Tensor packed, torch::Tensor x);
torch::Tensor dequant_iq3_xxs_cuda(torch::Tensor p, int64_t K);
torch::Tensor dequant_iq4_xs_cuda(torch::Tensor p, int64_t K);

static inline int64_t row_bytes_qt(int64_t qt){
  // bytes per 256-wide block for Qk_K family used by special layers
  if(qt==3) return 110;
  if(qt==4) return 144;
  if(qt==5) return 176;
  if(qt==6) return 210;
  return 0;
}

static torch::Tensor linear_q(torch::Tensor p, torch::Tensor x, int64_t K, int64_t qt){
  // IQ and Qk_K weights are dequantized before batched matmul.
  if(qt==1) return torch::matmul(x, dequant_iq3_xxs_cuda(p, K).to(torch::kFloat32).t());
  if(qt==2) return torch::matmul(x, dequant_iq4_xs_cuda(p, K).to(torch::kFloat32).t());
  return torch::matmul(x, dequant_k_cuda(p, K, qt).t());
}

static torch::Tensor ww_as_col(torch::Tensor ww){
  // Accept [N] or [N,1] -> [N,1]
  if(ww.dim()==1) return ww.unsqueeze(1);
  TORCH_CHECK(ww.dim()==2 && ww.size(1)==1, "ww must be [N] or [N,1]");
  return ww;
}

static void check_shapes(const torch::Tensor& gp, const torch::Tensor& up,
                         const torch::Tensor& dp, int64_t local, int64_t dqt){
  int64_t rb=row_bytes_qt(dqt);
  TORCH_CHECK(rb>0, "unsupported dqt");
  TORCH_CHECK(gp.size(1)==local, "gp local mismatch");
  TORCH_CHECK(up.size(1)==local, "up local mismatch");
  // dp: [E, D, row_bytes_for_local] layout used historically: dp.size(2)*256/rb == local
  TORCH_CHECK(dp.size(2)*256/rb==local, "dp local/bytes mismatch");
}

// v1 legacy (GPU offsets)

// v2: CPU offsets, expert for-loop (baseline production)

// v3: multi-stream, private [T,D] per stream then sum (known slower at e2e)

// v4: single gather -> sbuf multi-stream expert GEMMs -> one scatter
// Avoids per-expert index_select and private full [T,D] outs.


// v5: gather once + batch-dequant ALL active DOWN + multi-stream
// gate/up stay fused MMQ (gqt=1/2); down uses dequant_k + matmul (dqt=3..6)
// Goal: remove ~na serial dequant launches on down path.
torch::Tensor special_moe_rank_forward_v5(torch::Tensor x, torch::Tensor tok,
 torch::Tensor ww, torch::Tensor off_cpu, torch::Tensor gp, torch::Tensor up,
 torch::Tensor dp, int64_t gqt, int64_t dqt) {
  TORCH_CHECK(x.is_cuda() && tok.is_cuda() && ww.is_cuda());
  TORCH_CHECK(off_cpu.is_cpu() && off_cpu.scalar_type()==torch::kLong && off_cpu.is_contiguous());
  TORCH_CHECK(dqt>=3 && dqt<=6, "v5 requires Qk_K down (dqt 3..6) for batch dequant");
  int64_t E=gp.size(0), local=up.size(1), D=x.size(1);
  check_shapes(gp,up,dp,local,dqt);
  auto wwc=ww_as_col(ww);
  const int64_t* op = off_cpu.data_ptr<int64_t>();
  int64_t N = op[E];
  TORCH_CHECK(tok.numel()==N, "tok length must match offsets endpoint");
  TORCH_CHECK(wwc.size(0)==N, "ww length must match offsets endpoint");

  std::vector<int64_t> active; active.reserve(E);
  for (int64_t e=0;e<E;e++) if (op[e]!=op[e+1]) active.push_back(e);
  if (active.empty()) return torch::zeros_like(x);
  const int64_t na = (int64_t)active.size();

  // Bound both fixed expert-weight storage and query-dependent routed storage.
  // A global gather/sbuf has shape [TOPK*T,D] and is 6 GiB at T=32768.  Process
  // experts in deterministic CSR order instead: each expert gathers at most T
  // rows, computes its contribution, and immediately accumulates it into out.
  // Source-controlled down-projection batch; no environment tuning.
  int64_t down_batch = 4;
  if (down_batch > na) down_batch = na;

  auto out=torch::zeros_like(x);
  for(int64_t base=0;base<na;base+=down_batch){
    int64_t nb=std::min<int64_t>(down_batch,na-base);
    auto act = torch::from_blob(active.data()+base, {nb},
                                torch::TensorOptions().dtype(torch::kLong)).clone();
    auto act_dev = act.to(dp.device());
    auto d_act = dp.index_select(0, act_dev); // [nb, D, rb]
    int64_t rb = d_act.size(2);
    auto d_flat = d_act.reshape({nb * D, rb}).contiguous();
    auto d_fp = dequant_k_cuda(d_flat, local, dqt); // [nb*D, local]
    auto d_mat = d_fp.view({nb, D, local}); // [nb, D, local] fp32

    for(int64_t j=0;j<nb;j++){
      int64_t i=base+j, e=active[i];
      int64_t a=op[e], b=op[e+1], n=b-a;
      auto ti=tok.narrow(0,a,n);
      auto xe=x.index_select(0,ti).contiguous(); // at most [T,D]
      auto w=wwc.slice(0,a,b);
      auto g=linear_q(gp[e],xe,D,gqt);
      auto u=linear_q(up[e],xe,D,gqt);
      auto h=at::silu(g).mul_(u); g.reset(); u.reset();
      auto y=torch::matmul(h,d_mat[j].t()).mul_(w); h.reset();
      out.index_add_(0,ti,y);
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_k", &dequant_k_cuda);
  m.def("special_moe_rank_forward_v5", &special_moe_rank_forward_v5, "special batch-dequant down + multi-stream", pybind11::call_guard<pybind11::gil_scoped_release>());
}
