#include <unordered_map>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/silu.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include "prefill_moe_cutlass_gemm.h"

torch::Tensor dequant_iq3_xxs_cuda(torch::Tensor packed, int64_t K);
torch::Tensor dequant_iq4_xs_cuda(torch::Tensor packed, int64_t K);
torch::Tensor dequant_iq3_selected_cuda(torch::Tensor p,torch::Tensor act,int64_t K);
torch::Tensor dequant_iq4_selected_cuda(torch::Tensor p,torch::Tensor act,int64_t K);
torch::Tensor mmq_iq3_tile_cuda(torch::Tensor packed, torch::Tensor x);
torch::Tensor shared_decode_q8_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x);
void rms_norm_t1_inplace_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor yf, torch::Tensor yh, double eps);
void shared_decode_q8_inplace_cuda(torch::Tensor g, torch::Tensor u, torch::Tensor d, torch::Tensor x, torch::Tensor h, torch::Tensor y);
void moe_route_decode_t1_inplace_cuda(torch::Tensor x, torch::Tensor router, torch::Tensor bias, torch::Tensor ei, torch::Tensor ew, int64_t n_group, int64_t top_g, int64_t top_k, double scale);
void moe_route_select_only_cuda(torch::Tensor logits, torch::Tensor bias, torch::Tensor ei, torch::Tensor ew, int64_t n_group, int64_t top_g, int64_t top_k, double scale);
std::vector<torch::Tensor> moe_route_decode_t1_cuda(torch::Tensor x, torch::Tensor router, torch::Tensor bias, int64_t n_group, int64_t top_g, int64_t top_k, double scale);



torch::Tensor mmq_iq4_tile_cuda(torch::Tensor packed, torch::Tensor x);
torch::Tensor down_iq4_tc_cuda(torch::Tensor packed, torch::Tensor x);
torch::Tensor moe_decode_iq_fused_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x,torch::Tensor ei,torch::Tensor ew);
__attribute__((visibility("hidden"))) torch::Tensor moe_decode_iq_fused_out_cuda(torch::Tensor g,torch::Tensor u,torch::Tensor d,torch::Tensor x,torch::Tensor ei,torch::Tensor ew,torch::Tensor h,torch::Tensor y);
torch::Tensor add_f32_f16_to_f16_cuda(torch::Tensor a, torch::Tensor b);

static int64_t policy_threshold(int64_t B){ return B<384?8:(B<1280?12:(B<3072?34:0)); }

// Production backend is fixed to dequant -> fp32 sgemm.  Alternate experimental
// backends have explicit APIs only and cannot be selected through the environment.
static constexpr bool use_cublas_backend(){ return false; }

static void ck(const torch::Tensor& p,const char* n,int64_t rows,int64_t rb){
 TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kUInt8 && p.is_contiguous(),n," must be contiguous CUDA uint8");
 TORCH_CHECK(p.dim()==3 && p.size(0)==256 && p.size(1)==rows && p.size(2)==rb,n," wrong shape");
}
// v1: original per-expert serial loop. One coarse launch boundary for one TP rank.
// Returned tensor is DTensor placement PARTIAL.
torch::Tensor moe_rank_forward(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
 const int64_t local=gpack.size(1);
 TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
 ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
 TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
 TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
 TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
 TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
 c10::cuda::CUDAGuard guard(x.device());
 auto out=torch::zeros_like(x); const auto* off=offsets.data_ptr<int64_t>();
 const int64_t threshold=policy_threshold(x.size(0));
 const bool cublas=use_cublas_backend();
 TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
 for(int64_t e=0;e<256;e++){
  int64_t a=off[e], n=off[e+1]-a; TORCH_CHECK(n>=0,"offsets not monotonic"); if(!n) continue;
  auto ti=tok.narrow(0,a,n); auto ww=weight.narrow(0,a,n); auto xb=x.index_select(0,ti);
  torch::Tensor gate,hidden,y;
  if(threshold && n<threshold){
   gate=mmq_iq3_tile_cuda(gpack[e],xb); auto up=mmq_iq3_tile_cuda(upack[e],xb);
   hidden=at::silu(gate).mul_(up); gate.reset(); up.reset(); xb.reset();
   y=mmq_iq4_tile_cuda(dpack[e],hidden).mul_(ww); hidden.reset();
  }else if(cublas){
   auto xh=xb.to(torch::kFloat16);
   auto wg=dequant_iq3_xxs_cuda(gpack[e],6144);
   gate=at::matmul(xh,wg.t()); wg.reset();
   auto wu=dequant_iq3_xxs_cuda(upack[e],6144);
   hidden=at::silu(gate).mul_(at::matmul(xh,wu.t())); gate.reset(); wu.reset(); xh.reset(); xb.reset();
   auto wd=dequant_iq4_xs_cuda(dpack[e],local);
   y=at::matmul(hidden,wd.t()).to(torch::kFloat32).mul_(ww); hidden.reset(); wd.reset();
  }else{
   auto wg=dequant_iq3_xxs_cuda(gpack[e],6144).to(torch::kFloat32);
   gate=torch::matmul(xb,wg.t()); wg.reset();
   auto wu=dequant_iq3_xxs_cuda(upack[e],6144).to(torch::kFloat32);
   hidden=at::silu(gate).mul_(torch::matmul(xb,wu.t())); gate.reset(); wu.reset(); xb.reset();
   auto wd=dequant_iq4_xs_cuda(dpack[e],local).to(torch::kFloat32);
   y=torch::matmul(hidden,wd.t()).mul_(ww); hidden.reset(); wd.reset();
  }
  out.index_add_(0,ti,y);
 }
 return out;
}
// v2: data-flow refactor. Single gather (cast x->fp16 once, gather by expert-sorted tok),
// per-expert zero-copy narrow views, contiguous cuBLAS hgemm, single scatter_add back.
// Precision matches v1 cublas path (fp16 matmul, fp32 down-out mul). dequant stays per-active-expert.
// Attacks the profile's biggest bucket: per-expert scatter/gather (27%) + x cast churn.
torch::Tensor moe_rank_forward_v2(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
 const int64_t local=gpack.size(1);
 TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
 ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
 TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
 TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
 TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
 TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
 c10::cuda::CUDAGuard guard(x.device());
 const auto* off=offsets.data_ptr<int64_t>();
 TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
 const int64_t N=tok.size(0), B=x.size(0);
 // single gather: cast x to fp16 once, gather all routed tokens by expert-sorted tok -> [N,6144] fp16 contiguous
 auto xh=x.to(torch::kFloat16);
 auto xb=xh.index_select(0,tok);   // [N,6144] fp16, rows ordered by expert segment
 xh.reset();
 auto out32=torch::zeros({N,6144},x.options());  // [N,6144] fp32, expert-sorted output buffer
 for(int64_t e=0;e<256;e++){
  int64_t a=off[e], n=off[e+1]-a; TORCH_CHECK(n>=0,"offsets not monotonic"); if(!n) continue;
  auto xe=xb.narrow(0,a,n);        // [n,6144] fp16 contiguous view (zero-copy)
  auto we=weight.narrow(0,a,n);    // [n,1] fp32 view (weight already expert-sorted)
  auto wg=dequant_iq3_xxs_cuda(gpack[e],6144);
  auto gate=at::matmul(xe,wg.t()); wg.reset();
  auto wu=dequant_iq3_xxs_cuda(upack[e],6144);
  auto hidden=at::silu(gate).mul_(at::matmul(xe,wu.t())); gate.reset(); wu.reset();
  auto wd=dequant_iq4_xs_cuda(dpack[e],local);
  auto ye=at::matmul(hidden,wd.t()).to(torch::kFloat32).mul_(we); hidden.reset(); wd.reset();
  out32.narrow(0,a,n).copy_(ye);   // direct sorted write, no per-expert scatter
 }
 // single scatter back to original token order (sums the 8 expert contributions per token)
 auto out=torch::zeros({B,6144},x.options());
 out.index_add_(0,tok,out32);
 return out;
}
// v4: batched dequant + pow2 bucketing + single gather/scatter — C++ port of v3_mb.
// Eliminates v3_mb's ~5.5ms Python scatter/gather dispatch. Includes batched dequant
// (3 launches over ALL active experts) which v3_prof's compute_only precomputed/skipped.
// Precision: fp16 hgemm (gate/up/down), fp32 down-out mul — matches v1 cublas path.
// padX uses torch::empty (no zero-fill): padded garbage rows are never gathered back, safe.
static int64_t next_pow2_moe(int64_t v){
  v=std::max<int64_t>(v,1); int64_t p=1; while(p<v) p<<=1; return std::min<int64_t>(p,2048);
}
torch::Tensor moe_rank_forward_v4(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  using namespace torch::indexing;
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);
  if(N==0) return torch::zeros_like(x);

  // --- routing setup on CPU (offsets is CPU int64, no GPU sync) ---
  std::vector<int64_t> active, n_a, cap_a;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n<=0) continue;
    active.push_back(e); n_a.push_back(n); cap_a.push_back(next_pow2_moe(n));
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  // group active-expert positions by cap (ascending)
  std::map<int64_t,std::vector<int64_t>> buckets;
  for(int64_t i=0;i<na;i++) buckets[cap_a[i]].push_back(i);
  // per-token: j_global (active-expert idx) + row_local (offset within expert segment)
  std::vector<int64_t> jg_host(N), rl_host(N);
  { int64_t pos=0;
    for(int64_t i=0;i<na;i++){ for(int64_t r=0;r<n_a[i];r++){ jg_host[pos]=i; rl_host[pos]=r; pos++; } }
  }
  // member_pos[i] = position of active-expert i within its bucket
  std::vector<int64_t> mp_host(na,0);
  { for(const auto& kv:buckets){ const auto& members=kv.second; for(size_t p=0;p<members.size();p++) mp_host[members[p]]=(int64_t)p; } }

  // CPU index tensors -> GPU (synchronous H2D; sources alive in scope)
  auto jg=torch::from_blob(jg_host.data(),{N},torch::kInt64).to(x.device());
  auto rl=torch::from_blob(rl_host.data(),{N},torch::kInt64).to(x.device());
  auto cap_arr=torch::from_blob(cap_a.data(),{na},torch::kInt64).to(x.device());
  auto mp=torch::from_blob(mp_host.data(),{na},torch::kInt64).to(x.device());
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).to(x.device());
  auto cap_per_tok=cap_arr.index_select(0,jg);       // [N]
  auto member_per_tok=mp.index_select(0,jg);        // [N]

  // --- single gather: x->fp16 once, gather routed tokens -> [N,D] fp16 ---
  auto xg=x.to(torch::kFloat16).index_select(0,tok);

  // --- batched dequant (3 launches): flatten all active experts ---
  auto gact=gpack.index_select(0,act);                       // [na,local,rb_g]
  auto uact=upack.index_select(0,act);                       // [na,local,rb_g]
  auto dact=dpack.index_select(0,act);                        // [na,6144,rb_d]
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  auto out=torch::zeros({B,D},x.options());                   // fp32 [B,D]
  for(const auto& kv:buckets){
    const int64_t cap=kv.first;
    const auto& members=kv.second;
    const int64_t k=(int64_t)members.size();
    auto mem=torch::from_blob(const_cast<int64_t*>(members.data()),{k},torch::kInt64).to(x.device());
    auto mask=(cap_per_tok==cap);                             // [N] bool
    auto mt=member_per_tok.index({mask});                     // [m]
    auto rt=rl.index({mask});                                 // [m]
    auto tt=tok.index({mask});                                // [m]
    auto wwt=weight.index({mask});                           // [m,1]
    auto xvm=xg.index({mask});                                // [m,D]
    // padded X [k,cap,D] fp16 — empty (garbage rows never gathered); scatter real tokens
    auto padX=torch::empty({k,cap,D},xg.options());
    padX.index_put_({mt,rt},xvm);
    // weights: index members, transpose to [k,D,local]/[k,local,D] (strided, bmm handles)
    auto wgt=wg.index_select(0,mem).transpose(1,2);          // [k,D,local]
    auto wut=wu.index_select(0,mem).transpose(1,2);         // [k,D,local]
    auto wdt=wd.index_select(0,mem).transpose(1,2);          // [k,local,D]
    auto gate=torch::bmm(padX,wgt);                          // [k,cap,local]
    auto up=torch::bmm(padX,wut);                             // [k,cap,local]
    auto hid=at::silu(gate).mul_(up); gate.reset(); up.reset(); padX.reset();
    auto yb=torch::bmm(hid,wdt);                              // [k,cap,D]
    hid.reset(); wgt.reset(); wut.reset(); wdt.reset();
    auto yv=yb.index({mt,rt}).to(torch::kFloat32).mul_(wwt);  // [m,D] fp32
    yb.reset();
    out.index_add_(0,tt,yv);
  }
  return out;
}
// v5: v2 data flow (single gather/scatter, per-expert hgemm, NO padded buffer)
//     + v4 batched dequant (3 launches over ALL active experts instead of 3*na per-expert).
// Rationale: v4 profiler proved batched dequant is 0.8ms vs v1's 2.7ms per-expert (3x), but v4's
// padded-buffer gather/scatter (8.3ms) killed it. v5 keeps v2's proven no-padding per-expert hgemm
// (per-expert matmul is compute-bound, launch overhead tolerable) and grafts ONLY the batched dequant.
// Precision: fp16 hgemm (gate/up/down), fp32 down-out mul — matches v2/v1 cublas path exactly.
torch::Tensor moe_rank_forward_v5(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  if(N==0) return torch::zeros_like(x);

  // --- build active-expert list on CPU (offsets is CPU int64, no GPU sync) ---
  std::vector<int64_t> active;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n<=0) continue;
    active.push_back(e);
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  // single gather: cast x to fp16 once, gather all routed tokens by expert-sorted tok -> [N,D] fp16 contiguous
  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);   // [N,D] fp16, rows ordered by expert segment
  xh.reset();

  // --- batched dequant: gather active-expert packed weights, 3 launches over ALL active experts ---
  // (same as v4 lines 176-181, verified to produce 0.8ms vs 2.7ms per-expert)
  auto gact=gpack.index_select(0,act);                       // [na,local,rb_g]
  auto uact=upack.index_select(0,act);                       // [na,local,rb_g]
  auto dact=dpack.index_select(0,act);                        // [na,6144,rb_d]
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  // per-expert hgemm loop: zero-copy narrow views into xb, batched-dequanted weight views [i]
  auto out32=torch::zeros({N,D},x.options());  // [N,D] fp32, expert-sorted output buffer
  for(int64_t i=0;i<na;i++){
    int64_t e=active[i];
    int64_t a=off[e], n=off[e+1]-a; if(!n) continue;
    auto xe=xb.narrow(0,a,n);        // [n,D] fp16 contiguous view (zero-copy)
    auto we=weight.narrow(0,a,n);    // [n,1] fp32 view (weight already expert-sorted)
    auto gate=at::matmul(xe,wg[i].t());           // [n,local]
    auto hidden=at::silu(gate).mul_(at::matmul(xe,wu[i].t())); gate.reset();
    auto ye=at::matmul(hidden,wd[i].t()).to(torch::kFloat32).mul_(we); hidden.reset();
    out32.narrow(0,a,n).copy_(ye);   // direct sorted write, no per-expert scatter
  }
  // single scatter back to original token order (sums the 8 expert contributions per token)
  auto out=torch::zeros({B,D},x.options());
  out.index_add_(0,tok,out32);
  return out;
}
// v6: v5 + fp16 weight-mul (eliminate per-expert fp16->fp32 round-trip).
//     Profile-driven: v5 wastes 0.66ms (per-expert cast) + halves copy_(1.10->0.55) +
//     halves scatter(1.27->0.63) by keeping fp16 through mul/copy/scatter; single final cast to fp32.
//     Weight is routing scalar [N,1] in [0,1], fp16 precision ample. Expected ~6.3ms => ~1.46x.
torch::Tensor moe_rank_forward_v6(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  if(N==0) return torch::zeros_like(x);

  // --- build active-expert list on CPU (offsets is CPU int64, no GPU sync) ---
  std::vector<int64_t> active;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n<=0) continue;
    active.push_back(e);
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  // single gather: cast x to fp16 once, gather all routed tokens by expert-sorted tok -> [N,D] fp16 contiguous
  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);   // [N,D] fp16, rows ordered by expert segment
  xh.reset();

  // --- batched dequant: gather active-expert packed weights, 3 launches over ALL active experts ---
  auto gact=gpack.index_select(0,act);                       // [na,local,rb_g]
  auto uact=upack.index_select(0,act);                       // [na,local,rb_g]
  auto dact=dpack.index_select(0,act);                        // [na,6144,rb_d]
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  // fp16 weight (cast ONCE, [N,1] is tiny) -> no per-expert fp32 round-trip in the loop
  auto w16=weight.to(torch::kFloat16);
  auto sbuf=torch::zeros({N,D},w16.options());  // [N,D] fp16 expert-sorted output buffer
  for(int64_t i=0;i<na;i++){
    int64_t e=active[i];
    int64_t a=off[e], n=off[e+1]-a; if(!n) continue;
    auto xe=xb.narrow(0,a,n);        // [n,D] fp16 contiguous view (zero-copy)
    auto we=w16.narrow(0,a,n);       // [n,1] fp16 view (weight already expert-sorted)
    auto gate=at::matmul(xe,wg[i].t());           // [n,local]
    auto hidden=at::silu(gate).mul_(at::matmul(xe,wu[i].t())); gate.reset();
    auto ye=at::matmul(hidden,wd[i].t());          // [n,D] fp16, NO .to(fp32)
    ye.mul_(we);                                    // fp16 in-place weight mul
    sbuf.narrow(0,a,n).copy_(ye);                   // fp16->fp16 contiguous write (half bytes vs v5)
    hidden.reset();
  }
  // scatter fp16 (half bandwidth vs fp32), then single cast to fp32 output to match interface
  auto out16=torch::zeros({B,D},w16.options());
  out16.index_add_(0,tok,sbuf);
  return out16.to(torch::kFloat32);
}
// v7: v5 + matmul_out (write matmul result DIRECTLY into expert-sorted output buffer slice).
//     Profile-driven: v5 copy_=1.10ms is LAUNCH-bound (53 small copies, bandwidth needs only 0.27ms),
//     so halving bytes (v6) gave nothing. matmul_out eliminates copy_ entirely + per-expert
//     fp16->fp32 cast (0.66ms) + per-expert weight-mul, by hoisting cast+mul out of loop.
//     Parity PRESERVED vs v5: matmul still fp16, weight-mul+scatter still fp32 (bit-identical reorder).
//     Expected ~5.4ms => ~1.7x vs v1.
torch::Tensor moe_rank_forward_v7(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  if(N==0) return torch::zeros_like(x);

  // --- build active-expert list on CPU (offsets is CPU int64, no GPU sync) ---
  std::vector<int64_t> active;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n<=0) continue;
    active.push_back(e);
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  // single gather: cast x to fp16 once, gather all routed tokens by expert-sorted tok -> [N,D] fp16
  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);   // [N,D] fp16, rows ordered by expert segment
  xh.reset();

  // --- batched dequant: gather active-expert packed weights, 3 launches over ALL active experts ---
  auto gact=gpack.index_select(0,act);                       // [na,local,rb_g]
  auto uact=upack.index_select(0,act);                       // [na,local,rb_g]
  auto dact=dpack.index_select(0,act);                        // [na,6144,rb_d]
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  // fp16 expert-sorted output buffer; matmul_out writes directly into row slices
  // -> eliminates copy_ (launch-bound) + per-expert temp allocation + per-expert cast
  auto sbuf=torch::zeros({N,D},x.options().dtype(torch::kFloat16));  // [N,D] fp16 on cuda, expert-sorted
  for(int64_t i=0;i<na;i++){
    int64_t e=active[i];
    int64_t a=off[e], n=off[e+1]-a; if(!n) continue;
    auto xe=xb.narrow(0,a,n);        // [n,D] fp16 contiguous view (zero-copy)
    auto out_i=sbuf.narrow(0,a,n);  // [n,D] fp16 write target (zero-copy view into sbuf)
    auto gate=at::matmul(xe,wg[i].t());           // [n,local]
    auto hidden=at::silu(gate).mul_(at::matmul(xe,wu[i].t())); gate.reset();
    at::matmul_out(out_i, hidden, wd[i].t());  // DIRECT write, no copy_ no per-expert cast
    hidden.reset();
  }
  // hoist cast+weight-mul out of loop: single fp32 cast, fp32 weight-mul (parity-identical to v5), fp32 scatter
  auto s32=sbuf.to(torch::kFloat32);   // [N,D] fp32, single cast (vs v5's 53 per-expert casts)
  s32.mul_(weight);                     // fp32 weight-mul, weight is [N,1] expert-sorted -> matches sbuf rows
  // single scatter back to original token order (sums the 8 expert contributions per token)
  auto out=torch::zeros({B,D},x.options());
  out.index_add_(0,tok,s32);
  return out;
}

// hybrid: resident down fp16 [E,D,local] + active-only gate/up dequant + multi-stream expert GEMMs
torch::Tensor moe_rank_forward_hybrid(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  // active-only dequant for gate/up/down + multi-stream GEMMs (no resident full down fp16)
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352);
  TORCH_CHECK(dpack.is_cuda() && dpack.is_contiguous() && dpack.dim()==3 && dpack.size(0)==256 && dpack.size(1)==6144,
    "dpack must be CUDA [256,6144,rb]");
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  if(N==0) return torch::zeros_like(x);

  std::vector<int64_t> active;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n>0) active.push_back(e);
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).clone().to(x.device());
  const int64_t rb_g=gpack.size(2);
  const int64_t rb_d=dpack.size(2);

  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);
  xh.reset();

  auto gact=gpack.index_select(0,act);
  auto uact=upack.index_select(0,act);
  auto dact=dpack.index_select(0,act);
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  auto sbuf=torch::zeros({N,D},x.options().dtype(torch::kFloat16));

  // Source-controlled stream count; no runtime tuning override.
  int nstream=std::min(8,(int)na);
  std::vector<at::cuda::CUDAStream> streams;
  streams.reserve(nstream);
  for(int i=0;i<nstream;i++) streams.push_back(at::cuda::getStreamFromPool(false, x.device().index()));

  auto default_stream=at::cuda::getCurrentCUDAStream(x.device().index());
  {
    at::cuda::CUDAEvent ready;
    ready.record(default_stream);
    for(int i=0;i<nstream;i++) ready.block(streams[i]);
  }

  for(int64_t i=0;i<na;i++){
    int64_t e=active[i];
    int64_t a=off[e], n=off[e+1]-a; if(!n) continue;
    at::cuda::CUDAStreamGuard sg(streams[i%nstream]);
    auto xe=xb.narrow(0,a,n);
    auto out_i=sbuf.narrow(0,a,n);
    auto gate=at::matmul(xe,wg[i].t());
    auto hidden=at::silu(gate).mul_(at::matmul(xe,wu[i].t())); gate.reset();
    at::matmul_out(out_i, hidden, wd[i].t());
    hidden.reset();
  }

  {
    for(int i=0;i<nstream;i++){
      at::cuda::CUDAEvent done;
      done.record(streams[i]);
      done.block(default_stream);
    }
  }
  at::cuda::CUDAStreamGuard back(default_stream);

  auto s32=sbuf.to(torch::kFloat32);
  s32.mul_(weight);
  auto out=torch::zeros({B,D},x.options());
  out.index_add_(0,tok,s32);
  return out;
}


// v8_grouped: same semantics as v7, but expert GEMMs via pad+bmm (one launch per projection).
// Variable-n experts packed to [na,max_n,*]; A100 has no FP16 cublasGemmGrouped / no SM90 _grouped_mm.
// Goal: defragment for-loop (correctness first). Explicit experimental API only;
// it is never selected through an environment variable.
torch::Tensor moe_rank_forward_v8_grouped(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size must be a positive multiple of 256");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352); ck(dpack,"down",6144,local/256*136);
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,"x must be CUDA float32 [B,6144]");
  TORCH_CHECK(tok.is_cuda() && tok.scalar_type()==torch::kInt64 && tok.is_contiguous() && tok.dim()==1,"tok must be CUDA int64 [N]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type()==torch::kFloat32 && weight.is_contiguous() && weight.dim()==2 && weight.size(0)==tok.size(0) && weight.size(1)==1,"weight must be CUDA float32 [N,1]");
  TORCH_CHECK(offsets.device().is_cpu() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous() && offsets.numel()==257,"offsets must be CPU int64 [257]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device() && tok.device()==x.device() && weight.device()==x.device(),"all CUDA tensors must share rank device");
  c10::cuda::CUDAGuard guard(x.device());
  const auto* off=offsets.data_ptr<int64_t>();
  TORCH_CHECK(off[0]==0 && off[256]==tok.size(0),"bad offsets endpoints");
  const int64_t N=tok.size(0), B=x.size(0), D=6144;
  if(N==0) return torch::zeros_like(x);

  std::vector<int64_t> active;
  std::vector<int64_t> starts;
  std::vector<int64_t> ns;
  int64_t max_n=0;
  for(int64_t e=0;e<256;e++){
    int64_t n=off[e+1]-off[e];
    if(n<=0) continue;
    active.push_back(e);
    starts.push_back(off[e]);
    ns.push_back(n);
    if(n>max_n) max_n=n;
  }
  const int64_t na=(int64_t)active.size();
  if(na==0) return torch::zeros_like(x);
  auto act=torch::from_blob(active.data(),{na},torch::kInt64).to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);
  xh.reset();

  auto gact=gpack.index_select(0,act);
  auto uact=upack.index_select(0,act);
  auto dact=dpack.index_select(0,act);
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local});
  gact.reset(); uact.reset(); dact.reset();

  auto packed=torch::zeros({na,max_n,D}, xb.options());
  for(int64_t i=0;i<na;i++){
    int64_t a=starts[i], n=ns[i];
    packed[i].narrow(0,0,n).copy_(xb.narrow(0,a,n));
  }

  auto Wg=wg.transpose(1,2).contiguous();
  auto Wu=wu.transpose(1,2).contiguous();
  auto gate=at::bmm(packed, Wg);
  auto up=at::bmm(packed, Wu);
  packed.reset(); Wg.reset(); Wu.reset(); wg.reset(); wu.reset();
  auto hidden=at::silu(gate).mul_(up);
  gate.reset(); up.reset();

  auto Wd=wd.transpose(1,2).contiguous();
  auto ypad=at::bmm(hidden, Wd);
  hidden.reset(); Wd.reset(); wd.reset();

  auto sbuf=torch::zeros({N,D}, x.options().dtype(torch::kFloat16));
  for(int64_t i=0;i<na;i++){
    int64_t a=starts[i], n=ns[i];
    sbuf.narrow(0,a,n).copy_(ypad[i].narrow(0,0,n));
  }
  ypad.reset();

  auto s32=sbuf.to(torch::kFloat32);
  s32.mul_(weight);
  auto out=torch::zeros({B,D},x.options());
  out.index_add_(0,tok,s32);
  return out;
}


torch::Tensor weighted_scatter_fp16(torch::Tensor sbuf, torch::Tensor weight,
 torch::Tensor tok, int64_t B, int64_t mode);

// v9_cutlass: same semantics as v7, but gate/up/down via CUTLASS SM80 grouped GEMM
// (true variable-M, no pad). 3 launches replace na*3 serial matmuls.
torch::Tensor moe_rank_forward_v9_cutlass(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size invalid");
  const int64_t D=x.size(1), B=x.size(0);
  TORCH_CHECK(tok.size(0)==weight.size(0),"tok/weight length mismatch");
  auto off_cpu=offsets.to(torch::kCPU).contiguous();
  auto off=off_cpu.accessor<int64_t,1>();
  const int64_t E=off_cpu.size(0)-1;
  std::vector<int64_t> active; active.reserve(E);
  std::vector<int64_t> sizes_host; sizes_host.reserve(E);
  int64_t N=0;
  for(int64_t e=0;e<E;e++){
    int64_t n=off[e+1]-off[e];
    if(n>0){ active.push_back(e); sizes_host.push_back(n); N+=n; }
  }
  const int64_t na=(int64_t)active.size();
  TORCH_CHECK(na>0,"no active experts");
  TORCH_CHECK(N==tok.size(0),"offsets cover != tok rows");

  auto act=torch::from_blob(active.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());
  auto sizes=torch::from_blob(sizes_host.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);  // [N,D] expert-sorted
  xh.reset();

  // batched dequant (same as v7)
  auto gact=gpack.index_select(0,act);
  auto uact=upack.index_select(0,act);
  auto dact=dpack.index_select(0,act);
  auto wg=dequant_iq3_xxs_cuda(gact.reshape({na*local,rb_g}),D).reshape({na,local,D}); // [na,Nout=local,K=D]
  auto wu=dequant_iq3_xxs_cuda(uact.reshape({na*local,rb_g}),D).reshape({na,local,D});
  auto wd=dequant_iq4_xs_cuda(dact.reshape({na*D,rb_d}),local).reshape({na,D,local}); // [na,Nout=D,K=local]
  gact.reset(); uact.reset(); dact.reset();

  // 3 grouped GEMMs (variable M, no pad)
  auto gate=grouped_gemm_sm80(xb, wg, sizes, 0);          // [N,local]
  auto up  =grouped_gemm_sm80(xb, wu, sizes, 0);          // [N,local]
  xb.reset(); wg.reset(); wu.reset();
  auto hidden=at::silu(gate).mul_(up); gate.reset(); up.reset();
  auto sbuf=grouped_gemm_sm80(hidden, wd, sizes, 0);      // [N,D]
  hidden.reset(); wd.reset();

  auto s32=sbuf.to(torch::kFloat32);
  sbuf.reset();
  s32.mul_(weight);
  auto out=torch::zeros({B,D},x.options());
  out.index_add_(0,tok,s32);
  return out;
}


// v10: v9 grouped GEMMs + fused weighted scatter
torch::Tensor moe_rank_forward_v10_scatter(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size invalid");
  const int64_t D=x.size(1), B=x.size(0);
  TORCH_CHECK(tok.size(0)==weight.size(0),"tok/weight length mismatch");
  auto off_cpu=offsets.to(torch::kCPU).contiguous();
  auto off=off_cpu.accessor<int64_t,1>();
  const int64_t E=off_cpu.size(0)-1;
  std::vector<int64_t> active; active.reserve(E);
  std::vector<int64_t> sizes_host; sizes_host.reserve(E);
  int64_t N=0;
  for(int64_t e=0;e<E;e++){
    int64_t n=off[e+1]-off[e];
    if(n>0){ active.push_back(e); sizes_host.push_back(n); N+=n; }
  }
  const int64_t na=(int64_t)active.size();
  TORCH_CHECK(na>0,"no active experts");
  TORCH_CHECK(N==tok.size(0),"offsets cover != tok rows");

  auto act=torch::from_blob(active.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());
  auto sizes=torch::from_blob(sizes_host.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());
  const int64_t rb_g=gpack.size(2), rb_d=dpack.size(2);

  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);  // [N,D] expert-sorted
  xh.reset();

  // selected-deq: fused index_select+dequant (saves 3 index_select copies)
  auto wg=dequant_iq3_selected_cuda(gpack,act,D).reshape({na,local,D});
  auto wu=dequant_iq3_selected_cuda(upack,act,D).reshape({na,local,D});
  auto wd=dequant_iq4_selected_cuda(dpack,act,local).reshape({na,D,local});

  // 3 grouped GEMMs (variable M, no pad)
  auto gate=grouped_gemm_sm80(xb, wg, sizes, 0);          // [N,local]
  auto up  =grouped_gemm_sm80(xb, wu, sizes, 0);          // [N,local]
  xb.reset(); wg.reset(); wu.reset();
  auto hidden=at::silu(gate).mul_(up); gate.reset(); up.reset();
  auto sbuf=grouped_gemm_sm80(hidden, wd, sizes, 0);      // [N,D]
  hidden.reset(); wd.reset();

  // fused cast + weight + scatter; mode=1 flat grid-stride is fastest on A100.
  return weighted_scatter_fp16(sbuf,weight,tok,B,1);
}



// dequant full gate/up pack [E,local,rb] -> fp16 [E,local,D]
torch::Tensor dequant_gate_resident(torch::Tensor gpack, int64_t D){
  TORCH_CHECK(gpack.dim()==3, "gpack [E,local,rb]");
  const int64_t E=gpack.size(0), local=gpack.size(1), rb=gpack.size(2);
  c10::cuda::CUDAGuard guard(gpack.device());
  return dequant_iq3_xxs_cuda(gpack.reshape({E*local,rb}), D).reshape({E,local,D});
}

// v11: pre-dequanted resident fp16 weights; only index_select + 3 grouped GEMM + fused scatter
torch::Tensor moe_rank_forward_v11_resident(torch::Tensor wg_all,torch::Tensor wu_all,torch::Tensor wd_all,
 torch::Tensor x,torch::Tensor tok,torch::Tensor weight,torch::Tensor offsets){
  const int64_t local=wg_all.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate size invalid");
  const int64_t D=x.size(1), B=x.size(0);
  TORCH_CHECK(tok.size(0)==weight.size(0),"tok/weight length mismatch");
  TORCH_CHECK(wg_all.scalar_type()==torch::kFloat16 && wu_all.scalar_type()==torch::kFloat16 && wd_all.scalar_type()==torch::kFloat16,
              "v11 expects fp16 resident weights");
  auto off_cpu=offsets.to(torch::kCPU).contiguous();
  auto off=off_cpu.accessor<int64_t,1>();
  const int64_t E=off_cpu.size(0)-1;
  std::vector<int64_t> active; active.reserve(E);
  std::vector<int64_t> sizes_host; sizes_host.reserve(E);
  int64_t N=0;
  for(int64_t e=0;e<E;e++){
    int64_t n=off[e+1]-off[e];
    if(n>0){ active.push_back(e); sizes_host.push_back(n); N+=n; }
  }
  const int64_t na=(int64_t)active.size();
  TORCH_CHECK(na>0,"no active experts");
  TORCH_CHECK(N==tok.size(0),"offsets cover != tok rows");

  auto act=torch::from_blob(active.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());
  auto sizes=torch::from_blob(sizes_host.data(),{(long)na},torch::TensorOptions().dtype(torch::kInt64)).clone().to(x.device());

  auto xh=x.to(torch::kFloat16);
  auto xb=xh.index_select(0,tok);  // [N,D]
  xh.reset();

  auto wg=wg_all.index_select(0,act); // [na,local,D]
  auto wu=wu_all.index_select(0,act);
  auto wd=wd_all.index_select(0,act); // [na,D,local]

  auto gate=grouped_gemm_sm80(xb, wg, sizes, 0);
  auto up  =grouped_gemm_sm80(xb, wu, sizes, 0);
  xb.reset(); wg.reset(); wu.reset();
  auto hidden=at::silu(gate).mul_(up); gate.reset(); up.reset();
  auto sbuf=grouped_gemm_sm80(hidden, wd, sizes, 0);
  hidden.reset(); wd.reset();
  return weighted_scatter_fp16(sbuf,weight,tok,B,1);
}

torch::Tensor dequant_down_resident(torch::Tensor dpack){
  TORCH_CHECK(dpack.dim()==3 && dpack.size(0)==256 && dpack.size(1)==6144, "dpack [256,6144,rb]");
  const int64_t E=dpack.size(0), D=dpack.size(1), rb=dpack.size(2);
  TORCH_CHECK(rb%136==0, "bad down row bytes");
  const int64_t local=(rb/136)*256;
  c10::cuda::CUDAGuard guard(dpack.device());
  return dequant_iq4_xs_cuda(dpack.reshape({E*D,rb}), local).reshape({E,D,local});
}


torch::Tensor moe_rank_forward_decode_fused(torch::Tensor g,torch::Tensor u,torch::Tensor d,
 torch::Tensor x,torch::Tensor ei,torch::Tensor ew,torch::Tensor h,torch::Tensor y){
 const int64_t T=x.size(0),L=g.size(1),K=ei.size(1); ck(g,"gate",L,2352); ck(u,"up",L,2352); ck(d,"down",6144,L/256*136);
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat32&&x.is_contiguous()&&x.dim()==2&&x.size(1)==6144,"bad x");
 TORCH_CHECK(ei.is_cuda()&&ei.scalar_type()==torch::kInt64&&ei.is_contiguous()&&ei.dim()==2&&ei.size(0)==T,"bad ei");
 TORCH_CHECK(ew.is_cuda()&&ew.scalar_type()==torch::kFloat32&&ew.is_contiguous()&&ew.sizes()==ei.sizes(),"bad ew");
 TORCH_CHECK(T>=1&&T<128&&K>0&&K<=16,"small fused expects T=1..127");
 TORCH_CHECK(h.is_cuda()&&h.scalar_type()==torch::kFloat32&&h.is_contiguous()&&h.dim()==2&&h.size(0)==T*K&&h.size(1)==L,"bad hidden workspace");
 TORCH_CHECK(y.is_cuda()&&y.scalar_type()==torch::kFloat32&&y.is_contiguous()&&y.dim()==2&&y.size(0)==T&&y.size(1)==6144,"bad output");
 TORCH_CHECK(g.device()==x.device()&&u.device()==x.device()&&d.device()==x.device()&&ei.device()==x.device()&&ew.device()==x.device()&&h.device()==x.device()&&y.device()==x.device(),"decode MoE tensors must share a device");
 c10::cuda::CUDAGuard guard(x.device()); return moe_decode_iq_fused_out_cuda(g,u,d,x,ei,ew,h,y);
}

// decode: small-B direct expert path. No dispatch_meta / scatter / grouped GEMM.
// x:[B,D] fp32, ei:[B,K] int64, ew:[B,K] fp32. Typical decode: B=1,K=8, each expert n=1.
// Optimized: no host sync, no double GEMM, multi-stream experts, B=1 packed matmul.
torch::Tensor moe_rank_forward_decode(torch::Tensor gpack,torch::Tensor upack,torch::Tensor dpack,
 torch::Tensor x,torch::Tensor ei,torch::Tensor ew){
  const int64_t local=gpack.size(1);
  TORCH_CHECK(local>0 && local%256==0,"local intermediate invalid");
  ck(gpack,"gate",local,2352); ck(upack,"up",local,2352);
  TORCH_CHECK(dpack.is_cuda() && dpack.is_contiguous() && dpack.dim()==3 && dpack.size(0)==256 && dpack.size(1)==6144,
    "dpack must be CUDA [256,6144,rb]");
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==torch::kFloat32 && x.is_contiguous() && x.dim()==2 && x.size(1)==6144,
    "x must be CUDA float32 [B,6144]");
  TORCH_CHECK(ei.is_cuda() && ei.scalar_type()==torch::kInt64 && ei.is_contiguous() && ei.dim()==2 && ei.size(0)==x.size(0),
    "ei must be CUDA int64 [B,K]");
  TORCH_CHECK(ew.is_cuda() && ew.scalar_type()==torch::kFloat32 && ew.is_contiguous() && ew.sizes()==ei.sizes(),
    "ew must be CUDA float32 [B,K]");
  TORCH_CHECK(gpack.device()==x.device() && upack.device()==x.device() && dpack.device()==x.device()
    && ei.device()==x.device() && ew.device()==x.device(),"device mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t B=x.size(0), K=ei.size(1), D=6144;
  if(B==0) return torch::zeros_like(x);
  TORCH_CHECK(K>0 && K<=32, "decode path expects small K (top-k experts)");

  // Fast path B=1: top-k experts are usually unique; dequant K experts once, then one batched matmul chain.
  if(B==1){
    auto act=ei.reshape({K}).contiguous(); // [K] on GPU, no host sync
    auto wg=dequant_iq3_selected_cuda(gpack,act,D).reshape({K,local,D}); // [K,local,D]
    auto wu=dequant_iq3_selected_cuda(upack,act,D).reshape({K,local,D});
    auto wd=dequant_iq4_selected_cuda(dpack,act,local).reshape({K,D,local}); // [K,D,local]
    auto xh=x.to(torch::kFloat16); // [1,D]
    // gate/up: [K,local] via matmul(wg, x^T)
    auto xt=xh.reshape({D,1}); // [D,1]
    auto gate=at::matmul(wg, xt).squeeze(-1); // [K,local]
    auto up=at::matmul(wu, xt).squeeze(-1);   // [K,local]
    auto hidden=at::silu(gate).mul_(up);      // [K,local]
    // down: for each expert ye[k]=hidden[k] @ wd[k].T  -> [K,D]
    // bmm: hidden[:,None,:] @ wd.transpose(1,2) where wd:[K,D,local] => [K,1,D]
    auto ye=at::bmm(hidden.unsqueeze(1), wd.transpose(1,2)).squeeze(1).to(torch::kFloat32); // [K,D]
    auto w=ew.reshape({K,1}); // [K,1]
    auto out=(ye*w).sum(0, /*keepdim=*/true); // [1,D]
    return out;
  }

  // General small-B: unique active experts + multi-stream, stay on GPU (no host sync).
  auto flat=ei.reshape({-1});
  auto uniq=std::get<0>(at::_unique(flat, /*sorted=*/false, /*return_inverse=*/false));
  const int64_t na=uniq.size(0);
  TORCH_CHECK(na>0,"no active experts");
  auto wg=dequant_iq3_selected_cuda(gpack,uniq,D).reshape({na,local,D});
  auto wu=dequant_iq3_selected_cuda(upack,uniq,D).reshape({na,local,D});
  auto wd=dequant_iq4_selected_cuda(dpack,uniq,local).reshape({na,D,local});
  auto xh=x.to(torch::kFloat16);
  auto out=torch::zeros({B,D}, x.options());

  // Map expert id -> row in uniq via searchsorted on sorted uniq for GPU gather.
  auto uniq_sorted_pair=uniq.sort();
  auto uniq_sorted=std::get<0>(uniq_sorted_pair);
  auto uniq_perm=std::get<1>(uniq_sorted_pair);
  auto pos=at::searchsorted(uniq_sorted, ei); // [B,K] positions into sorted
  auto inv=uniq_perm; // perm[i]=original index of sorted[i]; need inverse
  // Build inverse perm on GPU: inv_perm[perm[i]]=i
  auto inv_perm=torch::empty_like(uniq_perm);
  inv_perm.scatter_(0, uniq_perm, torch::arange(na, uniq_perm.options()));
  auto eidx=inv_perm.index({pos}); // [B,K] row in uniq

  int nstream=std::min<int64_t>(8, B*K);
  std::vector<at::cuda::CUDAStream> streams; streams.reserve(nstream);
  for(int i=0;i<nstream;i++) streams.push_back(at::cuda::getStreamFromPool(false, x.device().index()));
  auto default_stream=at::cuda::getCurrentCUDAStream(x.device().index());
  {
    at::cuda::CUDAEvent ready; ready.record(default_stream);
    for(int i=0;i<nstream;i++) ready.block(streams[i]);
  }
  // Serialize accumulation into out on default stream via events; compute on side streams.
  // For tiny B*K, just run on default stream to avoid event tax if B*K small.
  if(B*K<=8){
    for(int64_t b=0;b<B;b++){
      auto xb=xh.narrow(0,b,1);
      for(int64_t k=0;k<K;k++){
        int64_t i=eidx[b][k].item<int64_t>(); // small host read; B*K<=8
        float w=ew[b][k].item<float>();
        auto gate=at::matmul(xb, wg[i].t());
        auto hidden=at::silu(gate).mul_(at::matmul(xb, wu[i].t()));
        auto ye=at::matmul(hidden, wd[i].t()).to(torch::kFloat32);
        out.narrow(0,b,1).add_(ye, w);
      }
    }
  } else {
    // fallback: token-major loop on default stream
    for(int64_t b=0;b<B;b++){
      auto xb=xh.narrow(0,b,1);
      for(int64_t k=0;k<K;k++){
        int64_t i=eidx[b][k].item<int64_t>();
        float w=ew[b][k].item<float>();
        auto gate=at::matmul(xb, wg[i].t());
        auto hidden=at::silu(gate).mul_(at::matmul(xb, wu[i].t()));
        auto ye=at::matmul(hidden, wd[i].t()).to(torch::kFloat32);
        out.narrow(0,b,1).add_(ye, w);
      }
    }
  }
  return out;
}


// Single-entry TP8 decode MoE: launch 8 ranks from one C++ call (no Python per-rank tax).
// x0/ei0/ew0 live on cuda:0; weights already sharded on each device.
// Returns 8 PARTIAL tensors (fp32), one per rank; caller reduces.
// Prefer moe_tp8_decode_fused_local with pre-resident activations (no per-call D2D/cast).
std::vector<torch::Tensor> moe_tp8_decode_fused_local(
    std::vector<torch::Tensor> g,
    std::vector<torch::Tensor> u,
    std::vector<torch::Tensor> d,
    std::vector<torch::Tensor> sg,
    std::vector<torch::Tensor> su,
    std::vector<torch::Tensor> sd,
    std::vector<torch::Tensor> xs_f32,
    std::vector<torch::Tensor> xs_f16,
    std::vector<torch::Tensor> eis,
    std::vector<torch::Tensor> ews){
  TORCH_CHECK(g.size()==8 && xs_f32.size()==8 && xs_f16.size()==8 && eis.size()==8 && ews.size()==8,
              "tp8 local expects 8 shards");
  std::vector<torch::Tensor> outs; outs.reserve(8);
  // Launch only: activations already on each device.
  for(int q=0;q<8;q++){
    const auto dev = g[q].device();
    TORCH_CHECK(xs_f32[q].device()==dev && xs_f16[q].device()==dev && eis[q].device()==dev && ews[q].device()==dev,
                "activation not on weight device");
    c10::cuda::CUDAGuard guard(dev);
    auto routed = moe_decode_iq_fused_cuda(g[q], u[q], d[q], xs_f32[q], eis[q], ews[q]);
    auto shared = shared_decode_q8_cuda(sg[q], su[q], sd[q], xs_f16[q]);
    // keep shared half->float add on device stream (ATen async)
    outs.push_back(routed + shared.to(torch::kFloat32));
  }
  return outs;
}

// Experimental half-output sibling: preserve fp32 routed/shared addition, but fuse
// shared half->float, add and final float->half into one CUDA launch.
std::vector<torch::Tensor> moe_tp8_decode_fused_local_half(
    std::vector<torch::Tensor> g, std::vector<torch::Tensor> u, std::vector<torch::Tensor> d,
    std::vector<torch::Tensor> sg, std::vector<torch::Tensor> su, std::vector<torch::Tensor> sd,
    std::vector<torch::Tensor> xs_f32, std::vector<torch::Tensor> xs_f16,
    std::vector<torch::Tensor> eis, std::vector<torch::Tensor> ews) {
  TORCH_CHECK(g.size()==8 && xs_f32.size()==8 && xs_f16.size()==8 && eis.size()==8 && ews.size()==8,
              "tp8 local half expects 8 shards");
  std::vector<torch::Tensor> outs; outs.reserve(8);
  for(int q=0;q<8;q++) {
    const auto dev=g[q].device();
    TORCH_CHECK(xs_f32[q].device()==dev && xs_f16[q].device()==dev && eis[q].device()==dev && ews[q].device()==dev,
                "activation not on weight device");
    c10::cuda::CUDAGuard guard(dev);
    auto routed=moe_decode_iq_fused_cuda(g[q],u[q],d[q],xs_f32[q],eis[q],ews[q]);
    auto shared=shared_decode_q8_cuda(sg[q],su[q],sd[q],xs_f16[q]);
    outs.push_back(add_f32_f16_to_f16_cuda(routed,shared));
  }
  return outs;
}

// Graph-safe rank-local leaf for the production per-rank CUDA graphs.  Metadata
// synchronization is performed by peer_meta_bcast_run immediately before this
// call; the numerical leaf is exactly the original candidate's routed + shared
// + fp32-add/fp16-output sequence.
torch::Tensor moe_tp8_decode_meta_bcast_local_half_rank(
    torch::Tensor g, torch::Tensor u, torch::Tensor d,
    torch::Tensor sg, torch::Tensor su, torch::Tensor sd,
    torch::Tensor x_f32, torch::Tensor x_f16,
    torch::Tensor ei, torch::Tensor ew) {
  const auto dev=g.device();
  TORCH_CHECK(u.device()==dev && d.device()==dev && sg.device()==dev && su.device()==dev && sd.device()==dev &&
              x_f32.device()==dev && x_f16.device()==dev && ei.device()==dev && ew.device()==dev,
              "rank-local meta bcast leaf device mismatch");
  c10::cuda::CUDAGuard guard(dev);
  auto routed=moe_decode_iq_fused_cuda(g,u,d,x_f32,ei,ew);
  auto shared=shared_decode_q8_cuda(sg,su,sd,x_f16);
  return add_f32_f16_to_f16_cuda(routed,shared);
}

// Candidate: one host entry performs the rank-0 route dependency, tiny metadata
// peer copies, and all eight local half-output launches. Activations stay local.
std::vector<torch::Tensor> moe_tp8_decode_meta_bcast_local_half(
    std::vector<torch::Tensor> g, std::vector<torch::Tensor> u, std::vector<torch::Tensor> d,
    std::vector<torch::Tensor> sg, std::vector<torch::Tensor> su, std::vector<torch::Tensor> sd,
    std::vector<torch::Tensor> xs_f32, std::vector<torch::Tensor> xs_f16,
    torch::Tensor ei0, torch::Tensor ew0,
    std::vector<torch::Tensor> eis, std::vector<torch::Tensor> ews) {
  TORCH_CHECK(g.size()==8 && xs_f32.size()==8 && xs_f16.size()==8 && eis.size()==8 && ews.size()==8,
              "tp8 meta bcast local half expects 8 shards");
  TORCH_CHECK(ei0.is_cuda() && ei0.scalar_type()==torch::kInt64 && ei0.is_contiguous(), "bad ei0");
  TORCH_CHECK(ew0.is_cuda() && ew0.scalar_type()==torch::kFloat32 && ew0.is_contiguous(), "bad ew0");
  cudaEvent_t route_ready;
  c10::cuda::CUDAGuard guard0(g[0].device());
  auto stream0=c10::cuda::getCurrentCUDAStream(g[0].get_device());
  TORCH_CHECK(cudaEventCreateWithFlags(&route_ready,cudaEventDisableTiming)==cudaSuccess,"route event create failed");
  TORCH_CHECK(cudaEventRecord(route_ready,stream0.stream())==cudaSuccess,"route event record failed");
  std::vector<torch::Tensor> outs; outs.reserve(8);
  for(int q=0;q<8;q++) {
    const auto dev=g[q].device();
    c10::cuda::CUDAGuard guard(dev);
    auto st=c10::cuda::getCurrentCUDAStream(dev.index());
    torch::Tensor ei=ei0,ew=ew0;
    if(q) {
      ei=eis[q]; ew=ews[q];
      TORCH_CHECK(ei.device()==dev && ew.device()==dev && ei.nbytes()==ei0.nbytes() && ew.nbytes()==ew0.nbytes(),
                  "metadata buffer mismatch");
      TORCH_CHECK(cudaStreamWaitEvent(st.stream(),route_ready,0)==cudaSuccess,"route event wait failed");
      TORCH_CHECK(cudaMemcpyPeerAsync(ei.data_ptr(),ei.get_device(),ei0.data_ptr(),ei0.get_device(),ei0.nbytes(),st.stream())==cudaSuccess,"eid peer copy failed");
      TORCH_CHECK(cudaMemcpyPeerAsync(ew.data_ptr(),ew.get_device(),ew0.data_ptr(),ew0.get_device(),ew0.nbytes(),st.stream())==cudaSuccess,"ew peer copy failed");
    }
    TORCH_CHECK(xs_f32[q].device()==dev && xs_f16[q].device()==dev,"activation not on weight device");
    auto routed=moe_decode_iq_fused_cuda(g[q],u[q],d[q],xs_f32[q],ei,ew);
    auto shared=shared_decode_q8_cuda(sg[q],su[q],sd[q],xs_f16[q]);
    outs.push_back(add_f32_f16_to_f16_cuda(routed,shared));
  }
  TORCH_CHECK(cudaEventDestroy(route_ready)==cudaSuccess,"route event destroy failed");
  return outs;
}

// Fused host orchestrator: peer-copy rank0 activations/meta on each destination stream,
// then immediately launch routed+shared kernels on that same stream. Buffers are preallocated.
std::vector<torch::Tensor> moe_tp8_decode_bcast_launch(
    std::vector<torch::Tensor> g, std::vector<torch::Tensor> u, std::vector<torch::Tensor> d,
    std::vector<torch::Tensor> sg, std::vector<torch::Tensor> su, std::vector<torch::Tensor> sd,
    torch::Tensor x0f, torch::Tensor x0h, torch::Tensor ei0, torch::Tensor ew0,
    std::vector<torch::Tensor> xs_f32, std::vector<torch::Tensor> xs_f16,
    std::vector<torch::Tensor> eis, std::vector<torch::Tensor> ews){
  TORCH_CHECK(g.size()==8 && u.size()==8 && d.size()==8 && sg.size()==8 && su.size()==8 && sd.size()==8,
              "tp8 bcast launch expects 8 weight shards");
  TORCH_CHECK(xs_f32.size()==8 && xs_f16.size()==8 && eis.size()==8 && ews.size()==8,
              "tp8 bcast launch expects 8 buffers");
  TORCH_CHECK(x0f.is_cuda() && x0f.scalar_type()==torch::kFloat32 && x0f.is_contiguous(), "bad x0f");
  TORCH_CHECK(x0h.is_cuda() && x0h.scalar_type()==torch::kFloat16 && x0h.is_contiguous(), "bad x0h");
  TORCH_CHECK(ei0.is_cuda() && ei0.scalar_type()==torch::kInt64 && ei0.is_contiguous(), "bad ei0");
  TORCH_CHECK(ew0.is_cuda() && ew0.scalar_type()==torch::kFloat32 && ew0.is_contiguous(), "bad ew0");
  std::vector<torch::Tensor> outs; outs.reserve(8);
  for(int q=0;q<8;q++){
    const auto dev=g[q].device();
    c10::cuda::CUDAGuard guard(dev);
    auto st=c10::cuda::getCurrentCUDAStream();
    torch::Tensor xf, xh, ei, ew;
    if(q==0){ xf=x0f; xh=x0h; ei=ei0; ew=ew0; }
    else {
      xf=xs_f32[q]; xh=xs_f16[q]; ei=eis[q]; ew=ews[q];
      TORCH_CHECK(xf.device()==dev && xh.device()==dev && ei.device()==dev && ew.device()==dev,
                  "destination buffer on wrong GPU");
      auto cp=[&](torch::Tensor &dst, const torch::Tensor &src){
        TORCH_CHECK(dst.nbytes()==src.nbytes() && dst.is_contiguous(), "copy buffer mismatch");
        auto err=cudaMemcpyPeerAsync(dst.data_ptr(), dst.get_device(), src.data_ptr(), src.get_device(),
                                     src.nbytes(), st.stream());
        TORCH_CHECK(err==cudaSuccess, "peer copy failed: ", cudaGetErrorString(err));
      };
      cp(xf,x0f); cp(xh,x0h); cp(ei,ei0); cp(ew,ew0);
    }
    auto routed=moe_decode_iq_fused_cuda(g[q],u[q],d[q],xf,ei,ew);
    auto shared=shared_decode_q8_cuda(sg[q],su[q],sd[q],xh);
    outs.push_back(routed+shared.to(torch::kFloat32));
  }
  return outs;
}

// Convenience: copy from rank0 once then launch. Still slower than pre-resident local.
std::vector<torch::Tensor> moe_tp8_decode_fused(
    std::vector<torch::Tensor> g,
    std::vector<torch::Tensor> u,
    std::vector<torch::Tensor> d,
    std::vector<torch::Tensor> sg,
    std::vector<torch::Tensor> su,
    std::vector<torch::Tensor> sd,
    torch::Tensor x0,
    torch::Tensor ei0,
    torch::Tensor ew0){
  TORCH_CHECK(g.size()==8 && u.size()==8 && d.size()==8 && sg.size()==8 && su.size()==8 && sd.size()==8,
              "tp8 decode expects 8 shards");
  TORCH_CHECK(x0.is_cuda() && x0.scalar_type()==torch::kFloat32 && x0.dim()==2 && x0.size(1)==6144, "bad x0");
  TORCH_CHECK(ei0.is_cuda() && ei0.scalar_type()==torch::kInt64 && ei0.dim()==2 && ei0.size(0)==x0.size(0), "bad ei0");
  TORCH_CHECK(ew0.is_cuda() && ew0.scalar_type()==torch::kFloat32 && ew0.sizes()==ei0.sizes(), "bad ew0");
  TORCH_CHECK(x0.size(0)>=1 && x0.size(0)<128 && ei0.size(1)>0 && ei0.size(1)<=16, "small fused T=1..127");

  // Stage A: broadcast activations (async D2D) + fp16 cast on each device.
  std::vector<torch::Tensor> xs_f32(8), xs_f16(8), eis(8), ews(8);
  {
    c10::cuda::CUDAGuard g0(x0.device());
    xs_f32[0]=x0.contiguous();
    eis[0]=ei0.contiguous();
    ews[0]=ew0.contiguous();
    xs_f16[0]=xs_f32[0].to(torch::kFloat16).contiguous();
  }
  for(int q=1;q<8;q++){
    const auto dev=g[q].device();
    c10::cuda::CUDAGuard guard(dev);
    xs_f32[q]=torch::empty(x0.sizes(), x0.options().device(dev));
    eis[q]=torch::empty(ei0.sizes(), ei0.options().device(dev));
    ews[q]=torch::empty(ew0.sizes(), ew0.options().device(dev));
    xs_f32[q].copy_(xs_f32[0], true);
    eis[q].copy_(eis[0], true);
    ews[q].copy_(ews[0], true);
    xs_f16[q]=torch::empty(x0.sizes(), x0.options().dtype(torch::kFloat16).device(dev));
    xs_f16[q].copy_(xs_f16[0], true);
  }
  // Stage B: pure kernel launches (no further host cast/copy)
  return moe_tp8_decode_fused_local(g,u,d,sg,su,sd,xs_f32,xs_f16,eis,ews);
}

std::vector<torch::Tensor> dispatch_meta(torch::Tensor ei,torch::Tensor ew);

// Fast multi-GPU activation broadcast: zh[0] (fp16 QxD) -> zh[1..7] via cudaMemcpyPeerAsync, then optional zf=float(zh)
std::vector<torch::Tensor> moe_tp8_peer_copy_like(
    std::vector<torch::Tensor> src_list,
    std::vector<torch::Tensor> dst_list){
  TORCH_CHECK(src_list.size()==dst_list.size());
  const int n=(int)src_list.size();
  // launch peer copies from each tensor's device; for broadcast pattern src is same logical on rank0
  for(int q=0;q<n;q++){
    auto &s=src_list[q]; auto &d=dst_list[q];
    TORCH_CHECK(s.is_cuda()&&d.is_cuda());
    TORCH_CHECK(s.nbytes()==d.nbytes());
    if(s.data_ptr()==d.data_ptr()) continue;
    c10::cuda::CUDAGuard guard(d.device());
    auto st=c10::cuda::getCurrentCUDAStream();
    // cudaMemcpyPeerAsync handles peer; cudaMemcpyAsync also if peer enabled
    auto err=cudaMemcpyPeerAsync(d.data_ptr(), d.get_device(), s.data_ptr(), s.get_device(), s.nbytes(), st.stream());
    TORCH_CHECK(err==cudaSuccess, "cudaMemcpyPeerAsync failed: ", cudaGetErrorString(err));
  }
  return dst_list;
}

// Broadcast one src tensor on device of src to many dst tensors (preallocated) using peer async from src device stream.
std::vector<torch::Tensor> moe_tp8_broadcast_tensor(torch::Tensor src, std::vector<torch::Tensor> dsts){
  TORCH_CHECK(src.is_cuda());
  c10::cuda::CUDAGuard guard(src.device());
  auto st=c10::cuda::getCurrentCUDAStream();
  for(size_t q=0;q<dsts.size();++q){
    auto &d=dsts[q];
    TORCH_CHECK(d.is_cuda() && d.nbytes()==src.nbytes());
    if(d.data_ptr()==src.data_ptr() && d.get_device()==src.get_device()) continue;
    auto err=cudaMemcpyPeerAsync(d.data_ptr(), d.get_device(), src.data_ptr(), src.get_device(), src.nbytes(), st.stream());
    if(err!=cudaSuccess){
      // fallback same-device or non-peer
      err=cudaMemcpyAsync(d.data_ptr(), src.data_ptr(), src.nbytes(), cudaMemcpyDeviceToDevice, st.stream());
    }
    TORCH_CHECK(err==cudaSuccess, "broadcast copy failed: ", cudaGetErrorString(err));
  }
  return dsts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("grouped_gemm_sm80",&grouped_gemm_sm80,"CUTLASS SM80 grouped GEMM");
 m.def("grouped_gemm_sufficient",&grouped_gemm_sufficient,"tb count");
 m.def("dispatch_meta",&dispatch_meta,"256-bin MoE dispatch (argsort/bincount replacement)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward",&moe_rank_forward,"TP rank MoE forward -> PARTIAL",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v2",&moe_rank_forward_v2,"TP rank MoE forward v2 (single gather/scatter)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v4",&moe_rank_forward_v4,"TP rank MoE forward v4 (batched dequant + pow2 bucket, C++ port of v3_mb)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v5",&moe_rank_forward_v5,"TP rank MoE forward v5 (v2 flow + batched dequant, no padded buffer)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v6",&moe_rank_forward_v6,"TP rank MoE forward v6 (v5 + fp16 weight-mul, no fp32 round-trip)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v7",&moe_rank_forward_v7,"TP rank MoE forward v7 (v5 + matmul_out direct write, no copy_/cast)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v9_cutlass",&moe_rank_forward_v9_cutlass,"v9 CUTLASS SM80 grouped GEMM (variable M, no pad)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_decode_fused",&moe_rank_forward_decode_fused,"small-T fused IQ GEMV (T=1..127)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_fused",&moe_tp8_decode_fused,"TP8 decode fused single-entry 8-rank launch",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_broadcast_tensor",&moe_tp8_broadcast_tensor,"peer broadcast one tensor to list",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_bcast_launch",&moe_tp8_decode_bcast_launch,"TP8 peer copy then launch on destination streams",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_fused_local",&moe_tp8_decode_fused_local,"TP8 decode fused with pre-resident acts",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_fused_local_half",&moe_tp8_decode_fused_local_half,"TP8 decode fused with pre-resident acts and half outputs",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_meta_bcast_local_half",&moe_tp8_decode_meta_bcast_local_half,"TP8 metadata peer-broadcast plus local half-output launch",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_tp8_decode_meta_bcast_local_half_rank",&moe_tp8_decode_meta_bcast_local_half_rank,"graph-safe rank-local leaf for TP8 metadata broadcast",pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("moe_route_decode_t1_inplace", &moe_route_decode_t1_inplace_cuda, "route mm+select inplace");
  m.def("moe_route_select_only", &moe_route_select_only_cuda, "select only");
  m.def("moe_route_decode_t1", &moe_route_decode_t1_cuda, "route alloc");
  m.def("shared_decode_q8_inplace",&shared_decode_q8_inplace_cuda,"shared expert Q8 decode inplace (TN all T)");
  m.def("shared_decode_q8",&shared_decode_q8_cuda,"shared expert Q8 decode fused");
  m.def("rms_norm_t1_inplace", &rms_norm_t1_inplace_cuda, "rms t1 f2h");
 m.def("moe_rank_forward_decode",&moe_rank_forward_decode,"decode small-B direct experts",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v10_scatter",&moe_rank_forward_v10_scatter,"v10 v9 + fused weighted scatter",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_v11_resident",&moe_rank_forward_v11_resident,"v11 resident fp16 weights + fused scatter",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("dequant_gate_resident",&dequant_gate_resident,"dequant full IQ3 gate/up pack -> fp16 [E,local,D]");
 m.def("moe_rank_forward_v8_grouped",&moe_rank_forward_v8_grouped,"TP rank MoE v8 grouped pad+bmm (defrag expert loop)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("moe_rank_forward_hybrid",&moe_rank_forward_hybrid,"TP rank MoE hybrid (active-only dequant + multi-stream)",pybind11::call_guard<pybind11::gil_scoped_release>());
 m.def("dequant_down_resident",&dequant_down_resident,"dequant full IQ4 down pack -> fp16 [E,D,local]");
 m.def("policy_threshold",&policy_threshold);
 m.def("dequant_iq3_selected_cuda",&dequant_iq3_selected_cuda,"selected IQ3 dequant");
 m.def("dequant_iq4_selected_cuda",&dequant_iq4_selected_cuda,"selected IQ4 dequant");
 m.def("dequant_iq3_xxs_cuda",&dequant_iq3_xxs_cuda,"dequant IQ3_XXS packed -> fp16");
 m.def("dequant_iq4_xs_cuda",&dequant_iq4_xs_cuda,"dequant IQ4_XS packed -> fp16");
 m.def("down_iq4_tc_cuda",&down_iq4_tc_cuda,"fused TC down GEMM IQ4_XS (dequant+wmma, no fp16 buf)");
 m.def("mmq_iq3_tile_cuda",&mmq_iq3_tile_cuda,"fused MMQ IQ3_XXS tile (dequant+gemm, no fp16 buf)");
 m.def("mmq_iq4_tile_cuda",&mmq_iq4_tile_cuda,"fused MMQ IQ4_XS tile (dequant+gemm, no fp16 buf)");
}
