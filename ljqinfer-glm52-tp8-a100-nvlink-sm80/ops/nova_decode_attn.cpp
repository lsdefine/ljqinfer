#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <mutex>

torch::Tensor q8_mmvq_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y);
void q8_mmvq_dual_forward_out(torch::Tensor x, torch::Tensor p0, torch::Tensor p1, int64_t K, torch::Tensor y0, torch::Tensor y1);
void q8_mmvq_grouped_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y);
torch::Tensor q8_mmvq_rms_forward_out(torch::Tensor x, torch::Tensor norm_w, torch::Tensor packed, int64_t K, torch::Tensor y, double eps);
void q8_mmvq_batch_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y, int64_t B, int64_t Q);
void q8_mmvq_batch_dual_forward_out(torch::Tensor x, torch::Tensor p0, torch::Tensor p1, int64_t K, torch::Tensor y0, torch::Tensor y1, int64_t B, int64_t Q);
void q8_mmvq_batch_grouped_forward_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y, int64_t B, int64_t Q);
void q8_mmvq_batch_rms_forward_out(torch::Tensor x, torch::Tensor norm_w, torch::Tensor packed, int64_t K, torch::Tensor y, double eps, int64_t B, int64_t Q);
torch::Tensor q8_cublas_forward_out_cuda(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y);
torch::Tensor rms_norm_half_out(torch::Tensor x, torch::Tensor w, torch::Tensor y);
torch::Tensor rope_half_out(torch::Tensor x, torch::Tensor positions, torch::Tensor y);
torch::Tensor rope_half_strided_out(torch::Tensor x, torch::Tensor positions, torch::Tensor y);
torch::Tensor kv_post_cache_fused_out(torch::Tensor kv, torch::Tensor w, torch::Tensor positions, torch::Tensor pool, torch::Tensor page_table);
torch::Tensor kv_post_cache_fused_batch_out(torch::Tensor kv, torch::Tensor w, torch::Tensor positions, torch::Tensor pool, torch::Tensor page_tables, int64_t Q);
// DECODE-ONLY production primitive (T=1..6, including batched decode).  Do not
// route model prefill through flash_mla_sm80*: prefill must use tc_mla for an
// identity page table or the true paged_prefill_mla kernel otherwise.
torch::Tensor native_h8_flash_mla_sm80_out_k0(torch::Tensor q_latent, torch::Tensor q_rope, torch::Tensor pool, torch::Tensor page_table, torch::Tensor k0, torch::Tensor out);
torch::Tensor native_h8_flash_mla_sm80_out_k0_batch(torch::Tensor q_latent, torch::Tensor q_rope, torch::Tensor pool, torch::Tensor pt_pack, torch::Tensor k0, torch::Tensor out);

struct AttnDecodeWs {
  static constexpr int64_t Tmax = 32;
  torch::Tensor xn, qa, qb, q_rope, q_latent, kv, out_latent, heads, partial;
};

// Invocation-owned scratch is a structural requirement, not an optimization.
// A process-global per-device workspace lets independently captured B1, B2 and
// MTP graphs bind the same addresses.  Their safety would then depend on an
// unenforced single-stream/no-overlap convention.  Allocations made while a
// CUDA graph is captured belong to that graph's private pool; eager calls keep
// their tensors alive through the returned partial view.
static AttnDecodeWs make_attn_ws(const torch::Tensor& x, int64_t T){
  TORCH_CHECK(T>=1 && T<=AttnDecodeWs::Tmax, "decode attention supports T=1..32");
  auto o=x.options();
  return {
    torch::empty({T,6144},o), torch::empty({T,2048},o),
    torch::empty({T,8,256},o), torch::empty({T,8,64},o),
    torch::empty({T,8,512},o), torch::empty({T,576},o),
    torch::empty({T,8,512},o), torch::empty({T,8,256},o),
    torch::empty({T,6144},o)
  };
}

// Unified production attention module: decode + prefill orchestration.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <thread>
#include <future>

torch::Tensor q8_cublas_forward_cuda(torch::Tensor x, torch::Tensor packed, int64_t K);
torch::Tensor q8_cublas_forward_grouped_cuda(torch::Tensor x, torch::Tensor packed, int64_t K);
torch::Tensor flash_mla_sm80(torch::Tensor q_latent, torch::Tensor q_rope, torch::Tensor cache, int64_t q_start);
torch::Tensor paged_prefill_mla(torch::Tensor q_latent, torch::Tensor q_rope, torch::Tensor pool, torch::Tensor page_table, int64_t q_start);

static torch::Tensor rms(torch::Tensor x, torch::Tensor w){
  auto xf=x.to(torch::kFloat32);
  return (xf*torch::rsqrt(xf.square().mean(-1,true)+1e-5)*w.to(torch::kFloat32)).to(x.scalar_type());
}
static torch::Tensor rope(torch::Tensor x,torch::Tensor p){
  auto opt=x.options().dtype(torch::kFloat32);
  auto pair=torch::arange(32,opt);
  auto th=p.to(torch::kFloat32).unsqueeze(1)*torch::pow(8000000.0,-2.0*pair.unsqueeze(0)/64.0);
  auto cs=th.cos().unsqueeze(1),sn=th.sin().unsqueeze(1);
  auto z=x.to(torch::kFloat32).reshape({x.size(0),x.size(1),32,2});
  auto e=z.select(-1,0),o=z.select(-1,1);
  return torch::stack({e*cs-o*sn,e*sn+o*cs},-1).flatten(-2).to(x.scalar_type());
}

// Small square causal mask for present tokens only: key_local > query_local
static torch::Tensor causal_mask_qq(const torch::Device& dev, int64_t Q){
  static std::mutex mu; static std::unordered_map<std::string,torch::Tensor> masks;
  std::string key=std::to_string(dev.index())+":qq:"+std::to_string(Q);
  { std::lock_guard<std::mutex> g(mu); auto it=masks.find(key); if(it!=masks.end()) return it->second; }
  auto m=torch::ones({Q,Q}, torch::TensorOptions().dtype(torch::kBool).device(dev)).triu(1);
  { std::lock_guard<std::mutex> g(mu); masks.emplace(key,m); }
  return m;
}

// Fast TC MLA for one causal query block. cache contains exactly the old
// prefix plus keys through this block, so only the final QxQ slice is masked.
static torch::Tensor tc_mla_block(torch::Tensor ql,torch::Tensor qr,torch::Tensor cache,int64_t block_start){
  const int64_t Q=ql.size(0), K=cache.size(0);
  TORCH_CHECK(block_start>=0 && block_start+Q==K, "invalid causal block range");
  auto latent=cache.slice(-1,0,512);
  auto q=torch::cat({ql.transpose(0,1),qr.transpose(0,1)},-1);
  q.mul_(0.0625);
  auto score=torch::matmul(q,cache.transpose(0,1)); // [H,Q,K]
  score.slice(/*dim=*/2,block_start,K).masked_fill_(
      causal_mask_qq(score.device(),Q).unsqueeze(0),
      -std::numeric_limits<float>::infinity());
  at::softmax_out(score,score,-1);
  return torch::matmul(score,latent).transpose(0,1).contiguous();
}

// Keep one request and one model-layer call, but bound Attention's quadratic
// workspace. Block b sees the full old prefix and all keys through b, exactly
// matching the original causal QxK operation. Small prefill remains unchanged.
static torch::Tensor tc_mla(torch::Tensor ql,torch::Tensor qr,torch::Tensor cache,int64_t q_start){
  const int64_t Q=ql.size(0), K=cache.size(0);
  TORCH_CHECK(q_start>=0 && q_start+Q==K, "invalid prefill range");
  constexpr int64_t BLOCK_Q=2048;
  if(Q<=BLOCK_Q) return tc_mla_block(ql,qr,cache,q_start);
  std::vector<torch::Tensor> out;
  out.reserve((Q+BLOCK_Q-1)/BLOCK_Q);
  for(int64_t q0=0;q0<Q;q0+=BLOCK_Q){
    const int64_t n=std::min<int64_t>(BLOCK_Q,Q-q0);
    out.push_back(tc_mla_block(
        ql.narrow(0,q0,n),qr.narrow(0,q0,n),
        cache.narrow(0,0,q_start+q0+n),q_start+q0));
  }
  return torch::cat(out,0);
}

// LEGACY CONTIGUOUS ABI / BENCHMARK PATH ONLY; this is not model prefill.
// Do not call append_mla or forward_rank_cached*_tc from the production prefill
// path.  Their flash_mla_sm80 branch is retained only for old direct ABI tests.
// Production prefill is exclusively forward_rank_paged_inplace_tc below: it
// dispatches identity pages to tc_mla and non-identity pages to paged_prefill_mla.
// flash_mla_sm80 remains production code only for decode T=1..6.
// This legacy ABI has one source-controlled choice: flash MLA when compatible,
// otherwise the type/shape fallback to tc_mla.  No environment selector.
static torch::Tensor append_mla(torch::Tensor ql,torch::Tensor qr,torch::Tensor cache,int64_t q_start){
  if(ql.scalar_type()==at::kHalf && qr.scalar_type()==at::kHalf
     && cache.scalar_type()==at::kHalf && cache.is_contiguous()
     && q_start>=0 && q_start<cache.size(0) && q_start+ql.size(0)<=cache.size(0)
     && ql.size(1)<=64){
    return flash_mla_sm80(ql.contiguous(),qr.contiguous(),cache,q_start);
  }
  return tc_mla(ql,qr,cache,q_start);
}

static std::vector<torch::Tensor> forward_rank_impl(torch::Tensor x,torch::Tensor positions,torch::Tensor attn_norm,torch::Tensor q_a,torch::Tensor q_a_norm,torch::Tensor q_b,torch::Tensor kv_a,torch::Tensor kv_a_norm,torch::Tensor k_b,torch::Tensor v_b,torch::Tensor attn_out,bool tc){
  TORCH_CHECK(x.is_cuda()&&positions.is_cuda(),"x/positions must be CUDA");c10::cuda::CUDAGuard guard(x.device());const int64_t T=x.size(0);
  auto xn=rms(x,attn_norm);auto qa=q8_cublas_forward_cuda(xn.contiguous(),q_a,6144);qa=rms(qa,q_a_norm);
  auto qb=q8_cublas_forward_cuda(qa.contiguous(),q_b,2048).view({T,8,256});auto q_nope=qb.slice(-1,0,192);auto q_rope=rope(qb.slice(-1,192,256),positions);auto q_latent=q8_cublas_forward_grouped_cuda(q_nope.contiguous(),k_b,192);
  auto kv=q8_cublas_forward_cuda(xn.contiguous(),kv_a,6144);auto latent=rms(kv.slice(-1,0,512),kv_a_norm);auto k_rope=rope(kv.slice(-1,512,576).unsqueeze(1),positions).select(1,0);auto cache=torch::cat({latent,k_rope},-1);
  auto out_latent=tc?tc_mla(q_latent.contiguous(),q_rope.contiguous(),cache.contiguous(),0):flash_mla_sm80(q_latent.contiguous(),q_rope.contiguous(),cache.contiguous(),0);
  auto heads=q8_cublas_forward_grouped_cuda(out_latent.contiguous(),v_b,512);auto partial=q8_cublas_forward_cuda(heads.reshape({T,-1}).contiguous(),attn_out,2048);return {cache,partial};
}
#define ARGS torch::Tensor x,torch::Tensor positions,torch::Tensor attn_norm,torch::Tensor q_a,torch::Tensor q_a_norm,torch::Tensor q_b,torch::Tensor kv_a,torch::Tensor kv_a_norm,torch::Tensor k_b,torch::Tensor v_b,torch::Tensor attn_out

// First paged ABI: the physical owner is [page, page_size, 576].  During the
// identity phase page_table is created once as [0,1,...], so flattening the pool
// is exactly the former contiguous cache with no gather/copy.  Keeping this ABI
// separate makes the current limitation explicit; shuffled-page addressing will
// replace only this adapter, not the model/cache ownership contract.

// Identity-page fast path.  When page_table is exactly 0..n-1 the logical
// prefix maps 1:1 onto the physical pool, so pool.view({-1,576}) IS the
// contiguous cache tc_mla wants -- no gather, no copy.  Measured at H=8:
// tc_mla beats paged_prefill_mla by 1.38-1.47x on 32k-64k prefixes.
// It is not free: tc_mla materializes score [H,BLOCK_Q,K] in fp32, so take it
// only when that workspace comfortably fits.  Falling back to paged is always
// safe, so every uncertain case rejects.
// NOTE: never route this through append_mla -- its default flash_mla_sm80 path
// measured 12.7x SLOWER here and allocates O(Q*K) (24GiB at 32k+32k).
static bool paged_tc_fastpath_ok(const torch::Tensor& page_table,int64_t K0,int64_t T,int64_t H){
  // Fresh identity prefill benefits from tc_mla. Append uses paged_prefill_mla:
  // its causal block skip wins end-to-end even though a bare tc_mla call is faster.
  if(T<=0 || K0>0) return false;
  const int64_t Ktot=K0+T;
  const int64_t block=std::min<int64_t>(T,2048);
  const size_t need=(size_t)H*(size_t)block*(size_t)Ktot*4u;
  size_t freeB=0,totalB=0;
  if(cudaMemGetInfo(&freeB,&totalB)!=cudaSuccess) return false;
  // torch's cached-but-unallocated blocks are reusable, count them as free.
  {
    auto st=c10::cuda::CUDACachingAllocator::getDeviceStats(page_table.device().index());
    const size_t idx=static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    const int64_t slack=st.reserved_bytes[idx].current-st.allocated_bytes[idx].current;
    if(slack>0) freeB+=(size_t)slack;
  }
  // 2x for the transient copies around the score tensor, plus a hard reserve.
  const bool mem_ok = !(need*2u+(size_t)768*1024u*1024u > freeB);
  if(!mem_ok) return false;
  auto host=page_table.to(torch::kCPU);
  const int64_t* p=host.data_ptr<int64_t>();
  for(int64_t i=0;i<host.numel();++i) if(p[i]!=i) return false;
  return true;
}

// General single-sequence paged prefill. page_table[logical_page] gives the
// physical page in cache_pool, and new KV is scattered directly to those pages.
// Fresh identity layout may view the pool directly for tc_mla. Append and every
// non-identity layout are consumed in place by paged_prefill_mla. There is no
// gather-to-flash fallback. Never route prefill through flash_mla_sm80*.
// Preserve this invariant for both single-sequence and existing batched prefill.

// CUDA-graph-safe B-agnostic decode.  Dense projections execute once over
// flattened [B*Q,D]; only paged KV scatter and ragged MLA remain per sequence.
// This is the native Bn operator below the permanent Python ABI.
std::vector<torch::Tensor> native_h8_forward_rank_paged_batch_k0(
    torch::Tensor x, torch::Tensor positions, torch::Tensor cache_pool,
    std::vector<torch::Tensor> page_tables, std::vector<torch::Tensor> K0s,
    torch::Tensor attn_norm, torch::Tensor q_a, torch::Tensor q_a_norm,
    torch::Tensor q_b, torch::Tensor kv_a, torch::Tensor kv_a_norm,
    torch::Tensor k_b, torch::Tensor v_b, torch::Tensor attn_out) {
  TORCH_CHECK(x.is_cuda() && positions.is_cuda() && cache_pool.is_cuda(),
              "x/positions/pool must be CUDA");
  TORCH_CHECK(x.dim()==2 && x.size(1)==6144 && x.is_contiguous(),
              "batch x must be contiguous [B*Q,6144]");
  const int64_t B=page_tables.size(), T=x.size(0);
  TORCH_CHECK(B>=1 && B<=16 && (int64_t)K0s.size()==B && T%B==0,
              "native attention supports B=1..16 with equal Q");
  const int64_t Q=T/B;
  TORCH_CHECK(Q>=1 && Q<=6 &&
              T<=AttnDecodeWs::Tmax && positions.dim()==1 &&
              positions.numel()==T && positions.is_contiguous(),
              "native attention supports BnQ1..6");
  TORCH_CHECK(cache_pool.dim()==3 && cache_pool.size(2)==576 && cache_pool.is_contiguous(),
              "cache pool must be contiguous [pages,page_size,576]");
  c10::cuda::CUDAGuard guard(x.device());
  auto w=make_attn_ws(x,T);
  auto& xn=w.xn; auto& qa=w.qa; auto& qb=w.qb;
  auto& q_latent=w.q_latent; auto& q_rope=w.q_rope; auto& kv=w.kv;
  auto& out_latent=w.out_latent; auto& heads=w.heads; auto& partial=w.partial;

  // Shared dense portion: one launch per projection for all B*Q rows.
  rms_norm_half_out(x,attn_norm,xn);
  q8_mmvq_batch_dual_forward_out(xn,q_a,kv_a,6144,qa,kv,B,Q);
  q8_mmvq_batch_rms_forward_out(qa,q_a_norm,q_b,2048,qb.reshape({T,2048}),1e-5,B,Q);
  auto q_nope=qb.narrow(-1,0,192);
  auto q_rope_src=qb.narrow(-1,192,64);
  rope_half_strided_out(q_rope_src,positions,q_rope);
  q8_mmvq_batch_grouped_forward_out(q_nope,k_b,192,q_latent,B,Q);

  // Ragged portion: each sequence owns its page table and device K0 scalar.
  const bool use_batch_mla = (B > 1 && Q > 1);
  int64_t Pmax = 0;
  if (use_batch_mla) for (auto& t : page_tables) Pmax = std::max<int64_t>(Pmax, t.numel());
  auto pt_pack = use_batch_mla
      ? torch::zeros({B, Pmax}, torch::TensorOptions().dtype(torch::kInt64).device(cache_pool.device()))
      : torch::Tensor();
  auto k0_pack = use_batch_mla
      ? torch::empty({B}, torch::TensorOptions().dtype(torch::kInt32).device(cache_pool.device()))
      : torch::Tensor();
  for(int64_t i=0;i<B;++i){
    auto table=page_tables[i]; auto k0=K0s[i];
    TORCH_CHECK(table.is_cuda() && table.scalar_type()==torch::kInt64 &&
                table.dim()==1 && table.is_contiguous(), "invalid page table");
    TORCH_CHECK(k0.is_cuda() && k0.scalar_type()==torch::kInt32 &&
                k0.numel()==1 && k0.is_contiguous(), "K0 must be CUDA int32[1]");
    TORCH_CHECK(table.device()==cache_pool.device() && k0.device()==cache_pool.device(),
                "page table/K0/pool device mismatch");
    const int64_t start=Q*i;
    if (use_batch_mla) {
      pt_pack.narrow(0,i,1).view({-1}).narrow(0,0,table.numel()).copy_(table, true);
      k0_pack.narrow(0,i,1).copy_(k0, true);
      continue;
    }
    kv_post_cache_fused_out(kv.narrow(0,start,Q),kv_a_norm,
                            positions.narrow(0,start,Q),cache_pool,table);
    auto out_i=out_latent.narrow(0,start,Q);
    native_h8_flash_mla_sm80_out_k0(
        q_latent.narrow(0,start,Q),q_rope.narrow(0,start,Q),cache_pool,table,k0,
        out_i);
  }
  if (use_batch_mla) {
    kv_post_cache_fused_batch_out(kv,kv_a_norm,positions,cache_pool,pt_pack,Q);
    native_h8_flash_mla_sm80_out_k0_batch(
        q_latent.view({B,Q,8,512}), q_rope.view({B,Q,8,64}), cache_pool,
        pt_pack, k0_pack, out_latent.view({B,Q,8,512}));
  }
  q8_mmvq_batch_grouped_forward_out(out_latent,v_b,512,heads,B,Q);
  // Q=6 is the production MTP width. Persistent dequantization is populated
  // during graph warm-up; one GEMM covers all B*Q rows.
  if (Q == 6)
    q8_cublas_forward_out_cuda(heads.reshape({T,2048}),attn_out,2048,partial);
  else
    q8_mmvq_batch_forward_out(heads.reshape({T,2048}),attn_out,2048,partial,B,Q);
  return {cache_pool,partial};
}


std::vector<torch::Tensor> native_h8_forward_rank_paged_batch_k0_projected(
    torch::Tensor positions, torch::Tensor cache_pool,
    std::vector<torch::Tensor> page_tables, std::vector<torch::Tensor> K0s,
    torch::Tensor qa, torch::Tensor q_a_norm, torch::Tensor q_b,
    torch::Tensor kv, torch::Tensor kv_a_norm, torch::Tensor k_b,
    torch::Tensor v_b, torch::Tensor attn_out) {
  TORCH_CHECK(qa.is_cuda() && kv.is_cuda() && positions.is_cuda() && cache_pool.is_cuda(),
              "projected latents/positions/pool must be CUDA");
  TORCH_CHECK(qa.dim()==2 && qa.size(1)==2048 && qa.is_contiguous(),
              "qa must be contiguous [B*Q,2048]");
  TORCH_CHECK(kv.dim()==2 && kv.size(1)==576 && kv.is_contiguous() &&
              kv.size(0)==qa.size(0) && kv.device()==qa.device(),
              "kv must be contiguous [B*Q,576] on qa device");
  const int64_t B=page_tables.size(), T=qa.size(0);
  TORCH_CHECK(B>=1 && B<=16 && (int64_t)K0s.size()==B && T%B==0,
              "projected attention supports B=1..16 with equal Q");
  const int64_t Q=T/B;
  TORCH_CHECK(Q>=1 && Q<=6 &&
              T<=AttnDecodeWs::Tmax && positions.dim()==1 &&
              positions.numel()==T && positions.is_contiguous(),
              "projected attention supports BnQ1..6");
  TORCH_CHECK(cache_pool.dim()==3 && cache_pool.size(2)==576 && cache_pool.is_contiguous(),
              "cache pool must be contiguous [pages,page_size,576]");
  c10::cuda::CUDAGuard guard(qa.device());
  auto w=make_attn_ws(qa,T);
  auto& qb=w.qb; auto& q_latent=w.q_latent; auto& q_rope=w.q_rope;
  auto& out_latent=w.out_latent; auto& heads=w.heads; auto& partial=w.partial;
  // Shared dense portion: one launch per projection for all B*Q rows.
  q8_mmvq_batch_rms_forward_out(qa,q_a_norm,q_b,2048,qb.reshape({T,2048}),1e-5,B,Q);
  auto q_nope=qb.narrow(-1,0,192);
  auto q_rope_src=qb.narrow(-1,192,64);
  rope_half_strided_out(q_rope_src,positions,q_rope);
  q8_mmvq_batch_grouped_forward_out(q_nope,k_b,192,q_latent,B,Q);

  // Ragged portion: each sequence owns its page table and device K0 scalar.
  const bool use_batch_mla = (B > 1 && Q > 1);
  int64_t Pmax = 0;
  if (use_batch_mla) for (auto& t : page_tables) Pmax = std::max<int64_t>(Pmax, t.numel());
  auto pt_pack = use_batch_mla
      ? torch::zeros({B, Pmax}, torch::TensorOptions().dtype(torch::kInt64).device(cache_pool.device()))
      : torch::Tensor();
  auto k0_pack = use_batch_mla
      ? torch::empty({B}, torch::TensorOptions().dtype(torch::kInt32).device(cache_pool.device()))
      : torch::Tensor();
  for(int64_t i=0;i<B;++i){
    auto table=page_tables[i]; auto k0=K0s[i];
    TORCH_CHECK(table.is_cuda() && table.scalar_type()==torch::kInt64 &&
                table.dim()==1 && table.is_contiguous(), "invalid page table");
    TORCH_CHECK(k0.is_cuda() && k0.scalar_type()==torch::kInt32 &&
                k0.numel()==1 && k0.is_contiguous(), "K0 must be CUDA int32[1]");
    TORCH_CHECK(table.device()==cache_pool.device() && k0.device()==cache_pool.device(),
                "page table/K0/pool device mismatch");
    const int64_t start=Q*i;
    if (use_batch_mla) {
      pt_pack.narrow(0,i,1).view({-1}).narrow(0,0,table.numel()).copy_(table, true);
      k0_pack.narrow(0,i,1).copy_(k0, true);
      continue;
    }
    kv_post_cache_fused_out(kv.narrow(0,start,Q),kv_a_norm,
                            positions.narrow(0,start,Q),cache_pool,table);
    auto out_i=out_latent.narrow(0,start,Q);
    native_h8_flash_mla_sm80_out_k0(
        q_latent.narrow(0,start,Q),q_rope.narrow(0,start,Q),cache_pool,table,k0,
        out_i);
  }
  if (use_batch_mla) {
    kv_post_cache_fused_batch_out(kv,kv_a_norm,positions,cache_pool,pt_pack,Q);
    native_h8_flash_mla_sm80_out_k0_batch(
        q_latent.view({B,Q,8,512}), q_rope.view({B,Q,8,64}), cache_pool,
        pt_pack, k0_pack, out_latent.view({B,Q,8,512}));
  }
  q8_mmvq_batch_grouped_forward_out(out_latent,v_b,512,heads,B,Q);
  // Q=6 is the production MTP width. Persistent dequantization is populated
  // during graph warm-up; one GEMM covers all B*Q rows.
  if (Q == 6)
    q8_cublas_forward_out_cuda(heads.reshape({T,2048}),attn_out,2048,partial);
  else
    q8_mmvq_batch_forward_out(heads.reshape({T,2048}),attn_out,2048,partial,B,Q);
  return {cache_pool,partial};
}


torch::Tensor native_rms_norm_half_out(
    torch::Tensor x, torch::Tensor weight, torch::Tensor out) {
  return rms_norm_half_out(x, weight, out);
}

// Concurrent 8-rank decode entry: launch each rank on its own host thread (GIL already released by pybind).
// Returns list[8] partials only; cache is written inplace into storages.


// Stage timing for decode rank (CUDA events). Returns {cache, partial}.


// --- A/B probes: expose the two append attention paths directly (test only) ---

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("flash_mla_out_k0_dbg",&native_h8_flash_mla_sm80_out_k0,"debug direct flash mla");
  m.def("forward_rank_paged_batch_k0",&native_h8_forward_rank_paged_batch_k0,"graph-safe paged Bn decode rank",pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("forward_rank_paged_batch_k0_projected",&native_h8_forward_rank_paged_batch_k0_projected,"private graph path with preprojected q/kv latents",pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("rms_norm_half_out",&native_rms_norm_half_out,"private graph-safe RMSNorm output",pybind11::call_guard<pybind11::gil_scoped_release>());
}
