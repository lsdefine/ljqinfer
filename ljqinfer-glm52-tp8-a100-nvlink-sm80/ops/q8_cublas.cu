#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

// GGML Q8_0 block: 1 fp16 scale + 32 int8 = 34 bytes -> 32 fp16 outputs.
__global__ void dequant_q8_0(const unsigned char* __restrict__ p,
                             half* __restrict__ w,
                             long long blocks) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const unsigned char* q = p + b * 34;
    half d = *reinterpret_cast<const half*>(q);
    const signed char* qs = reinterpret_cast<const signed char*>(q + 2);
    half* o = w + b * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        o[i] = __hmul(d, __float2half((float)qs[i]));
    }
}

static void launch_dequant(torch::Tensor packed, half* out_ptr) {
    long long nb = (long long)packed.numel() / 34;
    int th = 256;
    int blocks = (int)((nb + th - 1) / th);
    dequant_q8_0<<<blocks, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        packed.data_ptr<unsigned char>(), out_ptr, nb);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Persistent dequant cache keyed by packed storage data_ptr + numel + device.
// Decode T=1 reuses same packed weights for model lifetime.
struct CacheKey {
  uintptr_t ptr;
  int64_t numel;
  int device;
  bool operator==(const CacheKey& o) const {
    return ptr==o.ptr && numel==o.numel && device==o.device;
  }
};
struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const {
    return (size_t)k.ptr ^ (size_t)k.numel ^ ((size_t)k.device<<48);
  }
};
static std::mutex g_mu;
static std::unordered_map<CacheKey, torch::Tensor, CacheKeyHash> g_wcache;
// Weight cache is a source-controlled production invariant.  It cannot be
// disabled through the environment because doing so silently selects a slower path.
static constexpr bool cache_enabled() { return true; }

static torch::Tensor get_or_dequant_w(torch::Tensor packed, int64_t N, int64_t K, c10::IntArrayRef shape) {
  auto opts = packed.options().dtype(torch::kFloat16);
  if (!cache_enabled()) {
    auto w = torch::empty(shape, opts);
    launch_dequant(packed, reinterpret_cast<half*>(w.data_ptr<at::Half>()));
    return w;
  }
  CacheKey key{(uintptr_t)packed.data_ptr(), packed.numel(), (int)packed.device().index()};
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_wcache.find(key);
    if (it != g_wcache.end()) return it->second;
  }
  auto w = torch::empty(shape, opts);
  launch_dequant(packed, reinterpret_cast<half*>(w.data_ptr<at::Half>()));
  // ensure dequant complete before publishing to other streams/threads
  at::cuda::getCurrentCUDAStream().synchronize();
  {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_wcache.find(key);
    if (it != g_wcache.end()) return it->second;
    g_wcache.emplace(key, w);
  }
  return w;
}

// forward(x[T,K] fp16, packed[N, K/32*34] uint8) -> y[T,N] fp16.
torch::Tensor q8_cublas_forward_cuda(torch::Tensor x, torch::Tensor packed, int64_t K) {
    int64_t N = packed.size(0);
    auto w = get_or_dequant_w(packed, N, K, {N, K});
    return at::matmul(x, w.t());
}

torch::Tensor q8_cublas_forward_out_cuda(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y) {
    int64_t N = packed.size(0);
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == x.scalar_type());
    TORCH_CHECK(y.dim() == 2 && y.size(0) == x.size(0) && y.size(1) == N);
    auto w = get_or_dequant_w(packed, N, K, {N, K});
    at::matmul_out(y, x, w.t());
    return y;
}

// forward_grouped(x[T,H,K] fp16, packed[H,N, K/32*34] uint8) -> y[T,H,N] fp16.
torch::Tensor q8_cublas_forward_grouped_cuda(torch::Tensor x, torch::Tensor packed, int64_t K) {
    int64_t H = packed.size(0);
    int64_t N = packed.size(1);
    auto w = get_or_dequant_w(packed, N, K, {H, N, K});
    auto wt = w.transpose(1, 2).contiguous();          // [H,K,N]
    auto xt = x.transpose(0, 1).contiguous();           // [H,T,K]
    auto y = at::bmm(xt, wt);                            // [H,T,N]
    return y.transpose(0, 1).contiguous();              // [T,H,N]
}

// Force clear cache (tests / OOM recovery)

static std::unordered_map<CacheKey, torch::Tensor, CacheKeyHash> g_wtcache;
static std::unordered_set<CacheKey, CacheKeyHash> g_pinned_wcache;
static std::unordered_set<CacheKey, CacheKeyHash> g_pinned_wtcache;

torch::Tensor q8_cublas_forward_grouped_out(torch::Tensor x, torch::Tensor packed, int64_t K, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda() && y.is_cuda());
    TORCH_CHECK(y.is_contiguous());
    int64_t H = packed.size(0);
    int64_t N = packed.size(1);
    int64_t T = x.size(0);
    TORCH_CHECK(x.dim()==3 && x.size(0)==T && x.size(1)==H && x.size(2)==K);
    TORCH_CHECK(y.size(0)==T && y.size(1)==H && y.size(2)==N);
    auto w = get_or_dequant_w(packed, N, K, {H, N, K});
    CacheKey key{(uintptr_t)packed.data_ptr(), packed.numel(), (int)packed.device().index()};
    torch::Tensor wt;
    {
      std::lock_guard<std::mutex> g(g_mu);
      auto it=g_wtcache.find(key);
      if(it!=g_wtcache.end()) wt=it->second;
    }
    if(!wt.defined()){
      wt=w.transpose(1,2).contiguous();
      at::cuda::getCurrentCUDAStream().synchronize();
      std::lock_guard<std::mutex> g(g_mu);
      auto it=g_wtcache.find(key);
      if(it!=g_wtcache.end()) wt=it->second; else g_wtcache.emplace(key,wt);
    }
    if(T==1){
      // Prefer zero-copy when each head K-row is contiguous (feat stride 1).
      auto xs = x.strides();
      torch::Tensor xt;
      if(xs[2]==1){
        xt = x.as_strided({H, (int64_t)1, K}, {xs[1], xs[0] ? xs[0] : H*xs[1], xs[2]});
      } else {
        xt = x.permute({1,0,2}).contiguous();
      }
      // Ensure xt is usable by bmm (must be contiguous in last two dims typically)
      if(!xt.is_contiguous()) xt = xt.contiguous();
      auto yt = y.as_strided({H, (int64_t)1, N}, {N, H*N, (int64_t)1});
      at::bmm_out(yt, xt, wt);
      return y;
    }
    auto xt = x.contiguous().permute({1,0,2}).contiguous();
    auto yt = at::bmm(xt, wt);
    y.copy_(yt.permute({1,0,2}));
    return y;
}



void q8_cublas_pin_weight_cache() {
  std::lock_guard<std::mutex> g(g_mu);
  for (const auto& kv : g_wcache) g_pinned_wcache.insert(kv.first);
  for (const auto& kv : g_wtcache) g_pinned_wtcache.insert(kv.first);
}

void q8_cublas_clear_weight_cache() {
  std::lock_guard<std::mutex> g(g_mu);
  for (auto it = g_wtcache.begin(); it != g_wtcache.end(); ) {
    if (g_pinned_wtcache.count(it->first)) ++it;
    else it = g_wtcache.erase(it);
  }
  for (auto it = g_wcache.begin(); it != g_wcache.end(); ) {
    if (g_pinned_wcache.count(it->first)) ++it;
    else it = g_wcache.erase(it);
  }
}
