#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// Publishing the accepted rows of a whole batch is one launch: the slot, the
// start row and the accepted count of every request travel as kernel
// arguments, so nothing has to be staged to the device first.  Staging them
// through a tensor would cost a pageable H2D copy, and that copy blocks the
// host until the queued decode work has drained.
#define LJQ_MAX_SLOTS 8

struct CommitPlan {
  int64_t slot[LJQ_MAX_SLOTS];
  int64_t start[LJQ_MAX_SLOTS];
  int64_t count[LJQ_MAX_SLOTS];
};

template <typename scalar_t>
__global__ void paged_commit_kernel(scalar_t *__restrict__ pool_a,
                                    scalar_t *__restrict__ pool_b,
                                    const int64_t *__restrict__ table,
                                    const scalar_t *__restrict__ src_a,
                                    const scalar_t *__restrict__ src_b,
                                    const CommitPlan plan, int rows_per_page,
                                    int dim_a, int dim_b, int per,
                                    int table_stride) {
  const int b = blockIdx.y;
  const int i = blockIdx.x;
  if (i >= plan.count[b]) return;
  const int64_t pos = plan.start[b] + i;
  const int64_t page =
      table[plan.slot[b] * (int64_t)table_stride + pos / rows_per_page];
  if (page < 0) return;  // an unmapped page means the window never claimed it
  const int64_t dst = page * rows_per_page + pos % rows_per_page;
  const int64_t src = (int64_t)b * per + i;
  const bool second = blockIdx.z == 1;
  const int dim = second ? dim_b : dim_a;
  scalar_t *d = (second ? pool_b : pool_a) + dst * dim;
  const scalar_t *s = (second ? src_b : src_a) + src * dim;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) d[c] = s[c];
}

void paged_commit_pair(torch::Tensor pool_a, torch::Tensor pool_b,
                       torch::Tensor table, torch::Tensor src_a,
                       torch::Tensor src_b, std::vector<int64_t> slots,
                       std::vector<int64_t> starts,
                       std::vector<int64_t> counts) {
  TORCH_CHECK(pool_a.dim() == 3 && pool_b.dim() == 3, "pools are [pages, rows, dim]");
  TORCH_CHECK(src_a.dim() == 2 && src_b.dim() == 2, "sources are [rows, dim]");
  TORCH_CHECK(table.dim() == 2 && table.scalar_type() == torch::kLong, "table is [slots, pages] int64");
  TORCH_CHECK(pool_a.scalar_type() == src_a.scalar_type() &&
              pool_b.scalar_type() == src_b.scalar_type() &&
              pool_a.scalar_type() == pool_b.scalar_type(), "dtypes must agree");
  TORCH_CHECK(pool_a.size(1) == pool_b.size(1), "pools must share the page geometry");
  TORCH_CHECK(pool_a.size(2) == src_a.size(1) && pool_b.size(2) == src_b.size(1), "row width mismatch");
  TORCH_CHECK(pool_a.is_contiguous() && pool_b.is_contiguous() &&
              src_a.is_contiguous() && src_b.is_contiguous() && table.is_contiguous(),
              "contiguous tensors required");
  TORCH_CHECK(src_a.size(0) == src_b.size(0), "sources must carry the same rows");
  const int64_t B = (int64_t)slots.size();
  TORCH_CHECK(B == (int64_t)starts.size() && B == (int64_t)counts.size(), "plan sizes must agree");
  TORCH_CHECK(B <= LJQ_MAX_SLOTS, "batch exceeds the compiled slot bound");
  if (B == 0 || src_a.size(0) == 0) return;
  TORCH_CHECK(src_a.size(0) % B == 0, "sources must hold the same row count per request");
  const int per = (int)(src_a.size(0) / B);
  CommitPlan plan{};
  int64_t widest = 0;
  for (int64_t b = 0; b < B; ++b) {
    TORCH_CHECK(counts[b] >= 0 && counts[b] <= per, "count outside the window");
    TORCH_CHECK(slots[b] >= 0 && slots[b] < table.size(0), "slot out of range");
    TORCH_CHECK(starts[b] >= 0, "negative start row");
    plan.slot[b] = slots[b];
    plan.start[b] = starts[b];
    plan.count[b] = counts[b];
    // A row may only publish into pages its own slot has claimed.  Without
    // this the kernel indexes past the slot's stretch of the table, reads a
    // neighbour's page number -- a perfectly valid one, so the page<0 guard
    // never fires -- and publishes into another request's kv.
    if (counts[b] > 0) {
      const int64_t last_page = (starts[b] + counts[b] - 1) / pool_a.size(1);
      TORCH_CHECK(last_page < table.size(1),
                  "commit past the slot's page table: b=", b,
                  " slot=", slots[b], " start=", starts[b],
                  " count=", counts[b], " last_page=", last_page,
                  " pages=", table.size(1));
    }
    widest = std::max(widest, counts[b]);
  }
  if (widest == 0) return;
  const c10::cuda::CUDAGuard guard(pool_a.device());
  const dim3 grid((unsigned)widest, (unsigned)B, 2);
  const int dim_a = (int)pool_a.size(2), dim_b = (int)pool_b.size(2);
  const int threads = std::min(256, std::max(32, ((std::max(dim_a, dim_b) + 31) / 32) * 32));
  AT_DISPATCH_SWITCH(
      pool_a.scalar_type(), "paged_commit_pair",
      AT_DISPATCH_CASE(torch::kBFloat16, [&] {
        paged_commit_kernel<at::BFloat16><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            pool_a.data_ptr<at::BFloat16>(), pool_b.data_ptr<at::BFloat16>(),
            table.data_ptr<int64_t>(), src_a.data_ptr<at::BFloat16>(),
            src_b.data_ptr<at::BFloat16>(), plan, (int)pool_a.size(1), dim_a, dim_b,
            per, (int)table.stride(0));
      })
      AT_DISPATCH_CASE(torch::kHalf, [&] {
        paged_commit_kernel<at::Half><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            pool_a.data_ptr<at::Half>(), pool_b.data_ptr<at::Half>(),
            table.data_ptr<int64_t>(), src_a.data_ptr<at::Half>(),
            src_b.data_ptr<at::Half>(), plan, (int)pool_a.size(1), dim_a, dim_b,
            per, (int)table.stride(0));
      })
      AT_DISPATCH_CASE(torch::kFloat, [&] {
        paged_commit_kernel<float><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            pool_a.data_ptr<float>(), pool_b.data_ptr<float>(),
            table.data_ptr<int64_t>(), src_a.data_ptr<float>(),
            src_b.data_ptr<float>(), plan, (int)pool_a.size(1), dim_a, dim_b,
            per, (int)table.stride(0));
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_commit_pair", &paged_commit_pair,
        "single-launch masked paged publish of the kv/index row pair");
}
