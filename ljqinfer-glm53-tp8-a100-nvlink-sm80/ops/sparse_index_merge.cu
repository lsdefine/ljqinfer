#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <climits>
#include <cstdint>
#include <cmath>
using torch::Tensor;
static void check(const Tensor& t, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type()==dtype && t.is_contiguous(),
              "requires contiguous CUDA tensor of expected dtype");
}
static void same(const Tensor& a,const Tensor& b) {
  TORCH_CHECK(a.device()==b.device(), "cross-device input");
}
__device__ __forceinline__ unsigned score_order(float x) {
  // Torch descending: all NaNs first and equal; +/-0 equal. Original score
  // payload is never reconstructed from this canonicalized sorting key.
  if (isnan(x)) return 0u;
  unsigned u=__float_as_uint(x==0.f ? 0.f : x);
  return ~((u & 0x80000000u) ? ~u : (u ^ 0x80000000u));
}
struct RowInput {
  const float *best,*scores;
  const int64_t *old_ids,*new_ids,*positions;
  int old_width,new_width,rows;
  int64_t ids_stride, positions_stride, begin, key_count, ratio;
  bool implicit_ids,mask;
  __device__ int64_t id(int r,int i) const {
    if(i<old_width) return old_ids[(int64_t)r*old_width+i];
    int j=i-old_width;
    return implicit_ids ? begin+j : new_ids[(int64_t)r*ids_stride+j];
  }
  __device__ float score(int r,int i) const {
    if(i<old_width) return best[(int64_t)r*old_width+i];
    int j=i-old_width;
    float s=scores[(int64_t)r*new_width+j];
    if(mask) {
      int64_t ix=id(r,i);
      // Production positions nonnegative. Integer division equals Python floor.
      if(ix<0 || ix>=key_count || ix>=(positions[(int64_t)r*positions_stride]+1)/ratio)
        return -INFINITY;
    }
    return s;
  }
};
// Two stable LSD stages represent the entire (score desc,id asc,input offset)
// ordering WITHOUT narrowing int64 IDs or using epsilon perturbations.
template<int ITEMS>
__global__ void radix_rows(RowInput in,float* out_s,int64_t* out_i,int take,bool final) {
  using IdSort=cub::BlockRadixSort<unsigned long long,256,ITEMS,int>;
  using ScoreSort=cub::BlockRadixSort<unsigned,256,ITEMS,int>;
  __shared__ union { typename IdSort::TempStorage ids; typename ScoreSort::TempStorage scores; } temp;
  unsigned long long ik[ITEMS]; unsigned sk[ITEMS]; int offsets[ITEMS];
  int r=blockIdx.x,n=in.old_width+in.new_width;
  #pragma unroll
  for(int j=0;j<ITEMS;++j) {
    int i=threadIdx.x*ITEMS+j; offsets[j]=i;
    int64_t id=i<n ? in.id(r,i) : -1;
    ik[j]=i<n ? (id<0 ? (unsigned long long)LLONG_MAX : (unsigned long long)id) : ULLONG_MAX;
  }
  IdSort(temp.ids).Sort(ik,offsets);
  __syncthreads();
  #pragma unroll
  for(int j=0;j<ITEMS;++j) sk[j]=offsets[j]<n ? score_order(in.score(r,offsets[j])) : 0xffffffffu;
  ScoreSort(temp.scores).Sort(sk,offsets);
  #pragma unroll
  for(int j=0;j<ITEMS;++j) {
    int dest=threadIdx.x*ITEMS+j;
    if(dest<take) {
      float s=in.score(r,offsets[j]);
      out_s[(int64_t)r*take+dest]=s;
      out_i[(int64_t)r*take+dest]=(final && !isfinite(s)) ? -1 : in.id(r,offsets[j]);
    }
  }
}
static std::vector<Tensor> launch_sort(RowInput in,const Tensor& scores,int take,bool final,Tensor os,Tensor oi) {
  int n=in.old_width+in.new_width;
  TORCH_CHECK(n<=8192 && take>=0 && take<=n, "radix width/take outside supported contract");
  check(os,at::kFloat);check(oi,at::kLong);same(scores,os);same(scores,oi);
  TORCH_CHECK(os.dim()==2 && os.size(0)==in.rows && os.size(1)==take && oi.sizes()==os.sizes(),"output shape");
  TORCH_CHECK(os.data_ptr<float>()!=in.best && os.data_ptr<float>()!=in.scores && oi.data_ptr<int64_t>()!=in.old_ids && oi.data_ptr<int64_t>()!=in.new_ids,"output must not alias inputs");
  if(in.rows && take) {
    auto stream=at::cuda::getCurrentCUDAStream();
    if(n<=1024) radix_rows<4><<<in.rows,256,0,stream>>>(in,os.data_ptr<float>(),oi.data_ptr<int64_t>(),take,final);
    else if(n<=2048) radix_rows<8><<<in.rows,256,0,stream>>>(in,os.data_ptr<float>(),oi.data_ptr<int64_t>(),take,final);
    else if(n<=4096) radix_rows<16><<<in.rows,256,0,stream>>>(in,os.data_ptr<float>(),oi.data_ptr<int64_t>(),take,final);
    else radix_rows<32><<<in.rows,256,0,stream>>>(in,os.data_ptr<float>(),oi.data_ptr<int64_t>(),take,final);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {os,oi};
}
std::vector<Tensor> merge(Tensor best,Tensor old_ids,Tensor scores,Tensor ids,
                         Tensor positions,int64_t begin,int64_t key_count,int64_t ratio,
                         bool implicit_ids,bool final,Tensor out_scores,Tensor out_ids) {
  check(best,at::kFloat);check(old_ids,at::kLong);check(scores,at::kFloat);
  check(positions,at::kLong);same(best,old_ids);same(best,scores);same(best,positions);
  TORCH_CHECK(best.dim()==2 && best.sizes()==old_ids.sizes() && scores.dim()==2
      && best.size(0)==scores.size(0) && positions.dim()==1 && positions.size(0)==best.size(0),"merge shape");
  TORCH_CHECK(ratio>0 && begin>=0 && key_count>=0 && best.size(0)<=INT_MAX
      && best.size(1)+scores.size(1)<=8192,"merge bounds");
  if(!implicit_ids) {
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type()==at::kLong && ids.dim()==2 && ids.stride(1)==1
       && ids.sizes()==scores.sizes(),"candidate layout");same(best,ids);
  }
  c10::cuda::CUDAGuard guard(scores.device());
  RowInput in{};in.best=best.data_ptr<float>();in.old_ids=old_ids.data_ptr<int64_t>();
  in.scores=scores.data_ptr<float>();in.positions=positions.data_ptr<int64_t>();
  in.old_width=best.size(1);in.new_width=scores.size(1);in.rows=best.size(0);
  in.new_ids=implicit_ids?nullptr:ids.data_ptr<int64_t>();in.ids_stride=implicit_ids?0:ids.stride(0);
  in.positions_stride=positions.stride(0);in.begin=begin;in.key_count=key_count;in.ratio=ratio;
  in.implicit_ids=implicit_ids;in.mask=true;
  return launch_sort(in,scores,in.old_width,final,out_scores,out_ids);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("merge",&merge); }
