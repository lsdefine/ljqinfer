// Tensor boundary for the existing eager kernels. No bound addresses or plans.
#include "prefill_native.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" {
int pre_hc_collapse_launch(void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
int pre_hc_expand_launch(void*,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
int pre_hc_cast_stats_launch(void*,void*,void*,void*,uint32_t,float);
int pre_hc_sinkhorn_launch(void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
void routed_swiglu_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
void routed_combine_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
}

namespace {
using at::Tensor;
using ljq::launch;
void check(const Tensor& t, at::ScalarType dtype, const Tensor& reference) {
    TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1 &&
                t.device()==reference.device() && t.scalar_type()==dtype,
                "prefill: incompatible tensor device or dtype");
}
// Keep Tensor objects until host submission completes; recordStream protects
// storage until device completion, including allocations made on other streams.
void hc_shape(const Tensor& x) {
    check(x,at::kBFloat16,x);
    TORCH_CHECK(x.dim()==3 && x.size(1)==4 && x.size(2)>0 && x.size(2)%2560==0,
                "HC requires BF16[T,4,D], D divisible by 2560");
    TORCH_CHECK(x.size(0)<=UINT32_MAX,"HC row count overflow");
}
Tensor collapse(const Tensor& input,const Tensor& pre) {
    hc_shape(input); check(pre,at::kFloat,input);
    TORCH_CHECK(pre.sizes()==input.sizes().slice(0,2),"HC pre shape mismatch");
    c10::DeviceGuard guard(input.device());
    auto x=input.contiguous(), p=pre.contiguous();
    auto out=at::empty({x.size(0),x.size(2)},x.options());
    if(x.size(0)) launch("ljq_hc_collapse",{x,p,out},[=](void* s){
        return pre_hc_collapse_launch(s,x.data_ptr(),p.data_ptr(),out.data_ptr(),x.size(0),4,x.size(2),24);
    });
    return out;
}
Tensor expand(const Tensor& input,const Tensor& residual,const Tensor& post,const Tensor& comb) {
    hc_shape(residual); check(input,at::kBFloat16,residual);
    check(post,at::kFloat,residual); check(comb,at::kFloat,residual);
    auto t=residual.size(0), d=residual.size(2);
    TORCH_CHECK(input.sizes()==at::IntArrayRef({t,d}) &&
                post.sizes()==at::IntArrayRef({t,4}) && comb.sizes()==at::IntArrayRef({t,4,4}),
                "HC expand shape mismatch");
    c10::DeviceGuard guard(residual.device());
    auto x=input.contiguous(), r=residual.contiguous(), p=post.contiguous(), c=comb.contiguous();
    auto out=at::empty(r.sizes(),r.options());
    if(t) launch("ljq_hc_expand",{x,r,p,c,out},[=](void* s){
        return pre_hc_expand_launch(s,x.data_ptr(),r.data_ptr(),p.data_ptr(),c.data_ptr(),out.data_ptr(),t,4,d,24);
    });
    return out;
}
std::tuple<Tensor,Tensor> cast_stats(const Tensor& input,double eps) {
    hc_shape(input); TORCH_CHECK(input.size(2)==5120 && eps>0,"HC statistics requires D=5120 and positive epsilon");
    c10::DeviceGuard guard(input.device());
    auto x=input.contiguous(); auto t=x.size(0);
    auto flat=at::empty({t,20480},x.options().dtype(at::kFloat));
    auto stats=at::empty({t,1},flat.options());
    if(t) launch("ljq_hc_cast_stats",{x,flat,stats},[=](void* s){
        return pre_hc_cast_stats_launch(s,x.data_ptr(),flat.data_ptr(),stats.data_ptr(),t,eps);
    });
    return {flat,stats};
}
Tensor sinkhorn(const Tensor& input,int64_t iters) {
    check(input,at::kFloat,input);
    TORCH_CHECK(input.dim()==3 && input.size(1)==4 && input.size(2)==4 && iters>=0 && iters<=UINT32_MAX,
                "Sinkhorn requires FP32[T,4,4] and a nonnegative iteration count");
    c10::DeviceGuard guard(input.device());
    auto t=input.size(0), rows=(t+7)/8*8;
    TORCH_CHECK(rows<=UINT32_MAX,"Sinkhorn row count overflow");
    if(!t || !iters) return input.clone();
    auto buf=at::zeros({4,4,rows},input.options());
    buf.slice(2,0,t).copy_(input.permute({1,2,0}));
    launch("ljq_hc_sinkhorn",{buf},[=](void* s){
        return pre_hc_sinkhorn_launch(s,buf.data_ptr(),rows,4,iters,24);
    });
    return buf.slice(2,0,t).permute({2,0,1}).contiguous();
}
Tensor routed_swiglu(const Tensor& input,const Tensor& probability) {
    check(input,at::kBFloat16,input); check(probability,at::kFloat,input);
    TORCH_CHECK(input.dim()==2 && input.size(1)==576 && probability.numel()==input.size(0),"routed SwiGLU shape mismatch");
    c10::DeviceGuard guard(input.device());
    auto x=input.contiguous(), p=probability.contiguous();
    auto out=at::empty({x.size(0),288},x.options());
    if(x.size(0)) launch("ljq_routed_swiglu",{x,p,out},[=](void* s){
        routed_swiglu_launch(20,s,(uint8_t*)x.data_ptr(),(uint8_t*)p.data_ptr(),(uint8_t*)out.data_ptr(),x.size(0));return 0;
    });
    return out;
}
Tensor routed_combine(const Tensor& input,const Tensor& inverse) {
    check(input,at::kBFloat16,input); check(inverse,at::kLong,input);
    TORCH_CHECK(input.dim()==2 && input.size(1)==5120 && inverse.dim()==1 &&
                inverse.numel()%6==0 && inverse.numel()==input.size(0),
                "routed combine requires INT64[6*T] inverse permutation");
    c10::DeviceGuard guard(input.device());
    auto x=input.contiguous(), idx=inverse.contiguous();
    auto out=at::empty({idx.numel()/6,5120},x.options().dtype(at::kFloat));
    if(idx.numel()) launch("ljq_routed_combine",{x,idx,out},[=](void* s){
        routed_combine_launch(20,s,(uint8_t*)x.data_ptr(),(uint8_t*)idx.data_ptr(),(uint8_t*)out.data_ptr(),idx.numel()/6);return 0;
    });
    return out;
}
}
TORCH_LIBRARY(ljq_prefill,m) {
    m.def("hc_collapse(Tensor x, Tensor pre) -> Tensor");
    m.def("hc_expand(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
    m.def("hc_cast_stats(Tensor x, float eps) -> (Tensor, Tensor)");
    m.def("hc_sinkhorn(Tensor x, int iters) -> Tensor");
    m.def("routed_swiglu(Tensor x, Tensor probability) -> Tensor");
    m.def("routed_combine(Tensor x, Tensor inverse) -> Tensor");
}
TORCH_LIBRARY_IMPL(ljq_prefill,PrivateUse1,m) {
    m.impl("hc_collapse",collapse); m.impl("hc_expand",expand);
    m.impl("hc_cast_stats",cast_stats); m.impl("hc_sinkhorn",sinkhorn);
    m.impl("routed_swiglu",routed_swiglu); m.impl("routed_combine",routed_combine);
}
