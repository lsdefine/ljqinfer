#include "prefill_native.h"
using at::Tensor;
using ljq::launch;
extern "C" {
void pa_rope(void*,void*,void*,void*,int,int,int,int,int);
void pa_unrope(void*,void*,void*,void*,int,int,int,int,int);
void pa_freqs(void*,void*,void*,void*,int,int,int);
void pa_fp8_batch(void*,void*,void*,int,int);
void pa_fp4e4_batch(void*,void*,void*,int,int);
void pa_fp4pow2_batch(void*,void*,void*,int,int);
void pa_compress(void*,void*,void*,void*,void*,void*,void*,void*,int,int,int);
void pa_carry(void*,void*,void*,void*,void*,void*,int,int,int);
void w8_dequant_launch(uint32_t,void*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t);
void pa_score_reduce(void*,void*,void*,void*,void*,void*,int,int,int,int,int,int,int,int,float,int);
void pa_candidate_max(void*,void*,void*,void*,int,int,int,int,int,int,int);
void pa_candidate_expand(void*,void*,void*,void*,void*,void*,int,int,int,int,int,int,int);
void pa_candidate_mask(void*,void*,void*,void*,int,int,int,int,int,int,int);
void sorted_finish(void*,void*,void*,void*,int,int,int,int,int,int);
void joint_pack(void*,void*,void*,void*,void*,void*,void*,void*,void*,int,int,int,int,int,int,int,int);
void joint_correct(void*,void*,void*,void*,void*,void*,int,int,int,int,int);
void pa_swa_correct(void*,void*,void*,void*,void*,void*,int,int,int,int,int);
void paged_read(void*,void*,void*,void*,void*,void*,int,int,int,int,int);
}
namespace {
void valid(const Tensor& x,at::ScalarType dtype) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && x.scalar_type()==dtype && x.is_contiguous(),
                "unexpected NPU tensor dtype or noncontiguous layout");
}
Tensor rope(const Tensor& x,const Tensor& f,bool inverse) {
    valid(x,at::kBFloat16); valid(f,at::kFloat);
    TORCH_CHECK((x.dim()==2 || x.dim()==3) && f.dim()==3 && f.size(0)==x.size(0) &&
                f.size(2)==2 && f.size(1)*2<=x.size(-1) && x.device()==f.device(),"RoPE shape/device mismatch");
    c10::DeviceGuard guard(x.device()); auto out=at::empty_like(x);
    if(x.numel()) launch("ljq_rope",{x,f,out},[=](void* s){
        (inverse?pa_unrope:pa_rope)(s,x.data_ptr(),f.data_ptr(),out.data_ptr(),x.size(0),
            x.dim()==3?x.size(1):1,x.size(-1),f.size(1)*2,20); return 0;
    }); return out;
}
Tensor qdq(const Tensor& input,int64_t kind) {
    valid(input,at::kBFloat16); TORCH_CHECK(kind>=0 && kind<3 && input.numel()%32==0,"invalid QDQ block or kind");
    c10::DeviceGuard guard(input.device()); auto out=at::empty_like(input);
    if(input.numel()) launch("ljq_qdq",{input,out},[=](void* s){
        auto fn=kind==0?pa_fp8_batch:kind==1?pa_fp4e4_batch:pa_fp4pow2_batch;
        fn(s,input.data_ptr(),out.data_ptr(),input.numel(),20); return 0;
    }); return out;
}
Tensor frequencies(const Tensor& table,const Tensor& positions) {
    valid(table,at::kFloat); valid(positions,at::kLong);
    TORCH_CHECK(table.dim()==3 && table.size(2)==2 && positions.dim()==1 && table.device()==positions.device(),"frequency shape/device mismatch");
    c10::DeviceGuard guard(table.device());
    auto out=at::empty({positions.numel(),table.size(1),2},table.options());
    if(positions.numel()) launch("ljq_freqs",{table,positions,out},[=](void* s){
        pa_freqs(s,table.data_ptr(),positions.data_ptr(),out.data_ptr(),positions.numel(),table.size(1)*2,20);return 0;
    }); return out;
}
Tensor compress(const Tensor& v,const Tensor& scores,Tensor cv,Tensor cs,int64_t start) {
    valid(v,at::kFloat); valid(scores,at::kFloat); valid(cv,at::kFloat); valid(cs,at::kFloat);
    TORCH_CHECK(v.dim()==2 && v.size(1)>0 && v.size(1)<=512 && v.size(1)%8==0 &&
                v.sizes()==scores.sizes() && cv.numel()==4*v.size(1) && cs.numel()==4*v.size(1) && start>=0,
                "compression shapes mismatch");
    TORCH_CHECK(v.device()==scores.device() && v.device()==cv.device() && v.device()==cs.device(),"compression device mismatch");
    c10::DeviceGuard guard(v.device());
    auto cursor=at::full({1},start,v.options().dtype(at::kLong));
    auto out=at::empty({(v.size(0)+1)/2,v.size(1)},v.options());
    auto meta=at::empty({2},v.options().dtype(at::kLong));
    if(v.size(0)) launch("ljq_compress",{v,scores,cv,cs,cursor,out,meta},[=](void* s){
        pa_compress(s,v.data_ptr(),scores.data_ptr(),cv.data_ptr(),cs.data_ptr(),cursor.data_ptr(),out.data_ptr(),meta.data_ptr(),v.size(0),v.size(1),20);
        pa_carry(s,v.data_ptr(),scores.data_ptr(),cv.data_ptr(),cs.data_ptr(),cursor.data_ptr(),v.size(0),v.size(1),20);return 0;
    }); return out.slice(0,0,(start+v.size(0))/2-start/2);
}
Tensor w8_dequant(const Tensor& w,const Tensor& scale) {
    valid(w,at::kChar); valid(scale,at::kFloat);
    TORCH_CHECK(w.dim()==2 && w.size(1)%32==0 && scale.numel()==w.size(0) && w.device()==scale.device(),"W8 shape/device mismatch");
    c10::DeviceGuard guard(w.device()); auto out=at::empty(w.sizes(),w.options().dtype(at::kBFloat16));
    if(w.numel()) launch("ljq_w8_dequant",{w,scale,out},[=](void* s){
        w8_dequant_launch(40,s,(uint8_t*)w.data_ptr(),(uint8_t*)scale.data_ptr(),(uint8_t*)out.data_ptr(),w.size(0),w.size(1));return 0;
    }); return out;
}
// Tile-local Tensor interfaces: offsets are represented by Tensor views, not plans.
void together(const Tensor& ref,std::initializer_list<Tensor> xs) {
    for(const auto& x:xs) TORCH_CHECK(ref.device()==x.device(),"selection device mismatch");
}
void matrix(const Tensor& x,at::ScalarType dtype) {
    valid(x,dtype); TORCH_CHECK(x.dim()==2,"selection expects a matrix");
}
void positions_for(const Tensor& x,int64_t rows) {
    valid(x,at::kLong); TORCH_CHECK(x.dim()==1 && x.numel()==rows,"position count mismatch");
}
Tensor score_reduce(const Tensor& dot,const Tensor& weights,const Tensor& pos,const Tensor& count,
                    int64_t keys,int64_t ratio,double scale,Tensor out) {
    matrix(dot,at::kFloat); matrix(weights,at::kFloat); positions_for(pos,weights.size(0)); valid(count,at::kLong);
    int64_t t=weights.size(0),h=weights.size(1),k=dot.size(1);
    TORCH_CHECK(h>0 && dot.size(0)==t*h && k>0 && k%32==0 && keys>=0 && keys<=k && ratio>0 && count.numel()==1,"score shapes mismatch");
    together(dot,{weights,pos,count}); c10::DeviceGuard guard(dot.device());
    matrix(out,at::kFloat); together(dot,{out});
    TORCH_CHECK(out.size(0)==t && out.size(1)==k,"score output contract mismatch");
    if(t) launch("ljq_score_reduce",{dot,weights,pos,count,out},[=](void* st){
        pa_score_reduce(st,dot.data_ptr(),weights.data_ptr(),pos.data_ptr(),count.data_ptr(),out.data_ptr(),t,h,keys,ratio,0,0,t,k,scale,20);return 0;
    }); return out;
}
Tensor candidate_blocks(const Tensor& scores,const Tensor& pos,int64_t ratio) {
    matrix(scores,at::kFloat); positions_for(pos,scores.size(0)); together(scores,{pos});
    int64_t t=scores.size(0),k=scores.size(1),nb=((k+7)/8+31)/32*32;
    TORCH_CHECK(k>0 && k%32==0 && ratio>0,"candidate block shape mismatch");
    c10::DeviceGuard guard(scores.device());auto out=at::empty({t,nb},scores.options());
    if(t) launch("ljq_candidate_blocks",{scores,pos,out},[=](void* st){
        pa_candidate_max(st,scores.data_ptr(),pos.data_ptr(),out.data_ptr(),t,ratio,0,t,k,nb,20);return 0;
    });return out;
}
Tensor candidate_expand(const Tensor& scores,const Tensor& ids,const Tensor& pos,const Tensor& count,int64_t keys,int64_t ratio) {
    matrix(scores,at::kFloat);matrix(ids,at::kLong);positions_for(pos,scores.size(0));valid(count,at::kLong);
    TORCH_CHECK(scores.sizes()==ids.sizes() && scores.size(1)>0 && scores.size(1)%32==0 && count.numel()==1 && keys>=0 && ratio>0,"candidate expansion shape mismatch");
    together(scores,{ids,pos,count});c10::DeviceGuard guard(scores.device());
    auto out=at::empty({scores.size(0),16384},ids.options());
    if(scores.size(0)) launch("ljq_candidate_expand",{scores,ids,pos,count,out},[=](void* st){
        pa_candidate_expand(st,scores.data_ptr(),ids.data_ptr(),pos.data_ptr(),count.data_ptr(),out.data_ptr(),scores.size(0),keys,ratio,0,scores.size(0),scores.size(1),20);return 0;
    });return out;
}
Tensor candidate_mask(const Tensor& scores,const Tensor& ids,int64_t keys) {
    matrix(scores,at::kFloat);matrix(ids,at::kLong);together(scores,{ids});
    TORCH_CHECK(scores.size(0)==ids.size(0) && scores.size(1)>0 && scores.size(1)%32==0 && ids.size(1)%4==0 && keys>=0 && keys<=scores.size(1),"candidate mask shape mismatch");
    c10::DeviceGuard guard(scores.device());auto out=at::empty_like(scores);
    if(scores.size(0)) launch("ljq_candidate_mask",{scores,ids,out},[=](void* st){
        pa_candidate_mask(st,scores.data_ptr(),ids.data_ptr(),out.data_ptr(),scores.size(0),0,scores.size(0),scores.size(1),ids.size(1),keys,20);return 0;
    });return out;
}
Tensor sorted_ids(const Tensor& scores,const Tensor& ids,int64_t topk,Tensor out) {
    matrix(scores,at::kFloat);matrix(ids,at::kLong);together(scores,{ids});
    TORCH_CHECK(scores.sizes()==ids.sizes() && scores.size(1)>0 && scores.size(1)%32==0 && topk>0 && topk<=512 && topk%4==0,"sorted selection shape mismatch");
    matrix(out,at::kLong); together(scores,{out});
    TORCH_CHECK(out.size(0)==scores.size(0) && out.size(1)==topk,"selection output contract mismatch");
    c10::DeviceGuard guard(scores.device());
    if(scores.size(0)) launch("ljq_sorted_ids",{scores,ids,out},[=](void* st){
        sorted_finish(st,scores.data_ptr(),ids.data_ptr(),out.data_ptr(),scores.size(0),0,scores.size(0),scores.size(1),topk,20);return 0;
    });return out;
}
// Tensor ownership and shape checks live here; released kernels stay unchanged.
std::tuple<Tensor,Tensor,Tensor> pack_joint(const Tensor& local,const Tensor& bank,
        const Tensor& ids,const Tensor& pos,const Tensor& meta,int64_t ratio) {
    matrix(local,at::kBFloat16);matrix(bank,at::kBFloat16);matrix(ids,at::kLong);
    positions_for(pos,ids.size(0));valid(meta,at::kLong);
    int64_t t=ids.size(0),d=local.size(1),k=ids.size(1),n=128+k;
    TORCH_CHECK(d>0 && d<=512 && d%16==0 && bank.size(1)==d &&
                k>0 && k<=512 && k%32==0 && meta.numel()==2 && ratio>0,
                "joint pack shape mismatch");
    together(local,{bank,ids,pos,meta});c10::DeviceGuard guard(local.device());
    auto packed=at::empty({1,std::max(local.size(0)+bank.size(0)+1,n),1,d},local.options());
    auto indices=at::empty({1,t,1,n},ids.options().dtype(at::kInt));
    auto missing=at::empty({t,32},indices.options());
    launch("ljq_joint_pack",{local,bank,ids,pos,meta,packed,indices,missing},[=](void* st){
        joint_pack(st,local.data_ptr(),bank.data_ptr(),ids.data_ptr(),pos.data_ptr(),meta.data_ptr(),
                   packed.data_ptr(),indices.data_ptr(),missing.data_ptr(),t,local.size(0),bank.size(0),d,k,128,ratio,20);return 0;
    });return {packed,indices,missing};
}
void attention_output(const Tensor& out,const Tensor& mx,const Tensor& sm,int64_t stats) {
    valid(out,at::kBFloat16);valid(mx,at::kFloat);valid(sm,at::kFloat);
    TORCH_CHECK(out.dim()==3 && out.size(1)>0 && out.size(2)>0 && out.size(2)<=512 &&
                out.size(2)%16==0 && mx.numel()==out.size(0)*out.size(1)*stats && sm.sizes()==mx.sizes(),
                "attention correction shape mismatch");
    together(out,{mx,sm});
}
Tensor correct_joint(Tensor out,const Tensor& mx,const Tensor& sm,const Tensor& missing,
                     const Tensor& sink,int64_t count) {
    attention_output(out,mx,sm,1);matrix(missing,at::kInt);valid(sink,at::kFloat);
    TORCH_CHECK(missing.size(0)==out.size(0) && missing.size(1)==32 &&
                sink.numel()==out.size(1) && count>=160 && count<=640 && count%32==0,
                "joint correction metadata mismatch");
    together(out,{missing,sink});c10::DeviceGuard guard(out.device());
    if(out.numel())launch("ljq_joint_correct",{out,mx,sm,missing,sink},[=](void* st){
        joint_correct(st,out.data_ptr(),mx.data_ptr(),sm.data_ptr(),missing.data_ptr(),sink.data_ptr(),
                      out.size(0),out.size(1),out.size(2),count,20);return 0;
    });return out;
}
Tensor correct_swa(Tensor out,const Tensor& mx,const Tensor& sm,const Tensor& pos,const Tensor& meta) {
    attention_output(out,mx,sm,8);positions_for(pos,out.size(0));valid(meta,at::kLong);
    TORCH_CHECK(meta.numel()==2,"SWA metadata mismatch");together(out,{pos,meta});
    c10::DeviceGuard guard(out.device());
    if(out.numel())launch("ljq_swa_correct",{out,mx,sm,pos,meta},[=](void* st){
        pa_swa_correct(st,out.data_ptr(),mx.data_ptr(),sm.data_ptr(),pos.data_ptr(),meta.data_ptr(),
                       out.size(0),out.size(1),out.size(2),128,20);return 0;
    });return out;
}
Tensor read_paged(const Tensor& data,const Tensor& table,const Tensor& slot,const Tensor& count,Tensor out) {
    valid(data,at::kBFloat16);matrix(table,at::kLong);valid(slot,at::kLong);valid(count,at::kLong);
    matrix(out,at::kBFloat16);
    TORCH_CHECK(data.dim()==3 && data.size(1)>0 && data.size(2)==out.size(1) &&
                out.size(1)>0 && out.size(1)<=512 && out.size(1)%16==0 &&
                table.size(0)>0 && table.size(1)>0 && out.size(0)<=table.size(1)*data.size(1) &&
                slot.numel()==1 && count.numel()==1,"paged read shape mismatch");
    together(data,{table,slot,count,out});c10::DeviceGuard guard(data.device());
    // Slot, count and physical page IDs are caller-owned device metadata.
    if(out.numel())launch("ljq_paged_read",{data,table,slot,count,out},[=](void* st){
        paged_read(st,data.data_ptr(),table.data_ptr(),slot.data_ptr(),count.data_ptr(),out.data_ptr(),
                   out.size(0),out.size(1),data.size(1),table.size(1),20);return 0;
    });return out;
}
}
TORCH_LIBRARY_FRAGMENT(ljq_prefill,m) {
    m.def("joint_pack(Tensor local, Tensor bank, Tensor ids, Tensor positions, Tensor meta, int ratio) -> (Tensor, Tensor, Tensor)");
    m.def("joint_correct(Tensor(a!) out, Tensor maximum, Tensor denominator, Tensor missing, Tensor sink, int count) -> Tensor(a!)");
    m.def("swa_correct(Tensor(a!) out, Tensor maximum, Tensor denominator, Tensor positions, Tensor meta) -> Tensor(a!)");
    m.def("paged_read(Tensor data, Tensor table, Tensor slot, Tensor count, Tensor(a!) out) -> Tensor(a!)");
    m.def("rope(Tensor x, Tensor frequencies, bool inverse=False) -> Tensor");
    m.def("qdq(Tensor x, int kind) -> Tensor");
    m.def("frequencies(Tensor table, Tensor positions) -> Tensor");
    m.def("compress(Tensor values, Tensor scores, Tensor(a!) carry_values, Tensor(b!) carry_scores, int start) -> Tensor");
    m.def("w8_dequant(Tensor weight, Tensor scale) -> Tensor");
    m.def("score_reduce(Tensor dot, Tensor weights, Tensor positions, Tensor valid, int keys, int ratio, float scale, Tensor(a!) out) -> Tensor(a!)");
    m.def("candidate_blocks(Tensor scores, Tensor positions, int ratio) -> Tensor");
    m.def("candidate_expand(Tensor scores, Tensor ids, Tensor positions, Tensor valid, int keys, int ratio) -> Tensor");
    m.def("candidate_mask(Tensor scores, Tensor ids, int keys) -> Tensor");
    m.def("sorted_ids(Tensor scores, Tensor ids, int topk, Tensor(a!) out) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(ljq_prefill,PrivateUse1,m) {
    m.impl("joint_pack",pack_joint);m.impl("joint_correct",correct_joint);
    m.impl("swa_correct",correct_swa);m.impl("paged_read",read_paged);
    m.impl("rope",rope); m.impl("qdq",qdq); m.impl("frequencies",frequencies);
    m.impl("compress",compress); m.impl("w8_dequant",w8_dequant);
    m.impl("score_reduce",score_reduce);m.impl("candidate_blocks",candidate_blocks);
    m.impl("candidate_expand",candidate_expand);m.impl("candidate_mask",candidate_mask);m.impl("sorted_ids",sorted_ids);
}
