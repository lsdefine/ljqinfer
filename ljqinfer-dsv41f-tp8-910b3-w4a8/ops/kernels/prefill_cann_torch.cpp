#include "prefill_native.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_dynamic_quant.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"
#include "aclnn/acl_meta.h"
#include <memory>
#include <cmath>
#include <limits>
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_sparse_flash_attention.h"
#include "aclnnop/aclnn_lightning_indexer.h"

namespace {
using at::Tensor;
// Call-local descriptors only: no plan, address binding or executor cache.
struct Descriptors {
    std::vector<aclTensor*> tensors;
    std::vector<aclTensorList*> lists;
    aclIntArray* tuning=nullptr;
    std::vector<aclIntArray*> arrays;
    ~Descriptors() {
        for(auto* l:lists) aclDestroyTensorList(l);
        for(auto* t:tensors) if(t) aclDestroyTensor(t);
        if(tuning) aclDestroyIntArray(tuning);
        for(auto* a:arrays) aclDestroyIntArray(a);
    }
    aclIntArray* lengths(int64_t n) {
        auto* a=aclCreateIntArray(&n,1);TORCH_CHECK(a,"sequence length descriptor failed");
        arrays.push_back(a);return a;
    }
    aclTensorList* list(aclTensor* t) {
        auto* l=aclCreateTensorList(&t,1);
        TORCH_CHECK(l,"aclCreateTensorList failed");
        lists.push_back(l);
        for(auto& p:tensors) if(p==t) {p=nullptr;break;}
        return l; // TensorList owns its member descriptor.
    }
    aclTensor* packed(const Tensor& t,int64_t outputs) {
        int64_t shape[]={t.size(0),t.size(1),outputs};
        int64_t stride[]={t.size(1)*outputs,outputs,1};
        int64_t storage=t.numel()*8;
        auto* d=aclCreateTensor(shape,3,ACL_INT4,stride,0,ACL_FORMAT_ND,
                                &storage,1,t.data_ptr());
        TORCH_CHECK(d,"packed INT4 descriptor failed");
        tensors.push_back(d);return d;
    }
    aclTensor* add(const Tensor& t) {
        aclDataType dtype;
        switch(t.scalar_type()) {
        case at::kBFloat16: dtype=ACL_BF16;break;
        case at::kHalf: dtype=ACL_FLOAT16;break;
        case at::kFloat: dtype=ACL_FLOAT;break;
        case at::kChar: dtype=ACL_INT8;break;
        case at::kLong: dtype=ACL_INT64;break;
        case at::kInt: dtype=ACL_INT32;break;
        case at::kBool: dtype=ACL_BOOL;break;
        default: TORCH_CHECK(false,"unsupported CANN tensor dtype");
        }
        int64_t storage=t.storage().nbytes()/t.element_size();
        auto* d=aclCreateTensor(t.sizes().data(),t.dim(),dtype,t.strides().data(),
            t.storage_offset(),ACL_FORMAT_ND,&storage,1,t.storage().mutable_data());
        TORCH_CHECK(d,"aclCreateTensor failed");tensors.push_back(d);return d;
    }
};
// Experimental single serialized engine lane. Bind only after device sync,
// before capturing CED; retain storage until every graph has been reset.
Tensor fixed_workspace;
void bind_workspace(const Tensor& scratch) {
    TORCH_CHECK(scratch.device().type()==c10::DeviceType::PrivateUse1 &&
                scratch.scalar_type()==at::kByte && scratch.is_contiguous() &&
                scratch.dim()==1 && scratch.numel()>0,
                "workspace must be a nonempty contiguous NPU byte vector");
    TORCH_CHECK(!fixed_workspace.defined(), "workspace is immutable once bound");
    fixed_workspace=scratch;
}
// Caller drains the lane and resets all referencing graphs before unbinding.
void unbind_workspace(const Tensor& scratch) {
    TORCH_CHECK(fixed_workspace.defined() &&
                fixed_workspace.is_same(scratch), "workspace owner mismatch");
    fixed_workspace=Tensor();
}
template<class Run>
void submit(const char* name,std::vector<Tensor> tensors,
            std::shared_ptr<Descriptors> desc,uint64_t bytes,aclOpExecutor* executor,Run run) {
    Tensor scratch;
    if(fixed_workspace.defined()) {
        TORCH_CHECK(fixed_workspace.device()==tensors[0].device(),
                    "single-lane workspace device mismatch");
        TORCH_CHECK(bytes<=uint64_t(fixed_workspace.numel()), name,
                    " needs workspace bytes=",bytes," capacity=",fixed_workspace.numel());
        scratch=fixed_workspace;
    } else {
        scratch=at::empty({int64_t(bytes)},tensors[0].options().dtype(at::kByte));
    }
    tensors.push_back(scratch);
    ljq::launch(name,std::move(tensors),[desc,scratch,bytes,executor,run,name](void* stream) {
        auto status=run(bytes?scratch.data_ptr():nullptr,bytes,executor,stream);
        TORCH_CHECK(status==0,name," launch failed: ",status);
        return 0;
    });
}
template<class Query,class Run>
void cann(const char* name,std::vector<Tensor> tensors,Query query,Run run) {
    auto desc=std::make_shared<Descriptors>();
    for(const auto& t:tensors) desc->add(t);
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto status=query(desc->tensors,&bytes,&executor);
    TORCH_CHECK(status==0,name," workspace query failed: ",status);
    submit(name,std::move(tensors),desc,bytes,executor,run);
}
void check(const Tensor& x) {
    TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1,
                "CANN prefill expects an NPU tensor");
}
Tensor matmul_fp32_out(const Tensor& x,const Tensor& weight,Tensor out) {
    check(x);check(weight);
    TORCH_CHECK(x.dim()==2 && weight.dim()==2 && x.size(1)==weight.size(0) &&
                x.scalar_type()==at::kBFloat16 && weight.scalar_type()==at::kBFloat16 &&
                x.device()==weight.device(),"BF16 matmul shape/dtype/device mismatch");
    c10::DeviceGuard guard(x.device());
    TORCH_CHECK(out.device()==x.device() && out.scalar_type()==at::kFloat &&
                out.is_contiguous() && out.dim()==2 && out.size(0)==x.size(0) &&
                out.size(1)==weight.size(1), "matmul output contract mismatch");
    cann("aclnnMatmul",{x,weight,out},[](const auto& d,auto* bytes,auto** executor) {
        return aclnnMatmulGetWorkspaceSize(d[0],d[1],d[2],0,bytes,executor);
    },aclnnMatmul);
    return out;
}
Tensor matmul_fp32(const Tensor& x,const Tensor& weight) {
    auto out=at::empty({x.size(0),weight.size(1)},x.options().dtype(at::kFloat));
    return matmul_fp32_out(x,weight,out);
}
std::tuple<Tensor,Tensor> dynamic_quant(const Tensor& x) {
    check(x);
    TORCH_CHECK(x.dim()==2 && x.is_contiguous() &&
                (x.scalar_type()==at::kBFloat16 || x.scalar_type()==at::kHalf),
                "dynamic quant expects a contiguous BF16/FP16 matrix");
    c10::DeviceGuard guard(x.device());
    auto quantized=at::empty(x.sizes(),x.options().dtype(at::kChar));
    auto scale=at::empty({x.size(0)},x.options().dtype(at::kFloat));
    cann("aclnnDynamicQuant",{x,quantized,scale},[](const auto& d,auto* bytes,auto** executor) {
        return aclnnDynamicQuantGetWorkspaceSize(d[0],nullptr,d[1],d[2],bytes,executor);
    },aclnnDynamicQuant);
    return {quantized,scale};
}
Tensor grouped_int4(const Tensor& x,const Tensor& weight,const Tensor& scale,
                    const Tensor& bias,const Tensor& token_scale,const Tensor& counts) {
    check(x);c10::DeviceGuard guard(x.device());
    for(const auto& t:{weight,scale,bias,token_scale,counts})
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous(),"GMM device/layout mismatch");
    TORCH_CHECK(x.dim()==2 && x.is_contiguous() && x.scalar_type()==at::kChar &&
                weight.dim()==3 && weight.scalar_type()==at::kInt &&
                scale.dim()==3 && scale.size(1)==1 && scale.scalar_type()==at::kLong &&
                bias.dim()==2 && bias.scalar_type()==at::kFloat &&
                token_scale.dim()==1 && token_scale.numel()==x.size(0) &&
                token_scale.scalar_type()==at::kFloat && counts.dim()==1 &&
                counts.scalar_type()==at::kLong,"GMM tensor contract mismatch");
    auto e=weight.size(0), k=weight.size(1), n=scale.size(2);
    TORCH_CHECK(k==x.size(1) && k%64==0 && n%64==0 && weight.size(2)*8==n &&
                scale.size(0)==e && bias.size(0)==e && bias.size(1)==n && counts.numel()==e,"GMM geometry mismatch");
    auto out=at::empty({x.size(0),n},x.options().dtype(at::kHalf));
    auto d=std::make_shared<Descriptors>();
    auto* xs=d->list(d->add(x));auto* ws=d->list(d->packed(weight,n));
    auto* bs=d->list(d->add(bias));auto* ss=d->list(d->add(scale));
    auto* ts=d->list(d->add(token_scale));auto* groups=d->add(counts);
    auto* ys=d->list(d->add(out));int64_t tuning[]={0,1};
    d->tuning=aclCreateIntArray(tuning,2);TORCH_CHECK(d->tuning,"GMM tuning allocation failed");
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    auto status=aclnnGroupedMatmulV5GetWorkspaceSize(xs,ws,bs,ss,nullptr,nullptr,nullptr,
        ts,groups,nullptr,nullptr,nullptr,3,0,1,0,d->tuning,ys,nullptr,nullptr,&bytes,&executor);
    TORCH_CHECK(status==0,"GMM workspace query failed: ",status);
    submit("aclnnGroupedMatmulV5",{x,weight,scale,bias,token_scale,counts,out},
           d,bytes,executor,aclnnGroupedMatmulV5);
    return out;
}
void attention_query(const Tensor& q) {
    check(q);
    TORCH_CHECK(q.dim()==3 && q.is_contiguous() && q.scalar_type()==at::kBFloat16 &&
                q.size(0)>0 && q.size(1)>0 && q.size(2)>0 && q.size(2)<=512 && q.size(2)%16==0,
                "attention expects contiguous BF16 [T,H,D]");
}
std::tuple<Tensor,Tensor,Tensor> flash_swa(const Tensor& q,const Tensor& local,const Tensor& sink) {
    attention_query(q);
    TORCH_CHECK(local.device()==q.device() && local.is_contiguous() && local.dim()==2 &&
                local.scalar_type()==at::kBFloat16 && local.size(0)>0 && local.size(1)==q.size(2) &&
                sink.device()==q.device() && sink.is_contiguous() && sink.scalar_type()==at::kFloat &&
                sink.numel()==q.size(1),"SWA key/sink contract mismatch");
    c10::DeviceGuard guard(q.device());
    auto out=at::empty_like(q),key=local.unsqueeze(1);
    auto mx=at::empty({q.size(0),q.size(1),8},q.options().dtype(at::kFloat));
    auto sm=at::empty_like(mx),empty=at::empty({0},q.options());
    auto mask=at::ones({2048,2048},q.options().dtype(at::kBool)).triu_(1);
    auto desc=std::make_shared<Descriptors>();
    auto* query=desc->add(q);auto* kv=desc->add(key);
    auto* am=desc->add(mask);auto* sk=desc->add(sink);
    auto* maximum=desc->add(mx);auto* denominator=desc->add(sm);
    auto* softmax=desc->add(empty);auto* output=desc->add(out);
    auto* qlen=desc->lengths(q.size(0));auto* klen=desc->lengths(local.size(0));
    uint64_t bytes=0;aclOpExecutor* executor=nullptr;
    char layout[]="TND",softmax_layout[]="";
    auto status=aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(query,nullptr,kv,nullptr,kv,
        nullptr,nullptr,nullptr,am,sk,nullptr,qlen,klen,nullptr,nullptr,
        1.0/std::sqrt(double(q.size(2))),1.0,127,0,q.size(1),layout,0,4,1,softmax_layout,
        maximum,denominator,softmax,output,&bytes,&executor);
    TORCH_CHECK(status==0,"SWA workspace query failed: ",status);
    submit("aclnnFlashAttentionVarLenScoreV5",{q,key,sink,mask,mx,sm,empty,out},
           desc,bytes,executor,aclnnFlashAttentionVarLenScoreV5);
    return {out,mx,sm};
}
std::tuple<Tensor,Tensor,Tensor> flash_sparse(const Tensor& q,const Tensor& packed,const Tensor& indices) {
    attention_query(q);
    TORCH_CHECK(packed.device()==q.device() && packed.is_contiguous() && packed.dim()==4 &&
                packed.scalar_type()==at::kBFloat16 && packed.size(0)==1 && packed.size(1)>0 &&
                packed.size(2)==1 && packed.size(3)==q.size(2) && indices.device()==q.device() &&
                indices.is_contiguous() && indices.scalar_type()==at::kInt && indices.dim()==4 &&
                indices.size(0)==1 && indices.size(1)==q.size(0) && indices.size(2)==1 &&
                indices.size(3)>0 && indices.size(3)%32==0,"sparse attention contract mismatch");
    c10::DeviceGuard guard(q.device());
    auto query=q.unsqueeze(0),out=at::empty_like(q);
    auto output=out.unsqueeze(0);
    auto mx=at::empty({1,1,q.size(0),q.size(1)},q.options().dtype(at::kFloat));
    auto sm=at::empty_like(mx);
    auto qr=at::zeros({1,q.size(0),q.size(1),64},q.options());
    auto kr=at::zeros({1,packed.size(1),1,64},q.options());
    cann("aclnnSparseFlashAttention",{query,packed,indices,qr,kr,output,mx,sm},
        [=](const auto& d,auto* bytes,auto** executor) {
            char layout[]="BSND";
            return aclnnSparseFlashAttentionGetWorkspaceSize(d[0],d[1],d[1],d[2],nullptr,nullptr,
                nullptr,d[3],d[4],1.0/std::sqrt(double(q.size(2))),1,layout,layout,0,
                std::numeric_limits<int64_t>::max(),std::numeric_limits<int64_t>::max(),2,true,
                d[5],d[6],d[7],bytes,executor);
        },aclnnSparseFlashAttention);
    return {out,mx,sm};
}
// Paged history remains owned by Past; only queries and selected IDs are dense.
Tensor paged_index(const Tensor& q,const Tensor& key,const Tensor& weight,
                   const Tensor& seq_q,const Tensor& seq_k,const Tensor& table,
                   int64_t topk) {
    check(q);
    TORCH_CHECK(q.dim()==4 && q.scalar_type()==at::kBFloat16 &&
                key.dim()==4 && key.scalar_type()==at::kBFloat16 &&
                key.size(2)==1 && key.size(3)==q.size(3),
                "paged_index expects BSND queries and PA_BSND keys");
    TORCH_CHECK(weight.dim()==3 && weight.size(0)==q.size(0) &&
                weight.size(1)==q.size(1) && weight.size(2)==q.size(2) &&
                (weight.scalar_type()==at::kFloat || weight.scalar_type()==at::kBFloat16),
                "paged_index weight shape/dtype mismatch");
    TORCH_CHECK(seq_q.dim()==1 && seq_k.dim()==1 &&
                seq_q.size(0)==q.size(0) && seq_k.size(0)==q.size(0) &&
                seq_q.scalar_type()==at::kInt && seq_k.scalar_type()==at::kInt &&
                table.dim()==2 && table.size(0)==q.size(0) &&
                table.scalar_type()==at::kInt && topk>0 && topk<=512,
                "paged_index lengths/table/topk mismatch");
    for(const auto& t: {q,key,weight,seq_q,seq_k,table})
        TORCH_CHECK(t.device()==q.device() && t.is_contiguous(),
                    "paged_index requires contiguous tensors on one NPU");
    c10::DeviceGuard guard(q.device());
    auto ids=at::empty({q.size(0),q.size(1),1,topk},q.options().dtype(at::kInt));
    auto values=at::empty({1,1,1,1},q.options());
    cann("ljq_paged_index",{q,key,weight,seq_q,seq_k,table,ids,values},
         [=](const auto& t,uint64_t* bytes,aclOpExecutor** executor) {
             char query_layout[]="BSND",key_layout[]="PA_BSND";
             auto unlimited=std::numeric_limits<int64_t>::max();
             return aclnnLightningIndexerGetWorkspaceSize(
                 t[0],t[1],t[2],t[3],t[4],t[5],query_layout,key_layout,
                 topk,3,unlimited,unlimited,false,t[6],t[7],bytes,executor);
         },aclnnLightningIndexer);
    return ids;
}

}
TORCH_LIBRARY_FRAGMENT(ljq_prefill,m) {
    m.def("bind_workspace(Tensor scratch) -> ()");
    m.def("unbind_workspace(Tensor scratch) -> ()");
    m.def("paged_index(Tensor query, Tensor key, Tensor weight, Tensor query_lengths, Tensor key_lengths, Tensor block_table, int topk) -> Tensor");
    m.def("flash_swa(Tensor query, Tensor local, Tensor sink) -> (Tensor, Tensor, Tensor)");
    m.def("flash_sparse(Tensor query, Tensor packed, Tensor indices) -> (Tensor, Tensor, Tensor)");
    m.def("grouped_int4(Tensor x, Tensor weight, Tensor scale, Tensor bias, Tensor token_scale, Tensor counts) -> Tensor");
    m.def("matmul_fp32(Tensor x, Tensor weight) -> Tensor");
    m.def("matmul_fp32.out(Tensor x, Tensor weight, Tensor(a!) out) -> Tensor(a!)");
    m.def("dynamic_quant(Tensor x) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(ljq_prefill,PrivateUse1,m) {
    m.impl("bind_workspace",bind_workspace);
    m.impl("unbind_workspace",unbind_workspace);
    m.impl("paged_index",paged_index);
    m.impl("flash_swa",flash_swa);m.impl("flash_sparse",flash_sparse);
    m.impl("grouped_int4",grouped_int4);
    m.impl("matmul_fp32",matmul_fp32);
    m.impl("matmul_fp32.out",matmul_fp32_out);
    m.impl("dynamic_quant",dynamic_quant);
}
