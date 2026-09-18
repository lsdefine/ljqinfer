#include <aclnn/acl_meta.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/recurrent_gated_delta_rule/recurrent_gated_delta_rule_torch_adpt.h"
namespace vllm_ascend {
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
}
TORCH_LIBRARY(ljq_gdn, m) {
  m.def("recurrent(Tensor query, Tensor key, Tensor value, Tensor(a!) state, *, Tensor? beta=None, float? scale=None, Tensor? actual_seq_lengths=None, Tensor? ssm_state_indices=None, Tensor? num_accepted_tokens=None, Tensor? g=None, Tensor? gk=None) -> Tensor");
  m.impl("recurrent", c10::kPrivateUse1, &vllm_ascend::npu_recurrent_gated_delta_rule);
}
