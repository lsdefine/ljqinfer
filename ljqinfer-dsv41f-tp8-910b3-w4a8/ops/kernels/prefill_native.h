#pragma once
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
namespace ljq {
using at::Tensor;
template<class F> void launch(const char* name, std::vector<Tensor> tensors, F f) {
    auto stream=c10_npu::getCurrentNPUStream();
    for (const auto& t:tensors)
        c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(),stream);
    at_npu::native::OpCommand::RunOpApiV2(name,[tensors=std::move(tensors),stream,f]() mutable -> int {
        return f(stream.stream(false)); // Never drain the task queue from its own callback.
    });
}
}
