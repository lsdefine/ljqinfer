"""NPU regression for the raw B1Q8 GDN convolution stride contract."""
import argparse
import torch
import torch_npu  # noqa: F401

from ops.kernels import K

p = argparse.ArgumentParser()
p.add_argument("--device", default="npu:4")
a = p.parse_args()
torch.npu.set_device(a.device)
torch.manual_seed(31)

base = torch.randn(1, 3, 2560, device=a.device, dtype=torch.bfloat16)
# Splitting the packed qkvz projection creates the production non-contiguous view.
packed = torch.randn(1, 8, 4096, device=a.device, dtype=torch.bfloat16)
x = packed[:, :, :2560]
weight = torch.randn(2560, 1, 4, device=a.device, dtype=torch.bfloat16)
weight_kc = weight.reshape(2560, 4).transpose(0, 1).contiguous()

native_pending = torch.empty_like(x)
native = K.causal_conv_decode(x, base, native_pending, weight_kc)
sequence = torch.cat((base, x), dim=1)
w = weight.reshape(2560, 4)
# Preserve the original Q8 fallback reduction order: the native kernel is
# expected to match this short-sequence contract bit-for-bit.
windows = sequence.unfold(1, 4, 1)
reference = (windows * w[None, None]).sum(dim=-1)
reference_pending = x
torch.npu.synchronize(a.device)

output_diff = float((native.float() - reference.float()).abs().max().cpu())
pending_diff = float(
    (native_pending.float() - reference_pending.float()).abs().max().cpu())
print({
    "input_stride": tuple(x.stride()),
    "input_contiguous": x.is_contiguous(),
    "output_max_diff": output_diff,
    "pending_max_diff": pending_diff,
})
if (x.is_contiguous() or tuple(x.stride()) != (32768, 4096, 1)
        or not native.is_contiguous()
        or output_diff != 0.0 or pending_diff != 0.0):
    raise RuntimeError("native GDN convolution stride regression")

contiguous_x = x.contiguous()
contiguous_pending = torch.empty_like(contiguous_x)
contiguous_output = K.causal_conv_decode(
    contiguous_x, base, contiguous_pending, weight_kc)
torch.npu.synchronize(a.device)
contiguous_output_diff = float(
    (contiguous_output.float() - reference.float()).abs().max().cpu())
contiguous_pending_diff = float(
    (contiguous_pending.float() - contiguous_x.float()).abs().max().cpu())
print({
    "contiguous_output_max_diff": contiguous_output_diff,
    "contiguous_pending_max_diff": contiguous_pending_diff,
})
if contiguous_output_diff != 0.0 or contiguous_pending_diff != 0.0:
    raise RuntimeError("contiguous GDN convolution regression")
