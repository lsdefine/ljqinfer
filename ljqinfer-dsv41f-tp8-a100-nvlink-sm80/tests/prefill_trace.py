"""Diagnostic hooks only; no changes to model arithmetic or persistent state."""
import torch


class TraceCall:
    def __init__(self, inner, name, records):
        self.inner, self.name, self.records = inner, name, records

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        if self.name == '0.attention':
            import model.prefill_layer as layer
            original = layer.attend
            def capture(**kw):
                for key in ('q', 'kv', 'sink'):
                    value = kw[key]
                    if key == 'q': value = value.flatten(-2)
                    self.records.setdefault('0.attend.'+key, []).append(value.detach().float().cpu())
                value = original(**kw)
                self.records.setdefault('0.attend.output', []).append(value.detach().flatten(-2).float().cpu())
                return value
            layer.attend = capture
            try:
                result = self.inner(*args, **kwargs)
            finally:
                layer.attend = original
        else:
            result = self.inner(*args, **kwargs)
        self.records.setdefault(self.name, []).append(result.detach().float().cpu())
        return result


class TraceLinear:
    def __init__(self, inner, records):
        self.inner, self.records = inner, records

    def __call__(self, name, x):
        result = self.inner(name, x)
        if name.startswith('layers.0.attn.'):
            self.records.setdefault(name+'.input', []).append(x.detach().float().cpu())
            self.records.setdefault(name+'.output', []).append(result.detach().float().cpu())
        return result


def attach(model):
    records = {}
    model.blocks[0].attention.linear = TraceLinear(model.blocks[0].attention.linear, records)
    for i, block in enumerate(model.blocks):
        for name in ('engram', 'attention', 'moe'):
            inner = getattr(block, name)
            if inner is not None:
                setattr(block, name, TraceCall(inner, f'{i}.{name}', records))
    return records


def compare(actual, reference):
    rows = []
    for name, values in reference.items():
        x, y = actual[name][0], values[0]
        if name == 'layers.0.attn.wo_b.output':
            continue  # Rank-local partial sum, not comparable before collective.
        if x.shape != y.shape:
            assert x.shape[:-1] == y.shape[:-1] and y.shape[-1] == x.shape[-1]*8
            y = y[..., :x.shape[-1]]
        x, y = x.flatten(), y.flatten()
        error = x-y
        rows.append(dict(name=name, max_abs=error.abs().max().item(),
                         rmse=error.square().mean().sqrt().item(),
                         relative_l2=(error.norm()/y.norm().clamp_min(1e-30)).item()))
    return rows
