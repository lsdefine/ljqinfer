"""Independent per-query attention formula for bounded replay tests.
Only attention aggregation/masking is independent; projections and blocks are
shared with the subject. This is NOT a full independent model oracle.
"""
import torch


def sparse_reference(q, local_kv, local_start, global_kv, selected, positions,
                     sink, *, window, ratio, scale, **unused):
    outputs=[]
    for row, position in enumerate(positions.tolist()):
        lo=max(local_start, position-window+1, 0)
        keys=local_kv[lo-local_start:position-local_start+1]
        if selected is not None:
            ids=[i for i in selected[row].tolist()
                 if 0 <= i < min(len(global_kv), (position+1)//ratio)]
            if ids:
                keys=torch.cat((keys,global_kv[ids]))
        logits=(q[row].float() @ keys.float().T)*scale
        # An attention sink contributes only to the normalization denominator.
        maximum=torch.maximum(logits.max(-1).values,sink.float())
        numerator=torch.exp(logits-maximum[:,None])
        denominator=numerator.sum(-1)+torch.exp(sink.float()-maximum)
        outputs.append((numerator @ keys.float())/denominator[:,None])
    return torch.stack(outputs).to(q.dtype)
