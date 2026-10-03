"""Bind canonical weights to the speculative decode model, independently of prefill.

Same checkpoint, different pass: a decode step scores one Q-row window and
publishes nothing until the accept length is known, so its attention comes from
model.decode_layer. Workspaces are sized by the window rather than by a prefill
chunk, and every layer takes the fused FP4 MoE path because a verify window is
always short. Caller owns weight loading, the process group, Engram tables and
Past, exactly as in model.prefill_build.
"""
import torch
import torch.nn.functional as F
from model.decode import DecodeModel
from model.decode_layer import DecodeAttention
from ops.decode.route_gate import route as decode_route
from model.engram import EngramRows
from model.prefill_block import PrefillBlock, PrefillEngram, PrefillMoE, DenseRouted
from model.prefill_config import validate_config, rotary_frequencies
from ops.prefill.gemm import PrefillLinear


def build_decode(config, weights, hasher, host_tables, *, device, window,
                 parallel=None, embed=None, head=None, max_position=None,
                 batch=1):
    """Full released model wired for decode; weights unsharded (1) or TP8 (8).

    `window` is the verify width Q, the row count every workspace is cut for.
    `max_position` must cover the furthest position the slot can reach, since
    the rotary table is indexed by absolute position, not by window offset.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    validate_config(config)
    c = config
    world = 1 if parallel is None else parallel.world
    rank = None if parallel is None else parallel.rank
    reduce_sum = None if parallel is None else parallel
    if world not in (1, 8):
        raise ValueError('only unsharded or canonical EP8/TP8 weights')
    if set(host_tables) != set(c['engram_layer_ids']):
        raise ValueError('every configured host Engram table is required')
    if int(window) < 1:
        raise ValueError('a verify window needs at least one row')
    window = int(window)
    if int(batch) < 1:
        raise ValueError('a batch needs at least one request')
    # Batched decode flattens B verify windows into one stream, so every
    # row-sized workspace is cut for B*Q rows.  Attention keeps the per-row
    # window; only the row count grows.
    row_cap = window * int(batch)
    lin = PrefillLinear(weights)
    moe_buffer = None
    lin_side = None
    side_stream = None
    if parallel is not None:
        from ops.prefill.projection_workspace import ProjectionWorkspace
        from model.prefill_block import MoEReduceBuffer
        lin.projection_workspace = ProjectionWorkspace(row_cap, c['dim'], device)
        moe_buffer = MoEReduceBuffer(row_cap, c['dim'], device)
        # The shared expert runs on a side stream so it overlaps the routed
        # experts (see PrefillMoE).  Its projection scratch must be a separate
        # workspace constructed on that stream: the workspace refuses a foreign
        # stream precisely because its scratch is serialised, and two streams
        # sharing one scratch would race.
        side_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(side_stream):
            lin_side = PrefillLinear(weights)
            lin_side.projection_workspace = ProjectionWorkspace(row_cap, c['dim'], device)
    # The mHC gates do not feed the collapse that attention/MoE waits on, so
    # they get their own stream; sharing the shared-expert stream would only
    # queue them behind it.
    hc_stream = torch.cuda.Stream(device=device)
    span = max(int(max_position or window), window)
    fs = {b: rotary_frequencies(c, 2 if b else 0, span, device=device) for b in (False, True)}
    if embed is None:
        def embed(tokens):
            w = weights['embed.weight']
            ids = torch.as_tensor(tokens, device=device, dtype=torch.long)
            # The bounds check reads device data back to the host, which
            # is illegal mid capture; a replayed graph re-runs the same
            # validated ids, so the eager pass is where it has to hold.
            if not torch.cuda.is_current_stream_capturing():
                if (ids < 0).any() or (ids >= c['vocab_size']).any():
                    raise ValueError('token outside vocabulary')
            if world == 1:
                return F.embedding(ids, w)
            n = c['vocab_size'] // world
            local = ids - rank * n
            mask = (local < 0) | (local >= n)
            out = F.embedding(local.masked_fill(mask, 0), w).masked_fill(mask[:, None], 0)
            reduce_sum(out)
            return out
    if head is None:
        # Casting the vocabulary matrix to FP32 doubles the 165 MB weight read
        # for no information gain: BF16 GEMM already accumulates in FP32, so
        # match the stored dtype and widen only the logits.
        head_w = weights['head.weight']
        def head(hidden):
            logits = F.linear(hidden.to(head_w.dtype), head_w).float()
            return logits if parallel is None else parallel.logits(logits)
    expand_workspace = None
    if parallel is not None:
        expand_workspace = tuple(torch.empty((row_cap, 4, c['dim']),
            device=device, dtype=torch.bfloat16) for _ in range(2))
    routed_workspace = None
    engram_workspace = None
    blocks = []
    for layer in range(c['n_layers']):
        p = f'layers.{layer}'
        attention = DecodeAttention(layer, c, weights, lin,
            fs[bool(c['compress_ratios'][layer])], world=world, reduce_sum=reduce_sum)
        if parallel is None:
            routed = DenseRouted(p + '.ffn', c['n_routed_experts'], lin, c['swiglu_limit'])
        elif routed_workspace is None:
            from ops.prefill.moe_workspace import WorkspaceRouted
            routed_workspace = WorkspaceRouted(p + '.ffn', c, weights, parallel, row_cap)
            # Unlike prefill there is no chunk-length branch: a window is short,
            # so the fused FP4 path is the only one decode ever takes.
            routed_workspace.fused = True
            routed_workspace.decode_only = True
            routed = routed_workspace
        else:
            routed = routed_workspace.bind(p + '.ffn', fused=True)
        moe = PrefillMoE(layer, c, weights, lin, routed,
                         reduce_shared=reduce_sum, buffer=moe_buffer,
                         route=decode_route, side=side_stream,
                         side_linear=lin_side)
        engram = None
        if layer in host_tables:
            if parallel is not None:
                # One workspace per Engram layer: its pinned host buffers are
                # the source of a captured H2D, so layers that share a
                # workspace overwrite each other's rows during capture and a
                # replay hands every layer the last writer's gather.
                from ops.prefill.engram_rows import RowWorkspace
                engram_workspace = RowWorkspace(row_cap, 3, device)
            rows = EngramRows(hasher, layer, host_tables[layer], rank=rank,
                              workspace=engram_workspace)
            engram = PrefillEngram(layer, weights, lin, rows,
                                   eps=c['norm_eps'], reduce_sum=reduce_sum,
                                   row_cap=row_cap)
        blocks.append(PrefillBlock(layer, c, weights, attention, moe, engram=engram,
            expand_workspace=expand_workspace, decode=True, side=hc_stream))
    return DecodeModel(c, blocks, embed, head, weights['norm.weight'],
                       target_layers=c['dspark_target_layer_ids'])
