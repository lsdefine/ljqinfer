"""Bind canonical weights to the DSpark draft head, independently of decode.

The drafter is deliberately buildable on its own: it needs the main model's
target-layer hidden state as a tensor, not the main model itself, so latency
for the B1Q6 draft can be measured before the decode trunk exists.
"""
import torch
import torch.nn.functional as F
from model.dspark import DSparkAttention, DSparkBlock, DSparkDrafter, DSparkMoE
from model.prefill_config import validate_config, rotary_frequencies
from ops.prefill.gemm import PrefillLinear
from ops.decode.argmax import VocabShard
from ops.prefill.projection_workspace import ProjectionWorkspace


def build_drafter(config, weights, *, device, parallel=None, routed=None,
                  embed=None, head=None, markov=None, max_position=4096,
                  batch=1):
    """Assemble the three MTP stages over rank-local weights.

    ``parallel`` is the same PrefillParallel the trunk uses; ``routed`` lets a
    caller supply the expert path (a 5-row draft has no workspace to amortise,
    so the default is the dense one-expert-at-a-time loop).
    """
    validate_config(config)
    c = config
    world = 1 if parallel is None else parallel.world
    rank = None if parallel is None else parallel.rank
    reduce_sum = None if parallel is None else parallel
    if world not in (1, 8):
        raise ValueError('only unsharded or canonical EP8/TP8 weights')
    # Only the built-in heads emit a bare vocabulary slice; an injected head
    # is assumed to hand back a full row, so it must not be told it is sharded.
    sharded_heads = head is None and markov is None
    lin = PrefillLinear(weights)
    # The packed baseline dequantises a whole weight per call; the workspace
    # extension consumes FP4 directly and keeps the launch count flat.
    width = c['dim'] * len(c['dspark_target_layer_ids'])
    # A batched round drafts every request in one pass, so the row scratch
    # has to cover the whole batch's block, not one request's.
    rows = max(32, batch * (c['dspark_block_size'] + 2))
    lin.projection_workspace = ProjectionWorkspace(rows, width, device)
    # Draft rows sit just past the accepted token, so the rotary table has to
    # span the whole sequence, not just the block.
    span = max(int(max_position), c['dspark_block_size'] + 1)
    freqs = rotary_frequencies(c, 0, span, device=device)

    def _gathered(name):
        """Whole vocabulary table on every rank, assembled once at build time.

        A draft window looks up a handful of rows; the sharded lookup pays an
        all-reduce over [rows, dim] per call, which dominates a 5-row block.
        """
        local = weights[name]
        if world == 1:
            return local
        full = torch.empty((c['vocab_size'], local.shape[1]),
                           device=local.device, dtype=local.dtype)
        parallel.gather_rows(local, full)
        return full

    if embed is None:
        embed_full = _gathered('embed.weight')
        def embed(tokens):
            ids = torch.as_tensor(tokens, device=device, dtype=torch.long)
            return F.embedding(ids, embed_full)
    if head is None:
        def head(hidden):
            # Casting the vocabulary matrix per call would copy hundreds of
            # megabytes: match its dtype instead and widen only the logits.
            w = weights['head.weight']
            # The slice is left sharded: its only readers are the Markov bias
            # add and the sampler, and both now reduce two floats per row
            # instead of gathering 3MB of vocabulary per draft block.
            return F.linear(hidden.to(w.dtype), w).float()
    if markov is None:
        m = f"mtp.{c['n_mtp_layers'] - 1}.markov_head"
        # The chain refines block_size tokens one after another, so a sharded
        # lookup would cost one all-reduce per link. The table is 66 MB: gather
        # it once at build time and read it locally.
        e_full = _gathered(m + '.embed.weight')
        def markov(ids):
            # The bias half stays vocabulary-sharded: it is added to a logit
            # slice of the same shape, and the sampler is what agrees the
            # chain across ranks.
            e = F.embedding(ids, e_full)
            hw = weights[m + '.head.weight']
            return F.linear(e.to(hw.dtype), hw).float(), e

    blocks = []
    for stage in range(c['n_mtp_layers']):
        attention = DSparkAttention(stage, c, weights, lin, freqs,
                                    world=world, reduce_sum=reduce_sum)
        moe = DSparkMoE(stage, c, weights, lin, _routed(stage, c, weights, lin, routed, parallel, batch),
                        reduce_sum=reduce_sum)
        blocks.append(DSparkBlock(stage, c, weights, attention, moe))
    shard = None
    if parallel is not None and sharded_heads:
        shard = VocabShard(rank * weights['head.weight'].shape[0],
                           c['vocab_size'], parallel)
    return DSparkDrafter(c, weights, lin, blocks, embed=embed, head=head,
                         markov=markov, vocab_shard=shard)


def _routed(stage, c, weights, lin, routed, parallel, batch=1):
    """Expert path for one draft stage.

    A draft block is five rows wide, but the bank is 128 fp4 experts: looping
    over them in python costs far more than the rows save, so the default is
    the same grouped workspace the trunk uses, sized to the block.
    """
    if routed is not None:
        return routed(stage) if callable(routed) else routed
    if parallel is None:
        from model.prefill_block import DenseRouted
        return DenseRouted(f'mtp.{stage}.ffn', c['dspark_n_routed_experts'], lin,
                           c['swiglu_limit'])
    from ops.prefill.moe_workspace import WorkspaceRouted
    draft = dict(c, n_routed_experts=c['dspark_n_routed_experts'],
                 n_activated_experts=c['dspark_n_activated_experts'])
    routed = WorkspaceRouted(f'mtp.{stage}.ffn', draft, weights, parallel,
                             batch * (c['dspark_block_size'] + 2))
    # A draft block is at most six rows: the tiled CUTLASS path would first
    # unpack the whole 128-expert bank (18 dequant launches per draft), so
    # take the fused FP4 GEMV that decode_build already pins for the trunk.
    routed.fused = True
    routed.decode_only = True
    return routed
