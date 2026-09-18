"""Fixed-shape TP4 BxQ target-verify NPUGraph.

The graph aggregates all ``B*Q`` rows into each projection/GEMM.  Full
attention evaluates the Q-token causal chunk in one native tensor program;
only the mathematically recurrent GDN state update remains sequential over Q.
Speculative state is isolated in persistent pending buffers and
:meth:`commit` publishes exactly ``accepted_lengths`` rows per sequence.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch_npu

from .config import CONFIG
from .weights import FullAttentionWeights
from ops.kernels import K


class DecodeGraphRunner:
    """One resident graph for a fixed ``B`` and fixed query width ``Q``."""

    def __init__(self, engine, batch_size: int, max_prefix: int,
                 num_layers: int = CONFIG.num_hidden_layers,
                 query_width: int = 8):
        self.engine = engine
        self.rt = engine.collective
        self.weights = engine.weights
        self.device = engine.device
        self.batch_size = int(batch_size)
        self.query_width = int(query_width)
        self.max_prefix = int(max_prefix)
        self.num_layers = int(num_layers)
        if self.rt is None:
            raise ValueError("decode graph requires a TP collective")
        if self.batch_size not in (1, 2, 3, 4) or self.query_width != 8:
            raise ValueError(
                "decode graph native ABI requires batch_size in [1,4] and query_width=8")
        if self.max_prefix <= 0:
            raise ValueError("max_prefix must be positive")
        if self.query_width > self.max_prefix:
            raise ValueError("query_width cannot exceed max_prefix")
        if not 0 <= self.num_layers <= CONFIG.num_hidden_layers:
            raise ValueError("invalid num_layers")

        b, q, s = self.batch_size, self.query_width, self.max_prefix
        t = b * q
        self.ids = torch.zeros((b, q), dtype=torch.int64, device=self.device)
        self.positions = torch.arange(q, dtype=torch.int64, device=self.device)[None].expand(b, -1).clone()
        self.local_logits = torch.zeros(
            (b, q, CONFIG.vocab_size // CONFIG.tp), dtype=torch.bfloat16,
            device=self.device)
        # DFlash2 consumes target block outputs at these five 0-based layers.
        # Keep fixed-shape graph outputs resident without changing replay ABI.
        self.aux_layer_ids = tuple(
            i for i in (5, 19, 33, 47, 61) if i < self.num_layers)
        self.aux_hidden = torch.zeros(
            (len(self.aux_layer_ids), b, q, CONFIG.hidden_size),
            dtype=torch.bfloat16, device=self.device)
        self._aux_slot = {layer: slot for slot, layer in enumerate(self.aux_layer_ids)}

        ng = sum(i not in CONFIG.full_attention_layers
                 for i in range(self.num_layers))
        nf = sum(i in CONFIG.full_attention_layers
                 for i in range(self.num_layers))
        self.gdn_conv_in = torch.zeros(
            (ng, b, *CONFIG.gdn_conv_state_shape), dtype=torch.bfloat16,
            device=self.device)
        self.gdn_rec_in = torch.zeros(
            (ng, b, *CONFIG.gdn_recurrent_state_shape), dtype=torch.bfloat16,
            device=self.device)
        # Each candidate stores its projected input row. Commit reconstructs
        # the exact three-row convolution state from base + accepted rows.
        self.gdn_conv_pending = torch.empty(
            (ng, b, q, CONFIG.gdn_conv_state_shape[-1]),
            dtype=torch.bfloat16, device=self.device)
        self.gdn_rec_pending = torch.empty(
            (ng, b, q, *CONFIG.gdn_recurrent_state_shape), dtype=torch.bfloat16,
            device=self.device)

        kv_shape = (nf, b, s + q, CONFIG.local_kv_heads, CONFIG.head_dim)
        self.k_workspace = torch.zeros(
            kv_shape, dtype=torch.bfloat16, device=self.device)
        self.v_workspace = torch.zeros_like(self.k_workspace)
        # Committed S rows and speculative Q rows share contiguous storage.
        # Full Attention can consume the exact [history, candidate] layout
        # directly instead of materializing it with two cats in every layer.
        self.k_in = self.k_workspace[:, :, :s]
        self.v_in = self.v_workspace[:, :, :s]
        self.k_pending = self.k_workspace[:, :, s:]
        self.v_pending = self.v_workspace[:, :, s:]
        self._cache_index = torch.arange(s, dtype=torch.int64,
                                         device=self.device)[None, None, None, :]
        self._query_index = torch.arange(q, dtype=torch.int64,
                                         device=self.device)[None, None, :, None]
        self._batch_index = torch.arange(b, dtype=torch.int64,
                                         device=self.device)
        self._positions_host = torch.arange(q, dtype=torch.int64)[None].expand(b, -1).clone()
        # Static packed metadata for the native recurrent GDN.  Slot 0 of each
        # sequence is seeded with committed state; accepted=1 selects that base
        # and the operator overwrites all Q slots with candidate snapshots.
        self._gdn_actual_seq_lengths = torch.tensor(
            [0] + [q] * b, dtype=torch.int32, device=self.device)
        self._gdn_state_indices = torch.arange(
            q, dtype=torch.int32, device=self.device).repeat(b)
        self._gdn_base_slot = torch.ones(
            (b,), dtype=torch.int32, device=self.device)

        # Keep every captured weight alive and pre-load outside capture.
        self.layers = [self.weights.layer(i) for i in range(self.num_layers)]
        _ = self.weights.embedding, self.weights.final_norm, self.weights.lm_head
        # These GDN transforms depend only on immutable weights.  Materialize
        # them once on-device so every graph replay avoids four tiny kernels.
        self._gdn_decay = []
        self._gdn_dt_bias_f32 = []
        for layer in self.layers:
            if not isinstance(layer.attention, FullAttentionWeights):
                self._gdn_decay.append(-torch.exp(layer.attention.A_log.float()))
                self._gdn_dt_bias_f32.append(layer.attention.dt_bias.float())
        self.graph: Optional[object] = None
        self._pending = False
        self._t = t

    def _normalize_inputs(self, input_ids, positions):
        b, q = self.batch_size, self.query_width
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device="cpu")
        if ids.numel() != b * q:
            raise ValueError(f"fixed graph requires B={b}, Q={q}, got {ids.numel()} tokens")
        ids = ids.reshape(b, q).contiguous()

        pos = torch.as_tensor(positions, dtype=torch.int64, device="cpu")
        if pos.numel() == 1:
            starts = pos.reshape(1).expand(b)
            pos = starts[:, None] + torch.arange(q, dtype=torch.int64)[None]
        elif pos.numel() == b:
            starts = pos.reshape(b)
            pos = starts[:, None] + torch.arange(q, dtype=torch.int64)[None]
        elif pos.numel() == b * q:
            pos = pos.reshape(b, q).contiguous()
        else:
            raise ValueError(f"positions must be scalar, B={b} starts, or B*Q={b*q} rows")
        if q > 1 and bool((pos[:, 1:] != pos[:, :-1] + 1).any()):
            raise ValueError("each verify row must contain contiguous positions")
        position_limit = min(CONFIG.max_position_embeddings,
                             self.max_prefix + self.query_width)
        if bool(((pos < 0) | (pos >= position_limit)).any()):
            raise ValueError(f"positions must be in [0,{position_limit})")
        return ids, pos

    def prepare(self, input_ids, positions) -> None:
        if self._pending:
            raise RuntimeError("previous verify is pending; commit or rollback first")
        ids, pos = self._normalize_inputs(input_ids, positions)
        self._positions_host.copy_(pos)
        self.ids.copy_(ids, non_blocking=True)
        self.positions.copy_(pos, non_blocking=True)

    def prepare_draft(self, anchor_tokens, device_anchors, device_paths,
                      positions, active_rows=None) -> None:
        """Prepare verify IDs from resident draft outputs without a host round trip."""
        if self._pending:
            raise RuntimeError("previous verify is pending; commit or rollback first")
        anchors = [int(token) for token in anchor_tokens]
        if len(anchors) != self.batch_size:
            raise ValueError("anchor_tokens must match decode batch size")
        active = ([True] * self.batch_size if active_rows is None else
                  [bool(value) for value in active_rows])
        if len(active) != self.batch_size:
            raise ValueError("active_rows must match decode batch size")
        draft_anchors = torch.as_tensor(
            device_anchors, dtype=torch.int64, device=self.device).reshape(-1)
        draft_paths = torch.as_tensor(
            device_paths, dtype=torch.int64, device=self.device)
        if draft_anchors.numel() != self.batch_size:
            raise ValueError("device_anchors must contain one token per batch row")
        if draft_paths.numel() != self.batch_size * (self.query_width - 1):
            raise ValueError("device_paths must contain B*(Q-1) draft tokens")
        # Reuse the established CPU validation for positions only. The dummy IDs
        # never cross to the device; real IDs are copied D2D below.
        _, pos = self._normalize_inputs(
            torch.empty((self.batch_size, self.query_width), dtype=torch.int64),
            positions)
        self._positions_host.copy_(pos)
        self.positions.copy_(pos, non_blocking=True)
        self.ids[:, 0].copy_(draft_anchors, non_blocking=True)
        self.ids[:, 1:].copy_(
            draft_paths.reshape(self.batch_size, self.query_width - 1),
            non_blocking=True)
        for row, enabled in enumerate(active):
            if not enabled:
                self.ids[row].fill_(anchors[row])

    def _sequence_all_reduce(self, tensor):
        """Reduce all rows with one collective and exact small-message TP4 rounding."""
        if self.batch_size == 1:
            return self.engine.all_reduce(tensor)
        gathered = self.rt.all_gather(tensor)
        torch.add(gathered[0], gathered[1], out=tensor)
        tensor.add_(gathered[3])
        tensor.add_(gathered[2])
        return tensor

    def _sequence_qk_norm_rope(self, query, key, q_weight, k_weight,
                               frequencies):
        """Run native QK norm/RoPE once for the entire fixed batch."""
        if self.batch_size == 1:
            return K.qk_rms_norm_rope_decode(
                query, key, q_weight, k_weight, frequencies,
                CONFIG.rotary_dim, CONFIG.rms_norm_eps)
        return K.qk_rms_norm_rope_decode(
            query, key, q_weight, k_weight, frequencies,
            CONFIG.rotary_dim, CONFIG.rms_norm_eps)

    def _full_attention(self, hidden, w, slot: int, rope_frequencies,
                        atten_mask):
        b, q = self.batch_size, self.query_width
        if w.qkv is not None:
            packed = K.w8a8_linear(hidden, w.qkv)
            qw = CONFIG.local_q_heads * 2 * CONFIG.head_dim
            kw = CONFIG.local_kv_heads * CONFIG.head_dim
            q_gate, k, v = packed.split((qw, kw, kw), dim=-1)
        else:
            q_gate = K.w8a8_linear(hidden, w.q)
            k = K.w8a8_linear(hidden, w.k)
            v = K.w8a8_linear(hidden, w.v)
        q_gate = q_gate.reshape(b, q, CONFIG.local_q_heads, 2 * CONFIG.head_dim)
        k = k.reshape(b, q, CONFIG.local_kv_heads, CONFIG.head_dim)
        v = v.reshape_as(k)
        query, gate = q_gate.chunk(2, dim=-1)
        query, k = self._sequence_qk_norm_rope(
            query.reshape(self._t, CONFIG.local_q_heads, CONFIG.head_dim),
            k.reshape(self._t, CONFIG.local_kv_heads, CONFIG.head_dim),
            w.q_norm, w.k_norm, rope_frequencies)
        query = query.reshape(b, q, CONFIG.local_q_heads, CONFIG.head_dim)
        k = k.reshape(b, q, CONFIG.local_kv_heads, CONFIG.head_dim)

        self.k_pending[slot].copy_(k)
        self.v_pending[slot].copy_(v)
        y, _ = torch_npu.npu_fused_infer_attention_score(
            query, self.k_workspace[slot], self.v_workspace[slot],
            atten_mask=atten_mask,
            num_heads=CONFIG.local_q_heads,
            num_key_value_heads=CONFIG.local_kv_heads,
            scale=CONFIG.head_dim ** -0.5, input_layout="BSND", sparse_mode=0,
            inner_precise=0)
        y = y * torch.sigmoid(gate)
        y = K.w8a8_linear(y.reshape(self._t, -1), w.o)
        return self._sequence_all_reduce(y)


    def _gdn_attention(self, hidden, w, slot: int):
        """Project BxQ8 once and advance every row recurrent state natively."""
        b, q = self.batch_size, self.query_width
        if w.qkvz is not None:
            packed = K.w8a8_linear(hidden, w.qkvz)
            qkv_width = (2 * CONFIG.local_gdn_k_heads * CONFIG.linear_key_head_dim +
                         CONFIG.local_gdn_v_heads * CONFIG.linear_value_head_dim)
            qkv, z = packed.split((qkv_width, packed.shape[-1] - qkv_width), dim=-1)
        else:
            qkv = K.w8a8_linear(hidden, w.qkv)
            z = K.w8a8_linear(hidden, w.z)
        qkv, z = qkv.reshape(b, q, -1), z.reshape(b, q, -1)
        ab = K.bf16_linear(hidden, w.ab).reshape(b, q, -1)
        av, bv = ab.chunk(2, dim=-1)

        mixed = K.silu(K.causal_conv_decode(
            qkv, self.gdn_conv_in[slot], self.gdn_conv_pending[slot], w.conv_kc
        )).reshape(self._t, -1)

        qn = CONFIG.local_gdn_k_heads * CONFIG.linear_key_head_dim
        vn = CONFIG.local_gdn_v_heads * CONFIG.linear_value_head_dim
        qv, kv, vv = mixed.split((qn, qn, vn), dim=-1)
        qv, kv = K.paired_l2norm_decode(
            qv.reshape(self._t, CONFIG.local_gdn_k_heads,
                       CONFIG.linear_key_head_dim),
            kv.reshape(self._t, CONFIG.local_gdn_k_heads,
                       CONFIG.linear_key_head_dim))
        vv = vv.reshape(
            self._t, CONFIG.local_gdn_v_heads, CONFIG.linear_value_head_dim)
        decay = self._gdn_decay[slot]
        dt_bias = self._gdn_dt_bias_f32[slot]
        g = decay[None, None] * torch.nn.functional.softplus(
            av.float() + dt_bias[None, None])
        beta_logits = bv.to(vv.dtype).contiguous()

        pending = self.gdn_rec_pending[slot]
        base_state = self.gdn_rec_in[slot]
        # Native recurrent kernel reads the immutable base state for every B.
        y = K.gated_delta_decode(
            qv, kv, vv, g.reshape(self._t, -1), beta_logits.reshape(self._t, -1),
            pending.reshape(self._t, *CONFIG.gdn_recurrent_state_shape),
            self._gdn_actual_seq_lengths, self._gdn_state_indices,
            self._gdn_base_slot, base_state=base_state)
        y = K.rmsnorm_gated(
            y, z.reshape(self._t, CONFIG.local_gdn_v_heads,
                         CONFIG.linear_value_head_dim),
            w.norm, CONFIG.rms_norm_eps).reshape(self._t, vn)
        y = K.bf16_linear(y, w.out)
        return self._sequence_all_reduce(y)

    def _forward(self) -> None:
        hidden = K.embedding(self.ids.reshape(-1), self.weights.embedding)
        if not self.layers:
            normed = K.rms_norm(hidden, self.weights.final_norm,
                                CONFIG.rms_norm_eps)
            self.local_logits.copy_(
                self.engine.local_logits(normed).reshape_as(self.local_logits))
            return

        normed = K.rms_norm(hidden, self.layers[0].input_norm,
                            CONFIG.rms_norm_eps)
        rope_frequencies = atten_mask = None
        if self.k_in.shape[0]:
            rope_frequencies = K.rope_frequencies(
                self.positions.reshape(-1), CONFIG.rotary_dim,
                CONFIG.rope_theta, normed.dtype)
            history_visible = self._cache_index < self.positions[:, None, :, None]
            candidate_visible = self._query_index.transpose(2, 3) <= self._query_index
            atten_mask = ~torch.cat((
                history_visible.expand(
                    self.batch_size, 1, self.query_width, self.max_prefix),
                candidate_visible.expand(
                    self.batch_size, 1, self.query_width, self.query_width)),
                dim=-1)
        gs = fs = 0
        for idx, w in enumerate(self.layers):
            if isinstance(w.attention, FullAttentionWeights):
                attn = self._full_attention(
                    normed, w.attention, fs, rope_frequencies, atten_mask)
                fs += 1
            else:
                attn = self._gdn_attention(normed, w.attention, gs)
                gs += 1
            normed, hidden = K.add_rms_norm(
                attn, hidden, w.post_norm, CONFIG.rms_norm_eps)
            mlp = K.swiglu_mlp(
                normed, w.mlp, all_reduce=self._sequence_all_reduce)
            if idx + 1 < len(self.layers):
                normed, hidden = K.add_rms_norm(
                    mlp, hidden, self.layers[idx + 1].input_norm,
                    CONFIG.rms_norm_eps)
            else:
                normed, hidden = K.add_rms_norm(
                    mlp, hidden, self.weights.final_norm,
                    CONFIG.rms_norm_eps)
            aux_slot = self._aux_slot.get(idx)
            if aux_slot is not None:
                self.aux_hidden[aux_slot].copy_(
                    hidden.reshape(self.batch_size, self.query_width, -1))
        self.local_logits.copy_(
            self.engine.local_logits(normed).reshape_as(self.local_logits))

    def _device_sync(self) -> None:
        # Capture switches to an internal stream; synchronize the device rather
        # than the runtime's cached default stream.
        torch.npu.synchronize(self.device)

    def warm(self) -> None:
        self._forward()
        self._device_sync()

    def capture(self, pool=None, warm: bool = True) -> None:
        if self.graph is not None:
            return
        if warm:
            self.warm()
        graph = torch_npu.npu.NPUGraph()
        try:
            with torch_npu.npu.graph(graph, pool=pool):
                self._forward()
        except Exception:
            try:
                graph.reset()
            except Exception:
                pass
            raise
        self._device_sync()
        graph.replay()
        self._device_sync()
        self.graph = graph

    def replay(self, synchronize: bool = True) -> None:
        if self.graph is None:
            raise RuntimeError("capture the graph before replay")
        if self._pending:
            raise RuntimeError("previous verify is pending; commit or rollback first")
        self.graph.replay()
        if synchronize:
            self._device_sync()
        self._pending = True

    def run_eager(self, synchronize: bool = True) -> None:
        """Execute the identical fixed-shape body without graph replay."""
        if self._pending:
            raise RuntimeError("previous verify is pending; commit or rollback first")
        self._forward()
        if synchronize:
            self._device_sync()
        self._pending = True

    def _accepted(self, accepted_lengths):
        if accepted_lengths is None:
            values = [self.query_width] * self.batch_size
        elif isinstance(accepted_lengths, int):
            values = [int(accepted_lengths)] * self.batch_size
        else:
            values = [int(x) for x in accepted_lengths]
        if len(values) != self.batch_size:
            raise ValueError("accepted_lengths must have one value per batch row")
        if any(x < 0 or x > self.query_width for x in values):
            raise ValueError(f"accepted lengths must be in [0,{self.query_width}]")
        return values

    def commit(self, accepted_lengths=None) -> None:
        """Publish only accepted speculative state; rejected rows stay private."""
        if not self._pending:
            raise RuntimeError("no pending verify to commit")
        accepted = self._accepted(accepted_lengths)
        for row, count in enumerate(accepted):
            if count == 0:
                continue
            if self.gdn_conv_in.numel():
                sequence = torch.cat((self.gdn_conv_in[:, row],
                                      self.gdn_conv_pending[:, row, :count]), dim=1)
                self.gdn_conv_in[:, row].copy_(
                    sequence[:, -CONFIG.gdn_conv_state_shape[0]:])
                self.gdn_rec_in[:, row].copy_(self.gdn_rec_pending[:, row, count - 1])
            start = int(self._positions_host[row, 0])
            end = start + count
            if end > self.max_prefix:
                raise RuntimeError(
                    f"commit end {end} exceeds context capacity {self.max_prefix}")
            if self.k_in.numel():
                self.k_in[:, row, start:end].copy_(self.k_pending[:, row, :count])
                self.v_in[:, row, start:end].copy_(self.v_pending[:, row, :count])
        self._device_sync()
        self._pending = False

    def rollback(self) -> None:
        """Discard all pending candidates without touching committed state."""
        if not self._pending:
            raise RuntimeError("no pending verify to roll back")
        self._pending = False

    def reset(self) -> None:
        if self._pending:
            self.rollback()
        for x in (self.gdn_conv_in, self.gdn_rec_in, self.k_in, self.v_in):
            x.zero_()
        self._device_sync()

    def close(self) -> None:
        """Destroy an epoch-owned graph and drop every device allocation.

        Startup-resident runners are never closed.  Dynamic-capacity runners
        call this exactly once after their epoch has released all sequence
        pages, so a completed batch cannot keep graph-private KV arenas alive.
        """
        if getattr(self, "_closed", False):
            return
        if self._pending:
            self.rollback()
        self._device_sync()
        graph = self.graph
        self.graph = None
        reset_error = None
        try:
            if graph is not None:
                graph.reset()
            self._device_sync()
        except BaseException as exc:
            reset_error = exc
        finally:
            # Direct inputs/workspaces dominate graph HBM. The two lists hold
            # transformed device weights and must be detached explicitly too.
            for name, value in tuple(vars(self).items()):
                if isinstance(value, torch.Tensor):
                    setattr(self, name, None)
            self._gdn_decay = []
            self._gdn_dt_bias_f32 = []
            self._closed = True
        if reset_error is not None:
            raise reset_error
