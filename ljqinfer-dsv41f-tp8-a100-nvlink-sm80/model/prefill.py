"""Text backbone prefill only. Scheduling, sampling and decode live elsewhere."""
from dataclasses import dataclass
import torch
from ops.prefill import residual as r
from model.prefill_attention import PrefillScratch
from model.prefill_config import validate_config
from model.past import default_layer_views


@dataclass
class PrefillOutput:
    logits: torch.Tensor | None
    main_hidden: torch.Tensor | None


class PrefillModel:
    """Bound components contain weights, never private KV caches.

    embed(tokens) and head(hidden) return TP-reduced embeddings and full logits
    respectively; callers binding TP must explicitly supply those collectives.
    Engram rows must be supplied for configured layers, not silently skipped.
    forward encodes new tokens; replay reconstructs a bounded existing tail.
    """
    def __init__(self, config, blocks, embed, head, norm, *, target_layers=()):
        validate_config(config)
        self.c, self.blocks = dict(config), tuple(blocks)
        self.embed, self.head, self.norm = embed, head, norm
        self.targets = frozenset(target_layers)
        self.timing = None
        if [b.layer for b in self.blocks] != list(range(config['n_layers'])):
            raise ValueError('prefill requires every backbone layer in order')
        if not self.targets <= set(range(config['n_layers'])):
            raise ValueError('invalid target layer')
        for b in self.blocks:
            if (b.layer in config['engram_layer_ids']) != (b.engram is not None):
                raise ValueError('missing Engram row provider')

    def _check_past(self, past):
        if past.window != self.c['window_size'] or past.views != default_layer_views():
            raise ValueError('Past does not match released layer connections')
        for source in past.sources.values():
            if source.ckv_pool.data.shape[-1] != self.c['head_dim'] or source.index_pool.data.shape[-1] != self.c['index_head_dim']:
                raise ValueError('Past dimensions do not match model')

    def _embed(self, tokens, spans=(), start=0):
        x = self.embed(tokens)
        for offset, features in spans:
            lo, hi = max(start, offset), min(start+len(tokens), offset+len(features))
            if lo < hi:
                x[lo-start:hi-start] = features[lo-offset:hi-offset]
        if x.ndim != 2 or x.shape != (len(tokens), self.c['dim']):
            raise ValueError('embedding must return [tokens, configured dim]')
        h = x[:, None, :].expand(-1, self.c['hc_mult'], -1).clone()
        pre = torch.zeros(h.shape[:2], device=h.device, dtype=torch.float32)
        pre[:, 0] = 1.
        return h, pre

    def _retain_encoder(self, past, slot, start, h, pre, targets, *, replace=False):
        end = start + len(h)
        values = dict(h=h, pre=pre, features=torch.cat(targets, -1) if targets else h.new_empty((len(h), 0)))
        old = past.prefill_tails.get(slot)
        if not replace and start and (old is None or old['end'] != start):
            raise RuntimeError('missing contiguous encoder tail; restore before appending')
        if not replace and old is not None and start:
            values = {k: torch.cat((old[k], v), 0) for k, v in values.items()}
        past.prefill_tails[slot] = dict(end=end, **{
            k: v[-self.c['window_size']:].clone() for k, v in values.items()})

    @staticmethod
    def _image_mask(spans, start, length, device):
        if not spans:
            return None
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        for offset, features in spans:
            lo, hi = max(start, offset), min(start+length, offset+len(features))
            if lo < hi:
                mask[lo-start:hi-start] = True
        return mask

    @torch.inference_mode()
    def finish_prefill(self, *, past, slot):
        """Paper 3.2.2: run decoder on retained encoder outputs, not tokens.

        Truncate decoder SWA to the last window. Global KV is read-only.
        Does not recompute the encoder, allocate slots or commit positions.
        """
        self._check_past(past)
        tail = past.prefill_tails.get(slot)
        if slot in past.free_slots or slot in past.replay_pending or tail is None or tail['end'] != past.pos[slot]:
            raise ValueError('finish requires a positioned hot encoder tail')
        h, pre = tail['h'], tail['pre']
        start = tail['end'] - len(h)
        image_mask = self._image_mask(past.image_spans.get(slot, ()), start, len(h), h.device)
        scratch = PrefillScratch(read_global_only=True, replay_start=start)
        targets = [tail['features']] if tail['features'].shape[-1] else []
        if self.timing is not None:
            self.timing.mark('ced_layers_20_39')
        for block in self.blocks[20:]:
            if block.layer in self.targets:
                targets.append(h.mean(dim=-2))
            h, pre = block(h, pre, past=past, slot=slot, start=start, scratch=scratch, image_mask=image_mask)
        if self.timing is not None:
            self.timing.mark('norm_head')
        hidden = r.collapse_norm(h, pre, self.norm, self.c['norm_eps'])
        output = PrefillOutput(self.head(hidden[-1:]), torch.cat(targets, -1) if targets else None)
        if self.timing is not None:
            self.timing.mark('end')
        return output

    @torch.inference_mode()
    def replay(self, tokens, *, past, slot, start, history_tokens):
        """Reconstruct the bounded tail, never append global KV or commit position.

        Separate from forward: used after cold restore or explicitly before
        generation. SWA is truncated at start as in paper section 3.2.2.
        """
        self._check_past(past)
        end = past.pos[slot]
        if slot in past.free_slots or start != max(0, end-self.c['window_size']):
            raise ValueError('replay requires the complete bounded prefix tail')
        if not len(tokens) or start+len(tokens) != end:
            raise ValueError('replay tail must end at the committed position')
        if len(history_tokens) != min(start, self.c['engram_max_ngram_size']-1):
            raise ValueError('missing Engram history before replay tail')
        spans = past.image_spans.get(slot, ())
        h, pre = self._embed(tokens, spans, start)
        image_mask = self._image_mask(spans, start, len(tokens), h.device)
        scratch = PrefillScratch(read_global_only=True, replay_start=start)
        targets = []
        for block in self.blocks[:20]:
            h = block.prepare(h, slot=slot, start=start, tokens=tokens,
                              history_tokens=history_tokens, image_mask=image_mask)
            if block.layer in self.targets:
                targets.append(h.mean(dim=-2))
            h, pre = block(h, pre, past=past, slot=slot, start=start, scratch=scratch, image_mask=image_mask)
        # Decoder layers 20.. are deliberately not replayed: their sliding windows
        # are rebuilt by finish_prefill from this retained encoder tail, exactly as
        # on the cold path, where those layers never run during chunked prefill.
        self._retain_encoder(past, slot, start, h, pre, targets, replace=True)
        return PrefillOutput(None, None)

    @torch.inference_mode()
    def forward(self, tokens, *, past, slot, start, history_tokens=()):
        self._check_past(past)
        if slot in past.free_slots or slot in past.replay_pending or past.pos[slot] != start:
            raise ValueError('prefill requires a positioned hot slot')
        if len(tokens) == 0:
            raise ValueError('empty prefill chunk')
        if len(history_tokens) != min(start, self.c['engram_max_ngram_size']-1):
            raise ValueError('missing Engram history before prefill chunk')
        if self.timing is not None:
            self.timing.mark('embed')
        spans = past.image_spans.get(slot, ())
        h, pre = self._embed(tokens, spans, start)
        image_mask = self._image_mask(spans, start, len(tokens), h.device)
        scratch = PrefillScratch()
        targets = []
        for block in self.blocks[:20]:
            if block.engram is not None:
                block.engram.prefetch(slot=slot, start=start, tokens=tokens,
                                      history_tokens=history_tokens)
        for block in self.blocks[:20]:
            if self.timing is not None:
                self.timing.mark(f'layer_{block.layer}')
            h = block.prepare(h, slot=slot, start=start, tokens=tokens,
                              history_tokens=history_tokens, image_mask=image_mask)
            if block.layer in self.targets:
                targets.append(h.mean(dim=-2))
            h, pre = block(h, pre, past=past, slot=slot, start=start, scratch=scratch, image_mask=image_mask)
        # Layer 20's global branch consumes exactly its ordinary attention input,
        # but no decoder query/window/FFN or head is evaluated during prefill.
        if self.timing is not None:
            self.timing.mark('global_kv_and_tail')
        block = self.blocks[20]
        x = r.collapse_norm(h, pre, block.w['layers.20.attn_norm.weight'], self.c['norm_eps'])
        ck, ik = block.attention.project_global(x, past=past, slot=slot, start=start)
        source = past.sources[20]
        if ck is not None:
            source.write_ckv(slot, start, ck)
            source.write_index_k(slot, start, ik)
        self._retain_encoder(past, slot, start, h, pre, targets)
        if self.timing is not None:
            self.timing.mark('end')
        return PrefillOutput(None, None)
