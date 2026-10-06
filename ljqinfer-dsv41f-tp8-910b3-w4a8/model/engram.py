"""Released Engram hash semantics; metadata setup adapted from official engram.py."""
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from torch import nn


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max_ngram_size-gram), each split
    over `n_heads` heads. Every (n-gram size, head) pair owns its own prime-sized bucket range in the
    layer's table; the primes are drawn in order and never reused, which keeps the ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head] bucket modulus
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class EngramHash:
    """CPU hash metadata only. Request owns raw history; no slot/token cache."""
    def __init__(self, config, tokenizer):
        from types import SimpleNamespace
        args = SimpleNamespace(**config)
        self.layout = EngramLayout.from_args(args)
        if self.layout is None:
            raise ValueError('Engram configuration required')
        token_map, size = build_compressed_token_map(tokenizer)
        if size != args.engram_compressed_vocab_size:
            raise ValueError(f'compressed vocabulary mismatch: {size}')
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[args.engram_pad_id]
        self.primes = torch.tensor(self.layout.primes, dtype=torch.int64)
        flat = self.primes.flatten(1)
        self.offsets = flat.cumsum(-1) - flat
        if tuple(flat.sum(-1).tolist()) != self.layout.num_embeddings:
            raise ValueError('Engram table sizes do not match hash buckets')
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids,
            self.layout.max_ngram_size, size)

    def __call__(self, tokens, *, start, history_tokens=(), token_mask=None,
                 history_mask=None):
        n = self.layout.max_ngram_size
        if start < 0 or len(history_tokens) != min(start, n-1):
            raise ValueError('exact raw token history required before chunk')
        raw = torch.tensor(tuple(history_tokens)+tuple(tokens), dtype=torch.int64)
        if raw.numel() and (raw.min() < 0 or raw.max() >= len(self.token_map)):
            raise ValueError('token outside vocabulary')
        source = self.token_map[raw]
        def mask(values, count):
            result = torch.ones(count, dtype=torch.bool) if values is None else torch.as_tensor(values, dtype=torch.bool, device='cpu')
            if result.shape != (count,):
                raise ValueError('mask shape mismatch')
            return result
        live = torch.cat((mask(history_mask,len(history_tokens)), mask(token_mask,len(tokens))))
        source = torch.where(live, source, -1)
        pos = torch.arange(len(tokens)) + len(history_tokens)
        blocked = torch.zeros(len(tokens), dtype=torch.bool)
        lookback = []
        for shift in range(n):
            value = source[(pos-shift).clamp_min(0)]
            blocked |= (pos < shift) | (value == -1)
            lookback.append(torch.where(blocked, self.pad_id, value))
        products = torch.stack(lookback,-1)[:,None,:] * self.multipliers
        rolling, hashes = products[...,0], []
        for i in range(1,n):
            rolling = torch.bitwise_xor(rolling,products[...,i])
            hashes.append(rolling[...,None] % self.primes[:,i-1])
        return torch.cat(hashes,-1) + self.offsets


    def selected_ids(self, layer, rank, tokens, *, start, history_tokens=()):
        """Compute only this layer's selected hash columns; raw history stays explicit."""
        n = self.layout.max_ngram_size
        if start < 0 or len(history_tokens) != min(start, n - 1):
            raise ValueError('exact raw token history required before chunk')
        if rank is not None and (not 0 <= rank < 8):
            raise ValueError('invalid rank')
        raw = torch.tensor(tuple(history_tokens) + tuple(tokens), dtype=torch.int64)
        if raw.numel() and (raw.min() < 0 or raw.max() >= len(self.token_map)):
            raise ValueError('token outside vocabulary')
        source = self.token_map[raw]
        pos = torch.arange(len(tokens)) + len(history_tokens)
        blocked = torch.zeros(len(tokens), dtype=torch.bool)
        rolling = None
        parts = []
        heads = self.layout.n_heads
        low, high = (0, (n - 1) * heads) if rank is None else (rank * 3, (rank + 1) * 3)
        for shift in range(n):
            value = source[(pos - shift).clamp_min(0)]
            blocked |= (pos < shift) | (value == -1)
            prod = torch.where(blocked, self.pad_id, value) * self.multipliers[layer, shift]
            rolling = prod if rolling is None else torch.bitwise_xor(rolling, prod)
            if shift:
                a, b = (max(low, (shift - 1) * heads), min(high, shift * heads))
                if a < b:
                    primes = self.primes[layer, shift - 1, a - (shift - 1) * heads:b - (shift - 1) * heads]
                    parts.append(rolling[:, None] % primes + self.offsets[layer, a:b])
        return torch.cat(parts, -1)


class EngramRows:
    """Bind one layer's host table to hash->gather->dequantize; no hidden state."""
    def __init__(self, hasher, layer, table, *, rank=None):
        self.hasher, self.table, self.rank = hasher, table, rank
        self.layer = hasher.layout.layer_ids.index(layer)

    def __call__(self, slot, start, tokens, history_tokens=()):
        from ops.prefill.host_engram import gather
        ids = self.hasher.selected_ids(self.layer, self.rank, tokens,
            start=start, history_tokens=history_tokens)
        return gather(self.table.weight, self.table.scale, ids)
