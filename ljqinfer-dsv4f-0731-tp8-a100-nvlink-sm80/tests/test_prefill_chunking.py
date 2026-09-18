from model.model_api import _prefill_chunk_size


def chunks(total: int, limit: int = 12 * 1024, pos: int = 0) -> list[int]:
    out = []
    while pos < total:
        n = _prefill_chunk_size(pos, total, limit)
        out.append(n)
        pos += n
    return out


def test_short_prompts_remain_single_chunk_until_two_full_blocks():
    assert chunks(127) == [127]
    assert chunks(128) == [128]
    assert chunks(129) == [129]
    assert chunks(255) == [255]


def test_final_chunk_matches_cold_kv_replay_shape():
    assert chunks(256) == [128, 128]
    assert chunks(257) == [128, 129]
    assert chunks(384) == [256, 128]
    assert chunks(385) == [256, 129]
    assert chunks(257, pos=128) == [129]
    assert chunks(384, pos=256) == [128]


def test_large_prefixes_keep_large_aligned_chunks():
    assert chunks(12 * 1024 + 128) == [12 * 1024, 128]
    assert chunks(12 * 1024 + 1) == [12 * 1024 - 128, 129]
    assert chunks(640, limit=256) == [256, 256, 128]
