# Native image input — usable checkpoint (2026-09-30)

## Usage
The running TP8 decode worker (`strategy.decode_worker`, under torchrun) serves
localhost RPC 62001; `server.server` exposes HTTP 8000. Keep the existing model,
rank and environment configuration; do not start a second worker on occupied GPUs.
`/health` alone is not an inference test. Replace the development API key before
exposing the frontend. No remote image fetch or separate vision service is used.

```python
import base64, os, requests
from pathlib import Path
image = base64.b64encode(Path("picture.png").read_bytes()).decode()
body = {
    "model": "dsv41", "temperature": 0, "reasoning_effort": "off",
    "max_tokens": 64, "stream": True,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What color is the square?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
    ]}],
}
with requests.post("http://127.0.0.1:8000/v1/chat/completions",
                   headers={"Authorization": "Bearer " + os.environ["API_KEY"]},
                   json=body, stream=True, timeout=(5, 180)) as response:
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line:
            print(line)
```
Repeat image/text entries for multiple images. For multi-turn requests retain
prior user messages with their original image entries and assistant text in
`messages`; text/image order is preserved. Plain text messages remain supported.

Limits: static PNG/JPEG/WEBP/BMP only; no remote URLs, local paths, animation or
video. At most 8 images, 16 MiB decoded file bytes/image, 32 MiB/request, 40 million
source pixels/image and 8192 expanded vision tokens total. Bad input is rejected,
not silently dropped. Vision grids are resized/planned by the model policy.

## Implementation
Symmetric TP8 native vision (head-parallel attention and sharded MLP), with patch
broadcast and replicated aligned outputs. Absolute image embedding spans, visual
MoE routing bias, Engram current/history masks and CED tail-window masks preserve
image positions. Image-containing requests bypass cold KV caching; text-only
requests do not execute image collectives. Existing text cache algorithms were
not rewritten. Preparation failures vote across ranks; a 600-second image OPEN
watchdog exits a stuck rank for torchrun to tear down the group.

## Verified scope
- CPU regression: 215 passed / 16 pre-existing failures / 78 skipped; no new
  failures versus baseline 5b2379d. Skips are not GPU validation.
- Four independent leaf numeric tests (official MoE gate and Engram hash);
  state/slot cleanup, span/tail masking and image cache-bypass contracts.
- TP8 vision vs official reference, multi-image/history, real cross-chunk image
  span, invalid-image and preparation-allocation rejection with recovery.
- Live cancellation after headers and after first token; following text and
  image requests succeed. This does not prove cancellation before admission.
- Restored HTTP SSE: 2+3 -> `5`, red -> `Red`, blue -> `Blue`; eight ranks alive.
- Watchdog subprocess and isolated eight-rank Gloo shutdown (all ranks exited).
- Same-start 25219-token text A/B: baseline and modified version both produced
  32 tokens, accepted 14, 17 decode steps in this sample. TTFT baseline
  2.187/.696/.697 s, modified 2.177/.695/.680 s. Not a broad performance claim.

## Cache correctness criterion
Official *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*,
section 3.2.2, PDF page 20, explains **SWA Bounded Replay**: replaying only the
last n_win tokens deliberately accepts approximate states. Suffix global/SWA KV
varies with cache-hit position; decoder reconstructed SWA KV is not equivalent
to a full forward pass either. Thus cold/warm generation need not match token
for token. Output-length differences alone are not evidence of a new bug.
Use baseline-versus-modified comparisons under matched conditions.
Source: model's `DeepSeek_V41_Tech_Report.pdf`; SHA256
`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.

## Not verified / not claimed
Independent whole-model CED/spec logits and acceptance parity; actual CUDA/NCCL
device-hang recovery; deterministic pre-admission cancellation; whole-service
peak VRAM and broad steady-state throughput. Baseline also exhibits batch/solo
output differences; no unrelated engine rewrite was undertaken to eliminate them.
Leaf, Gloo, and natural-language checks do not substitute for those validations.

Raw evidence: sibling `../image_impl_20260930/` on node09 and local
`node09_image_audit_20260930/` (progress.md, final_http_results.txt,
image_cancel_integration_results.txt, cpu_numeric_diff_summary.txt,
image_numeric_results.txt, watchdog_group_results.txt, restore_status_172.txt).
Restart configuration is referenced, not copied into the repository.

### Re-run scoped CPU tests

From the repository root (the existing test helpers use top-level imports):

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:tests python -m pytest -q tests/test_image_input.py tests/test_image_state.py tests/test_image_numeric.py
```

Final scoped run: 29 passed. Official-reference numeric tests require the model
reference files at the default path, or `DSV41_REFERENCE` pointing to inference/.
