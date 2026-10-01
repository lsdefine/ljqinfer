# ljqinfer CUDA TP4 — Huihui Qwen3.8-27B + DFlash2

Dedicated A800 PCIe CUDA BF16 engine on `10.176.40.140`, physical GPU **4–7**. This is the CUDA port, not the historical NPU W8A8 deployment. The target config uses Qwen3.5 geometry; the deployed directory retains the Qwen3.8 name.

## Design
- `server/`: shared HTTP protocol, authentication, tokenizer, OpenAI JSON/SSE and tool handling.
- `strategy/`: admission, cancellation, dynamic batching and TP control.
- `model/`: explicit KV/GDN past, separate prefill/decode, resident DFlash2 draft sidecar.
- `ops/`: CUDA/Triton operator ABI. Rejected experiments are outside the production tree.
- Target BF16 TP4; DFlash2 + Q8 target graphs; decode B1–B4; prefill chunk 12,288 tokens.
- HTTP temperature defaults to 1; temperature=0 uses greedy decoding. Positive temperatures use top-k=20 / top-p=.95 sampling; see `SAMPLING.md`.
- Decode step includes draft, verification and commit, NOT latency per output token.

## Deployment
Root: `/data/ljq/src/ljqinfer_qwen_tp4`
Python: `/data/ljq/vllm-env/bin/python`
Target: `/data1/models/Huihui-Qwen3.8-27B-abliterated`
Draft: `/data1/models/Qwen3.8-27B-DFlash2`

Foreground:
```bash
cd /data/ljq/src/ljqinfer_qwen_tp4
source scripts/env_tp4.sh
"$PYTHON_BIN" scripts/serve.py
```
Managed deployment:
```bash
sudo install -m 644 scripts/ljqinfer-tp4.service /etc/systemd/system/ljqinfer-tp4.service
sudo systemctl daemon-reload
sudo systemctl start ljqinfer-tp4
sudo systemctl status ljqinfer-tp4
sudo systemctl stop ljqinfer-tp4
```
The unit is not automatically enabled at boot. Never launch a second supervisor or a GPU benchmark while serving.

- Facade: `http://10.176.40.140:18084`, `/v1/models`, `/v1/chat/completions`, `/health`.
- Engine health: `http://127.0.0.1:62001/health`. Facade health alone is not model acceptance.
- API requests require `LJQINFER_API_KEY` from root-only `/etc/ljqinfer-tp4.env`; provision this file before starting the unit. Do not put credentials in this document.
- Logs: `.runtime/serve_engine.log`, `.runtime/serve_api.log`, systemd journal; rank logs `/tmp/qwen_tp4_serve_rank{0,1,2,3}.log`.

## Validation
- `scripts/cuda_acceptance.py`: real HTTP authentication, JSON, SSE and concurrency acceptance.
- CPU regression: `CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests`.
- CUDA numerical tests are retained in `tests/`; run only on idle dedicated GPUs.
- Known numerical limitation: the prior exact BF16 norm comparison reported one differing element (max absolute error 0.001953125). Directory cleanup does not fix or relax this check.
- Operator contract: `docs/cuda_ops_abi.md`.

Intermediate experiments, old NPU implementations and historical reports are not part of this CUDA release. They are archived outside the repository under `/data/ljq/project_archives/`; prior source is also recoverable through Git.
Configured context capacity is not a blanket numerical or performance acceptance claim. Model/operator arithmetic is unchanged by this cleanup.
