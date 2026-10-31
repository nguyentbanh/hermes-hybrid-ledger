# hermes-hybrid-ledger

Hermes agent plugin: **compaction worker** (`auxiliary.compression.model` lane).

Compacts long agent context windows into a 4-heading recap that preserves verbatim
identifiers:

```
### Decisions/blockers
### Active artifacts
### Pending user ask
### Numerical state
```

## Layout
- `engine.py` — the worker. Calls a local model server (oMLX :8000) with an
  **N=2 pick**: greedy T=0.0 floor + T=0.4 upside; the upside candidate is accepted
  only on `finish_reason == "stop"` AND strictly higher identifier recall (ties keep
  greedy). Failed floor → raise → gateway fallback/cooldown. No truncated recap can ship.
- `scripts/run_lane_v3.sh` — full lane runbook: harvest real session windows from
  `state.db` → self-label with N=2 pick → merge to ~150 rows (pre-fit gate:
  prompt ≤24k chars, completion ≤3200 tokens) → `mlx_lm.lora` SFT (r=8 scale=16,
  450 iters, cosine 1e-5, mask-prompt) → `mlx_lm.fuse` → double gate (new32 + locked13).

## Shipped model
[Bananatata/compact-q2b-v3-fused on HuggingFace](https://huggingface.co/Bananatata/compact-q2b-v3-fused) —
fused LoRA over Qwen3.5-2B-4bit. Rulers: new32 0.2148 (2.4× the 0.8B predecessor),
locked13 contract 0.6066 (parity), ~1.2 s wall per summarize on M3 Ultra.

## Rules baked in (from a month of lanes)
1. Pre-train fit gate: filter `len(prompt)+len(completion) < max_seq_length` — over-length rows + `--mask-prompt` = silent all-masked → NaN loss.
2. Lock explicit sampling temps in eval AND production before training v1 (gateway default T=0.7 masks quality).
3. 0.8B only for schema/stop-token/scout validation; graduate to 2B in one run (1/3 teacher-audited + 2/3 self-labeled rows, ~3 epochs).
4. Stop the serving server before training; never poll in-flight training (unified-memory swap freezes).
5. Fair bars: score both models on the identical harness; single-draw vs N=2-pick comparisons are artifacts.
