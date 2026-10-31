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
- `scripts/run_lane_v3.sh` — full lane runbook (harvest → self-label → merge → SFT →
  fuse → double gate). All paths are `$HOME`-relative; the API key is read from
  `~/.omlx/settings.json` at runtime, nothing hardcoded.

## Training summary (Sep 29 → Oct 31, local-only on M3 Ultra 128GB)

### Data
- Harvest real Hermes session windows from `state.db` (≥15 msgs; used-sids excluded so
  eval rulers never train). Teacher = best prior model self-labels via N=2 pick.
- Final set: 150 rows = 60 frontier-audited + 90 self-labeled. Pre-fit gate:
  prompt ≤24k chars, completion ≤3200 tokens.

### 0.8B line (v1 → v27)
- Sweeps over rank/iters/lr, one DPO attempt (failed), hard-negative mining,
  hint-audited prompts. Best: `v25-it48-fused` — locked13 0.61, new32 ~0.09–0.18.
- Ceiling: d_model=1024 saturates copy recall. Two key findings — gateway default
  T=0.7 silently burned 25–30% of recall (fixed to explicit T=0.0/0.4 pick), and
  single-draw vs N=2-pick comparisons are harness artifacts.

### 2B line (q2b-v1 → v3, shipped)
- v1 (96 it): locked13 0.385 — cold start, never saw v25's distribution.
- v2 (48 it): NaN — root cause: over-length rows + `--mask-prompt` ⇒ all-masked
  completion ⇒ 0/0 loss. Fixed by the pre-fit gate.
- v3 (150 rows × 450 it, r=8 scale=16, cosine 1e-5): clean run, 0 nan.
  Shipped as `compact-q2b-v3-fused` (fused with `mlx_lm.fuse`, no LoRA keys remain).

### Results
| Ruler | v25 (0.8B) | q2b-v3-fused |
|---|---|---|
| new32 (unseen windows) | 0.0890 | **0.2148** (2.4×) |
| locked13 (production contract, same harness) | 0.6122 | 0.6066 (parity) |
| wall per summarize | ~1.2 s | ~1.2 s |

## Rules baked in (reusable for any local lane)
1. Pre-train fit gate: filter `len(prompt)+len(completion) < max_seq_length` — skips
   the silent-NaN trap.
2. Lock explicit sampling temps in eval AND production before training v1.
3. 0.8B only for schema/stop-token/scout validation; graduate to 2B in one run
   (1/3 teacher-audited + 2/3 self-labeled rows, ~3 epochs).
4. Stop the serving server before training; never poll in-flight training
   (unified-memory swap freezes).
5. Fair bars: score both models on the identical harness; fuse-and-reset — each lane
   trains on the fused winner, never the original base.

## Model card
https://huggingface.co/Bananatata/compact-q2b-v3-fused
