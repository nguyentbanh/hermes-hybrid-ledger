#!/bin/sh
# q2b-v3 extended lane: harvest ~70 more rows (floor >=15 msgs, fresh sids, v25 N=2 pick,
# drop utok>700/fin!=stop) -> merge with v27 60 + q2b-v2 41 = ~150 train rows ->
# SFT 450 iters (3 ep) cosine 1e-5 -> fuse compact-q2b-v3-fused -> double gate
# (new32 vs v25, locked13 vs 0.6059, fin_ok required).
set -eu

BASE=/Users/b/lora-finetune/compact
TOOLS=/Users/b/.local/bin
KEY=$(python3 -c 'import json;print(json.load(open("/Users/b/.omlx/settings.json"))["auth"]["api_key"])')
API=http://127.0.0.1:8000/v1/chat/completions
HEALTH=http://127.0.0.1:8000/health
MODELS=$HOME/.omlx/models
LOGDIR=$BASE/logs/q2b-v3-$(date +%Y%m%d)
HARVEST_DIR=$BASE/data/q2b-v3-sft
mkdir -p "$LOGDIR" "$HARVEST_DIR"
trap 'curl -s -m 3 "$HEALTH" >/dev/null 2>&1 || brew services start omlx >/dev/null 2>&1 || true' EXIT

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- stage 1: harvest while omlx is UP (student calls) ---------------------
python3 - <<'PYEOF' > "$LOGDIR/stage1-harvest.log" 2>&1
import json, os, re, sqlite3, sys, time, urllib.request

BASE = '/Users/b/lora-finetune/compact'
KEY = json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key']
API = 'http://127.0.0.1:8000/v1/chat/completions'
STUDENT = 'compact-q35-v25-it48-fused'
sys.path.insert(0, '/Users/b/.hermes/plugins/hybrid_ledger')
from engine import COMPACTION_SYSTEM_PROMPT, identifiers, bhit, norm, THINK, STOP

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def call(user_content, temperature):
    payload = {'model': STUDENT,
               'messages': [{'role': 'system', 'content': COMPACTION_SYSTEM_PROMPT},
                           {'role': 'user', 'content': user_content}],
               'stream': False, 'max_tokens': 900, 'temperature': temperature,
               'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json',
                                        'Authorization': f'Bearer {KEY}'})
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode())
        ch = data['choices'][0]
        return (ch.get('message', {}).get('content', '') or '',
                ch.get('finish_reason'), data.get('usage', {}).get('completion_tokens', 0))
    except Exception as exc:
        return '', f'ERR:{type(exc).__name__}', 0

used_sids = set()
for p in (f'{BASE}/data/v27-bon/winners.jsonl',
          f'{BASE}/data/lora-data-compact-q35-v6/valid.jsonl',
          f'{BASE}/data/v24/dev16.jsonl',
          f'{BASE}/data/q2b-v2-sft/v25-picked.jsonl'):
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            if r.get('sid'): used_sids.add(r['sid'])

db = sqlite3.connect(os.path.expanduser('~/.hermes/state.db'))
cur = db.cursor()
rows = cur.execute(
    "SELECT id, source, message_count FROM sessions "
    "WHERE source != 'cron' AND message_count >= 15 ORDER BY started_at DESC"
).fetchall()
cands = [r for r in rows if r[0] not in used_sids]
log(f'candidate sessions (fresh, >=15 msgs): {len(cands)}')

out = open(f'{BASE}/data/q2b-v3-sft/v25-picked.jsonl', 'w')
n_kept = n_drop_len = n_drop_thin = 0
for sid, source, mc in cands:
    if n_kept >= 70: break
    msgs = cur.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    msgs = [(r, (c or '').strip()) for r, c in msgs if (c or '').strip()]
    span = '\n'.join(f'{r.upper()}: {c[:1200]}' for r, c in msgs[-40:] if r != 'system')
    if len(span) < 1500: continue
    c0, f0, u0 = call(span, 0.0)
    c4, f4, u4 = call(span, 0.4)
    need = identifiers(span)
    r0 = sum(1 for g in need if bhit(g, norm(c0))) / max(1, len(need))
    r4 = sum(1 for g in need if bhit(g, norm(c4))) / max(1, len(need))
    pick, fin, utok = (c4, f4, u4) if r4 > r0 else (c0, f0, u0)
    src_t = '0.4' if r4 > r0 else '0.0'
    if utok > 700 or fin != 'stop':
        n_drop_len += 1
        continue
    if r0 == 0 and r4 == 0:
        n_drop_thin += 1
        continue
    out.write(json.dumps({'sid': sid, 'source': source, 'prompt': span,
                          'completion': pick, 'picked_t': src_t,
                          'recall': round(max(r0, r4), 4),
                          'utok': utok}) + '\n')
    n_kept += 1
out.close()
log(f'HARVEST DONE kept={n_kept} drop_len={n_drop_len} drop_thin={n_drop_thin}')
PYEOF
log "harvest: $(tail -1 "$LOGDIR/stage1-harvest.log")"

# ---- stage 2: merge = v27 60 + v2 41 + v3 harvest, cap prompt 24k ---------
python3 - <<'PYEOF' > "$LOGDIR/stage2-merge.log" 2>&1
import json
BASE = '/Users/b/lora-finetune/compact'
rows, seen = [], set()
n_drop = 0
for path in (f'{BASE}/data/v27-sft/train.jsonl',
             f'{BASE}/data/q2b-v2-sft/v25-picked.jsonl',
             f'{BASE}/data/q2b-v3-sft/v25-picked.jsonl'):
    for line in open(path):
        r = json.loads(line)
        key = r.get('sid') or r['prompt'][:80]
        if key in seen: continue
        seen.add(key)
        if len(r['prompt']) > 24000 or len(r['completion']) > 3200:
            n_drop += 1
            continue
        rows.append({'prompt': r['prompt'], 'completion': r['completion']})
train_rows, valid_rows = rows[:len(rows)-8], rows[len(rows)-8:]
with open(f'{BASE}/data/q2b-v3-sft/train.jsonl', 'w') as f:
    for r in train_rows: f.write(json.dumps(r) + '\n')
with open(f'{BASE}/data/q2b-v3-sft/valid.jsonl', 'w') as f:
    for r in valid_rows: f.write(json.dumps(r) + '\n')
print(f'merged train={len(train_rows)} valid={len(valid_rows)} dropped={n_drop}')
PYEOF
log "merge: $(tail -1 "$LOGDIR/stage2-merge.log")"

# ---- stage 3: SFT 450 (omlx DOWN, hard rule) -------------------------------
brew services stop omlx >/dev/null 2>&1 || true
i=0
while [ "$i" -lt 30 ]; do
  code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null || echo 000)
  [ "$code" != "200" ] && break
  sleep 2; i=$((i+1))
done
log "omlx down after ${i} polls; SFT v3 (450 iters, ga4, cosine 1e-5)"

cat > "$HARVEST_DIR/config.yaml" <<'YAML'
lora_parameters:
  rank: 8
  scale: 16.0
  dropout: 0.0
lr_schedule:
  name: cosine_decay
  arguments: [1.0e-5, 450]
YAML

/usr/bin/time -l env MLX_DISABLE_COMPILE=1 "$TOOLS/mlx_lm_lora.train" \
  --config "$HARVEST_DIR/config.yaml" --train --train-mode sft --train-type lora \
  --model mlx-community/Qwen3.5-2B-4bit --data "$HARVEST_DIR" \
  --batch-size 1 --gradient-accumulation-steps 4 --iters 450 \
  --mask-prompt --num-layers 16 --max-seq-length 8192 --steps-per-report 25 \
  --adapter-path "$BASE/runs/q2b-v3-adapters" --seed 42 > "$LOGDIR/stage3-sft.log" 2>&1
NAN=$(grep -c 'loss nan' "$LOGDIR/stage3-sft.log" || true)
log "SFT done; nan reports: $NAN (must be 0)"

# ---- stage 4: fuse + register -------------------------------------------------
env MLX_DISABLE_COMPILE=1 "$TOOLS/mlx_lm.fuse" \
  --model mlx-community/Qwen3.5-2B-4bit \
  --adapter-path "$BASE/runs/q2b-v3-adapters" \
  --save-path "$MODELS/compact-q2b-v3-fused" > "$LOGDIR/fuse.log" 2>&1
LK=$(python3 - "$MODELS/compact-q2b-v3-fused" <<'PYEOF'
import glob, json, os, sys
n = 0
for st in glob.glob(os.path.join(sys.argv[1], '*.safetensors')):
    with open(st, 'rb') as f:
        hlen = int.from_bytes(f.read(8), 'little')
        hdr = json.loads(f.read(hlen))
    n += sum(1 for k in hdr if 'lora' in k.lower())
print(n)
PYEOF
)
log "fuse check: lora keys=$LK (must be 0)"

brew services restart omlx >/dev/null 2>&1 || true
i=0
while [ "$i" -lt 90 ]; do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  sleep 2; i=$((i+1))
done
log "omlx health=$code after ${i} polls"

# ---- stage 5: double gate ------------------------------------------------------
python3 - <<'PYEOF' > "$LOGDIR/stage5-gate.log" 2>&1
import json, os, re, sys, time, urllib.request

BASE = '/Users/b/lora-finetune/compact'
KEY = json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key']
API = 'http://127.0.0.1:8000/v1/chat/completions'
sys.path.insert(0, '/Users/b/.hermes/plugins/hybrid_ledger')
from engine import COMPACTION_SYSTEM_PROMPT, identifiers, bhit, norm, THINK, STOP

def call(model, user_content):
    payload = {'model': model,
               'messages': [{'role': 'system', 'content': COMPACTION_SYSTEM_PROMPT},
                           {'role': 'user', 'content': user_content}],
               'stream': False, 'max_tokens': 900,
               'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json',
                                        'Authorization': f'Bearer {KEY}'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode())
        ch = data['choices'][0]
        return ch.get('message', {}).get('content', '') or '', ch.get('finish_reason'), time.time() - t0
    except Exception as exc:
        return '', f'ERR:{type(exc).__name__}', time.time() - t0

def make_hint(span):
    need = identifiers(span)
    seen, ids = set(), []
    for w in re.findall(r'\S+', norm(THINK.sub(' ', span))):
        w = re.sub(r'[*_`]+', '', w)
        prev = None
        while prev != w:
            prev = w
            w = re.sub(r'^[^a-z0-9/._-]+|[^a-z0-9/._-]+$', '', w)
        if not w or w in STOP or 'think' in w or w[0] == '.' or w[-1] == '.':
            continue
        shaped = ('.' in w or '/' in w or '-' in w[1:] or '_' in w
                  or re.fullmatch(r'[0-9]{2,}', w))
        if shaped and len(w) < 40 and w in need and w not in seen:
            seen.add(w); ids.append(w)
    nums = [x for x in ids if any(c.isdigit() for c in x)]
    artifacts = [x for x in ids if x not in nums and ('/' in x or '.' in x)]
    decisions = [x for x in ids if x not in nums and x not in artifacts]
    lines = []
    if decisions:
        lines.append('- For ### Decisions/blockers: ' + '; '.join(decisions))
    if artifacts:
        lines.append('- For ### Active artifacts: ' + '; '.join(artifacts))
    if nums:
        lines.append('- For ### Numerical state: ' + '; '.join(nums))
    words = min(170, max(80, len(ids) * 2 + 40))
    return ('\n\n[Verbatim technical identifiers & numbers to integrate into the summary:\n'
            + '\n'.join(lines)
            + '\nStrict format: write 2-3 telegraphic bullets per heading (max 8 bullets). '
            + 'Emit each listed item at most once without repetition. '
            + f'Write at most {words} words. Conclude recap and STOP immediately after ### Numerical state.]')

res = {'q2b_v3': [], 'v25': []}
wins = [json.loads(l) for l in open(f'{BASE}/data/v27-bon/winners.jsonl')]
for w in wins:
    for model, tag in (('compact-q2b-v3-fused', 'q2b_v3'),
                       ('compact-q35-v25-it48-fused', 'v25')):
        text, fin, wall = call(model, w['prompt'])
        need = identifiers(w.get('span') or w['prompt'])
        rec = sum(1 for g in need if bhit(g, norm(text))) / max(1, len(need))
        res[tag].append({'win_idx': w['win_idx'], 'recall': round(rec, 4),
                         'fin': fin, 'wall': round(wall, 2)})
    print(f"w{w['win_idx']:02d} q2b={res['q2b_v3'][-1]['recall']:.3f}({res['q2b_v3'][-1]['fin']}) "
          f"v25={res['v25'][-1]['recall']:.3f}({res['v25'][-1]['fin']})", flush=True)
n32_q = sum(r['recall'] for r in res['q2b_v3']) / len(res['q2b_v3'])
n32_v = sum(r['recall'] for r in res['v25']) / len(res['v25'])
print(f'new32: q2b_v3={n32_q:.4f} v25={n32_v:.4f}', flush=True)

lq = []
for line in open(f'{BASE}/data/lora-data-compact-q35-v6/valid.jsonl'):
    r = json.loads(line)
    blocks = re.findall(r'<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>',
                        r['text'], re.S)
    span = '\n'.join(f'{role.upper()}: {content.strip()}'
                     for role, content in blocks if role != 'system')
    text, fin, wall = call('compact-q2b-v3-fused', span + make_hint(span))
    need = identifiers(span)
    rec = sum(1 for g in need if bhit(g, norm(text))) / max(1, len(need))
    lq.append({'recall': round(rec, 4), 'fin': fin, 'wall': round(wall, 2)})
    print(f"locked w{len(lq):02d}: {rec:.3f} fin={fin} wall={wall:.1f}s", flush=True)
lmean = sum(r['recall'] for r in lq) / len(lq)
fin_ok = all(r['fin'] == 'stop' for r in lq)
wallmean = sum(r['wall'] for r in lq) / len(lq)

ship = (n32_q > n32_v) and (lmean > 0.6059) and fin_ok
d = {'new32_q2b_v3': round(n32_q, 4), 'new32_v25': round(n32_v, 4),
     'locked13_mean': round(lmean, 4), 'fin_ok': fin_ok,
     'wallmean_s': round(wallmean, 3), 'ship': ship,
     'verdict': ('SHIP q2b-v3 (both gates pass)' if ship else
                 f'KEPT v25 (new32 {n32_q:.4f}/{n32_v:.4f}, locked13 {lmean:.4f}/0.6059, fin_ok={fin_ok})')}
json.dump(d, open(f'{BASE}/data/q2b/v3_ship_decision.json', 'w'), indent=1)
print('DECISION:', d['verdict'])
PYEOF

log "lane complete: $(tail -1 "$LOGDIR/stage5-gate.log")"
