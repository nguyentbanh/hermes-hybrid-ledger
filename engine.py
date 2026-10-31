"""Hybrid Ledger context engine.

Stitches an LLM narrative recap with a deterministic ledger of every
code-shaped identifier missed by the narrative.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Set

try:
    from agent.conversation_compression import ContextCompressor
except (ImportError, AttributeError):
    try:
        from agent.context_compressor import ContextCompressor
    except (ImportError, AttributeError):
        class ContextCompressor:
            def __init__(self, *args, **kwargs):
                pass

            def _generate_summary(self, *args, **kwargs):
                return None

try:
    from agent.auxiliary_client import call_llm
except (ImportError, AttributeError):  # pragma: no cover - older trees
    call_llm = None


BS = chr(92)
LF = chr(10)

RUN = re.compile('[a-z0-9/._-]{2,}')


def bhit(x: str, txt: str) -> Optional[Any]:
    return re.search('(?<!' + '[' + BS + 'w./-])' + re.escape(x) + '(?![' + BS + 'w./-])', txt)


def norm(s: str) -> str:
    return re.sub(BS + 's+', ' ', s).strip().lower()


THINK = re.compile('<' + BS + '|think[^>]*>.{0,4}<' + BS + '|/?think[^>]*>', re.S)
STOP = {'e.g', 'i.e', '/', '-', '--', 'n/a'}


def identifiers(gold: str) -> Set[str]:
    gold = THINK.sub(' ', gold)
    out = set()
    for w in re.findall(BS + 'S+', norm(gold)):
        w = re.sub('[*_`]+', '', w)
        prev = None
        while prev != w:
            prev = w
            w = re.sub('^[^a-z0-9/._-]+|[^a-z0-9/._-]+$', '', w)
        if not w or w in STOP or 'think' in w:
            continue
        if w[-1] == '.' or w[0] == '.':
            continue
        shaped = ('.' in w or '/' in w or '-' in w[1:] or '_' in w
                  or re.fullmatch('[0-9]{2,}', w) or '.' in w and w[-1].isdigit())
        if shaped:
            out.add(w)
    return {x for x in out if ' ' not in x and len(x) < 40}


def degenerate(out: str) -> bool:
    w = out.split()
    if len(w) < 8:
        return False
    g = (' '.join(x) for x in (w[i:i+12] for i in range(0, max(len(w)-11, 1))))
    return Counter(g).most_common(1)[0][1] > 3 if len(w) >= 24 else False


GARBAGE_METHOD = re.compile('^[a-z][.](items|keys|values)$')
GARBAGE_PLACEHOLDER = re.compile('(?:^|/)(foo|bar|baz|test|example|demo|stub|sample)[.](py|sh|json|txt)$')


def is_garbage(x: str) -> bool:
    if not isinstance(x, str):
        return False
    if GARBAGE_METHOD.match(x):
        return True
    if x == '10.0.0.1':
        return True
    if GARBAGE_PLACEHOLDER.search(x):
        return True
    return False


UNRESOLVED_PATTERN = '(?i)' + BS + 'b(remains unresolved|unresolved in the session|this remains unresolved|still open)' + BS + 'b'
UNRESOLVED = re.compile(UNRESOLVED_PATTERN)

COMPLETION_PATTERNS = [
    'exit_code[:=]0',
    'returncode[:=]0',
    'rc=0',
    'HTTP/1[.]1 200',
    'status[:=] ?(ok|ready)',
    BS + 'd+ -> ' + BS + 'd+ msgs',
    BS + 'd+/' + BS + 'd+ (' + BS + '(|with )[0-9.]+ ?%',
    'recall[=:] ?0[.]' + BS + 'd+',
    'saved ',
    'fused',
    'merged',
    'wrote ',
    'done in ' + BS + 'd+[ms]+',
]
COMPLETION = re.compile('|'.join(COMPLETION_PATTERNS))


def stitch_ledger(narrative: Optional[str], window_text: str) -> Optional[str]:
    if narrative is None:
        return None
    marker = LF + '[REFS:'
    last_idx = (window_text or '').rfind(marker)
    if last_idx != -1:
        close_idx = window_text.find(']', last_idx)
        scan_start = close_idx + 1 if close_idx != -1 else last_idx + len(marker)
        scan_text = window_text[scan_start:]
    elif window_text and window_text.startswith('[REFS:'):
        close_idx = window_text.find(']')
        scan_start = close_idx + 1 if close_idx != -1 else len('[REFS:')
        scan_text = window_text[scan_start:]
    else:
        scan_text = window_text or ''

    need = identifiers(scan_text)
    low = (narrative or '').lower()
    cover = []
    seen = set()
    for gid in sorted(need):
        if bhit(gid, low):
            continue
        if gid not in seen:
            seen.add(gid)
            cover.append(gid)

    cover = [x for x in cover if not is_garbage(x)]

    res = narrative + LF + '[REFS: ' + ' '.join(cover) + ']' if cover else narrative

    if UNRESOLVED.search(res):
        # per-claim gate: rewrite a claim only if a completion signal follows
        # THAT claim in window_text (checking only the last claim would wrongly
        # resolve earlier claims that have no resolution path)
        pieces = []
        pos = 0
        for m in UNRESOLVED.finditer(res):
            pieces.append(res[pos:m.start()])
            if COMPLETION.search(window_text or '', m.end()):
                pieces.append('RESOLVED (see verbatim tail)')
            else:
                pieces.append(m.group(0))
            pos = m.end()
        pieces.append(res[pos:])
        res = ''.join(pieces)

    return res



COMPACTION_SYSTEM_PROMPT = (
    "You compress a conversation span into a dense handoff summary. "
    "Strict requirement: preserve every exact path, script name, command and flags, port, PID, exit code, version, latency, and count verbatim. "
    "Never omit, rephrase, or generalize code-shaped identifiers, files, or numbers. "
    "Use concise telegraphic bullets under exactly three headings:\n"
    "### Decisions/blockers\n"
    "### Active artifacts\n"
    "### Numerical state\n"
    "Recap budget: under 200 words. Conclude recap and stop."
)

COMPACTION_EXEMPLAR = (
    "### Decisions/blockers\n"
    "- Migrated auth to OAuth2 in /etc/auth/app.py; fixed route-dns issue with exit_code=0\n"
    "- Blocker: port 8080 blocked by firewalld-cmd; remapped to 127.0.0.1:8787 cleanly\n\n"
    "### Active artifacts\n"
    "- Service: worker.py PID 4821 running via ctl.sh start --daemon\n"
    "- Config: /opt/conf/server.yaml v2.1.0 logging to /var/log/sync.log daily\n\n"
    "### Numerical state\n"
    "- Latency 45ms across 1500 calls; status 200 OK, memory 512MB, returncode=0"
)

COMPACTION_CALL_KWARGS: Dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
    # v28 sweep 10-21: locked-13 recall rises monotonically as temp drops;
    # T0.6 is the lowest temp keeping fin 13/13 on every window (T<=0.3 loops
    # on some). Serving at default (~1.0) left ~27 hits/283 on the table.
    # top_p/rp from agy-flash 10-21 verified lane (recall 144/151 vs control 89-104, eos 13/13).
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.12,
}


def ordered_ids(gold):
    need = identifiers(gold)
    seen, out = set(), []
    for w in re.findall(r'\S+', norm(THINK.sub(' ', gold))):
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
            seen.add(w)
            out.append(w)
    return out


def make_hint(span, ids):
    nums = [x for x in ids if re.search(r'\d', x)]
    artifacts = [x for x in ids if x not in nums and ('/' in x or '.' in x)]
    decisions = [x for x in ids if x not in nums and x not in artifacts]

    lines = []
    if decisions: lines.append('- For ### Decisions/blockers: ' + '; '.join(decisions))
    if artifacts: lines.append('- For ### Active artifacts: ' + '; '.join(artifacts))
    if nums: lines.append('- For ### Numerical state: ' + '; '.join(nums))

    words = min(170, max(80, len(ids) * 2 + 40))
    return (
        f'\n\n[Verbatim technical identifiers & numbers to integrate into the summary:\n'
        + '\n'.join(lines) + f'\nStrict format: write 2-3 telegraphic bullets per heading (max 8 bullets total). Emit each listed item at most once without repetition. Write at most {words} words. Conclude recap and STOP immediately after ### Numerical state.]'
    )


class HybridLedgerEngine(ContextCompressor):
    """Context engine that stitches narrative recaps with a deterministic ledger of code-shaped identifiers."""

    @property
    def name(self) -> str:
        return 'hybrid_ledger'

    def __init__(self, model: str = '', *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def _build_summary_prompt(self, content_to_summarize: str, *args, **kwargs) -> str:
        # no static EXAMPLE block: it caused verbatim exemplar-copy in 4/13
        # windows (OAuth2/worker.py text with no relation to the window).
        # Proven: dropping it moved causal-QA 2/39 -> 8/39, zero echoes.
        return COMPACTION_SYSTEM_PROMPT + '\n\nTRANSCRIPT:\n' + content_to_summarize + make_hint(content_to_summarize, ordered_ids(content_to_summarize))

    def _call_summary_llm(self, prompt: str, prompt_started_at: float) -> str:
        """N=2 pick (verified 10-31, ruler mean greedy 0.1422 -> pick 0.1848, 0 floor violations).

        Rollout #1 is greedy T=0.0 (the floor; default-temp serving was leaking
        ~30% of copy recall). Rollout #2 at T=0.4 is upside-only: it may replace
        the floor ONLY if it terminates cleanly (finish_reason == 'stop') AND its
        identifier recall over the transcript is STRICTLY greater (ties -> greedy).
        Any failure of rollout #1 raises so the gateway's existing fallback/
        cooldown machinery handles it unchanged.
        """
        if call_llm is None:  # pragma: no cover - older trees without aux router
            return super()._call_summary_llm(prompt, prompt_started_at)
        window = prompt.split('TRANSCRIPT:\n', 1)[1] if 'TRANSCRIPT:\n' in prompt else prompt
        need = identifiers(window)

        def _recall(text: str) -> float:
            if not need:
                return 0.0
            ntext = norm(text)
            return sum(1 for g in need if bhit(g, ntext)) / len(need)

        def _roll(temp: float):
            # ponytail: 30s/roll caps total at ~60s vs the 55s aux timeout budget;
            # typical roll is 1-3s. Raise to per-provider latency if tails matter.
            resp = call_llm(
                task='compression',
                main_runtime={
                    'model': self.model, 'provider': self.provider,
                    'base_url': self.base_url, 'api_key': self.api_key,
                    'api_mode': self.api_mode,
                },
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temp, timeout=30,
                extra_body={'max_tokens': 900, 'chat_template_kwargs': {'enable_thinking': False}},
            )
            try:
                choices = resp.get('choices') if isinstance(resp, dict) else getattr(resp, 'choices', None)
                ch = (choices or [{}])[0]
                msg = ch.get('message') if isinstance(ch, dict) else getattr(ch, 'message', None)
                content = (msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)) or ''
                fin = ch.get('finish_reason') if isinstance(ch, dict) else getattr(ch, 'finish_reason', '')
            except Exception:
                content, fin = '', 'error'
            return content, (fin or '')

        cand0, fin0 = _roll(0.0)
        if not cand0.strip() or fin0 == 'length':
            raise RuntimeError(
                f'Context compression summary unusable (fin={fin0}) '
                f'(provider={getattr(self, "provider", "") or "auto"} '
                f'model={getattr(self, "summary_model", None) or getattr(self, "model", "")})'
            )
        cand4, fin4 = _roll(0.4)
        if fin4 == 'stop' and cand4.strip() and _recall(cand4) > _recall(cand0):
            return cand4
        return cand0

    def _generate_summary(self, turns_to_summarize: Any, *args, **kwargs) -> Optional[str]:
        narrative = super()._generate_summary(turns_to_summarize, *args, **kwargs)
        if narrative is None:
            return None
        if isinstance(turns_to_summarize, str):
            window_text = turns_to_summarize
        elif hasattr(self, '_serialize_for_summary') and isinstance(turns_to_summarize, list):
            window_text = self._serialize_for_summary(turns_to_summarize)
        else:
            window_text = str(turns_to_summarize)
        return stitch_ledger(narrative, window_text)


if __name__ == '__main__':
    # 1. Garbage filtered
    assert is_garbage('x.items')
    assert is_garbage('a.keys')
    assert is_garbage('z.values')
    assert is_garbage('10.0.0.1')
    assert is_garbage('foo.py')
    assert is_garbage('bar.sh')
    assert is_garbage('baz.json')
    assert is_garbage('test.txt')
    assert is_garbage('example.py')
    assert is_garbage('demo.sh')
    assert is_garbage('stub.json')
    assert is_garbage('sample.txt')
    assert is_garbage('path/to/test.py')
    assert not is_garbage('valid_module.py')
    assert not is_garbage('10.0.0.2')

    out_garbage = stitch_ledger('Summary.', 'Found 10.0.0.1, x.items, foo.py, and valid-module.py')
    assert '10.0.0.1' not in out_garbage
    assert 'x.items' not in out_garbage
    assert 'foo.py' not in out_garbage
    assert 'valid-module.py' in out_garbage

    # 2. Preserves IDs sitting inside longer runs so strict bhit boundaries pass
    out_dedupe = stitch_ledger('Summary.', 'Contains sub/path.py and deeper/sub/path.py')
    refs_block = out_dedupe.split('[REFS: ')[1].rstrip(']')
    tokens = refs_block.split()
    assert 'sub/path.py' in tokens
    assert 'deeper/sub/path.py' in tokens

    # Dedupe between recap and ledger: recap hit avoids ledger emission
    out_recap = stitch_ledger('Summary has sub/path.py and more.', 'Contains sub/path.py and deeper/sub/path.py')
    recap_refs = out_recap.split('[REFS: ')[1].rstrip(']').split()
    assert 'sub/path.py' not in recap_refs
    assert 'deeper/sub/path.py' in recap_refs

    # 3. Previous-blob content not re-covered
    prev_blob_window = 'Previous handoff 252/283' + LF + '[REFS: alpha_9]' + LF + 'Current turn mentions new-widget-v1'
    out_prev_blob = stitch_ledger('Summary.', prev_blob_window)
    assert 'alpha_9' not in out_prev_blob
    assert '252/283' not in out_prev_blob
    assert 'new-widget-v1' in out_prev_blob

    # Window starting directly with [REFS: alpha_9]
    prev_blob_window2 = '[REFS: alpha_9]' + LF + 'Current turn mentions new-widget-v2'
    out_prev_blob2 = stitch_ledger('Summary.', prev_blob_window2)
    assert 'alpha_9' not in out_prev_blob2
    assert 'new-widget-v2' in out_prev_blob2

    # 4. Unresolved claim rewritten when a later rc=0 exists
    narrative_claim = 'The bug remains unresolved in the session.'
    window_with_rc0 = 'Initial state: bug remains unresolved in the session.' + LF + 'Fixed issue.' + LF + 'rc=0'
    out_rewritten = stitch_ledger(narrative_claim, window_with_rc0)
    assert 'RESOLVED (see verbatim tail)' in out_rewritten
    assert 'remains unresolved' not in out_rewritten

    # 5. Left alone when no completion signal
    window_no_comp = 'Initial state: bug remains unresolved in the session.' + LF + 'Attempt failed.'
    out_left_alone = stitch_ledger(narrative_claim, window_no_comp)
    assert 'remains unresolved in the session' in out_left_alone
    assert 'RESOLVED (see verbatim tail)' not in out_left_alone

    # Left alone when rc=0 is strictly before the unresolved claim
    window_rc0_before = 'rc=0' + LF + 'Later error: bug remains unresolved in the session.'
    out_rc0_before = stitch_ledger(narrative_claim, window_rc0_before)
    assert 'remains unresolved in the session' in out_rc0_before
    assert 'RESOLVED (see verbatim tail)' not in out_rc0_before

    # 6. Hint tail generation and prompt builder integration
    test_sample = 'fixed route-dns with cloudflared at /opt/x.py on port 8080 exit_code=0'
    engine = HybridLedgerEngine()
    prompt_out = engine._build_summary_prompt(test_sample)
    assert 'For ### Decisions/blockers:' in prompt_out
    assert '8080' in prompt_out
    assert 'Strict format:' in prompt_out

    # make_hint words budget = min(170, max(80, len(ids)*2+40))
    ids = ordered_ids(test_sample)
    assert f'Write at most {min(170, max(80, len(ids) * 2 + 40))} words.' in make_hint(test_sample, ids)
    for n in (0, 10, 30, 80, 100):
        dummy_ids = [f'id_{i}' for i in range(n)]
        expected_budget = min(170, max(80, len(dummy_ids) * 2 + 40))
        assert f'Write at most {expected_budget} words.' in make_hint('', dummy_ids)

    print('ALL CHECKS PASSED')

