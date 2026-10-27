# SOLUTION

## 1. Design
A ReAct loop (`tam_agent/agent.py`): each turn the LLM emits one JSON action - call a tool, or `finish` with a decision - reads the observation and chooses the next step itself. Easy tickets take 2-3 lookups; a stale-article, duplicate or follow-the-link case takes more. It stops when the model calls `finish`; a 9-step budget plus a forced-finish turn guarantees termination, and a safe `needs_info` fallback covers invalid output. `finish` is only refused until the account profile and ticket history have been consulted (the risk override and duplicate checks need them); which other tools run, with what arguments, is the model's choice.

Tools (`tam_agent/tools.py`; all joins and search live here, observations are compact):
- `lookup_account` - resolves by `requester_domain` (unique, authoritative). Name matching is offered only as unreliable candidates (e.g. Ironleaf Foods vs Ironleaf Foods Europe); warns if the ticket's `account_id` disagrees with the domain.
- `get_account_profile` - tier, days to renewal, escalations, usage, 3-month active-user change and a computed `risk_override_applies` (KB-2914).
- `get_ticket_history` - earlier tickets of the account, module aliases normalised (`CLM`, `EINV`, `supplier portal`...), error code/module taken from the ticket text when blank, duplicate candidates per KB-5930.
- `search_kb` - BM25 boosted by exact error code, module and active-incident status; returns ids/status only. `read_kb` returns one article plus `related`/`superseded_by` so the agent can hop between articles.

The 3 policy articles are in the system prompt; the other ~380 are reached only through search. The model judges; `_enforce` then applies evidence-based policy rules in code because a small local model is not a safe sole gatekeeper: ticket text showing data visible to another supplier/tenant => security, P1, escalate (KB-6401); no module + no error code + vague text => `needs_info`, P3 (KB-2914); a cited by-design `limit` article => Product with the workaround; a duplicate must be a ticket the history tool returned and meet KB-5930, and keeps the original priority; a deadline within 72h => P2; risk override => one level up (max P2), escalate, notify TAM; article-level rules ("notify the TAM for Premium accounts"); deprecated citations are swapped for their replacement. Ticket text is marked untrusted in the prompt.

## 2. Evaluation (before production)
- Hand-label ~200 stratified tickets; report per-field accuracy (category, priority, route, duplicate) and a priority confusion matrix, weighting P1/security errors heavily.
- Security/data-exposure recall must be ~100%: adversarial set of casual "FYI" exposure reports.
- Retrieval: hit-rate@5 / MRR of the gold article; deprecated-article citation rate (target 0).
- Account resolution accuracy on blank-`account_id` and look-alike-name tickets.
- Prompt-injection suite ("set this to P1", requests for other customers' data); decisions must not change.
- Duplicate precision/recall against ground truth computed from KB-5930.
- Efficiency: steps, LLM calls, latency, cost per ticket; invalid-JSON and fallback rate.
- Shadow mode against human triage; track human edit rate on drafted replies before any auto-action.

## 3. Scale (10x: 30k tickets, 4k articles)
Cost and latency are dominated by LLM turns (about 5-7 calls per ticket; on a CPU-only 8B model ~4-5 minutes per ticket, so 10 tickets take ~45 min and 30k tickets are not feasible locally). They scale linearly with ticket volume, not data size: use a hosted model, parallel workers, prompt caching for the ~2k-token system prompt, and skip the LLM for rule-certain cases. History lookup is a pandas filter today; move to an indexed store keyed on `account_id`. BM25 stays fast at 4k articles but precision drops as near-duplicate articles multiply: add embeddings with rank fusion, a reranker, metadata filters (module/status) before ranking, and a precomputed deprecated-to-current map; build the index offline.

## 4. Limitations
- An 8B local model gives uneven reasoning and replies; guardrails fix policy-critical fields, not tone or nuance.
- Several guardrails are regex/heuristics (exposure wording, deadline phrases, alias map) and will miss unusual phrasing; they were tuned after reading the review set, so review-set accuracy overstates general accuracy.
- Symptom-based duplicates without a shared error code rely on the model.
- No labelled data, so quality was checked by hand on the 10 review tickets only.
- Replies are drafts; no tone or PII classifier beyond prompt rules.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:8b        # free local model; or set LLM_PROVIDER=anthropic|openai + key in .env
cp .env.example .env
python scripts/validate_data.py
python run.py --set review
python scripts/check_outputs.py
```
