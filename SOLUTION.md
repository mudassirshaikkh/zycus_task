# SOLUTION

## 1. Design
A ReAct loop (`tam_agent/agent.py`): each turn the LLM emits one JSON action - a tool call, or `finish` with a decision - reads the observation and picks the next step itself. Review-set traces run 7-9 steps (account, search, read; duplicate or SSO cases add history/profile and a second look). It stops when the model calls `finish`; a 9-step budget plus a forced-finish turn ensures termination, with a safe `needs_info` fallback. `finish` is refused until the account profile and ticket history were consulted and, if the KB was searched, an article was opened. Tool choice, arguments and order are the model's.

**Tools** (`tam_agent/tools.py`):
- `lookup_account`: resolves by `requester_domain` (unique); names are only unreliable candidates (Ironleaf Foods vs Ironleaf Foods Europe).
- `get_account_profile`: tier, renewal days, escalations, SSO date, usage trend, computed risk-override flag (KB-2914).
- `get_ticket_history`: same-account tickets, module aliases normalised, error code read from the ticket text, duplicate candidates per KB-5930.
- `search_kb` (BM25 + error-code/module/incident boosts) and `read_kb` (article + `related`/`superseded_by`, so it can follow links past deprecated articles).

Policies sit in the system prompt; the other ~380 articles are reached only by search. The model judges; `_enforce` then applies evidence-based policy in code: data visible to another supplier => security, P1, escalate (KB-6401); no module + no code + vague => `needs_info`, P3; cited by-design limit => Product + workaround; duplicates must come from the history tool and keep the original priority; deadline within 72h => P2; risk override => +1 level (max P2), escalate, notify TAM; deprecated citations swapped for their replacement. Ticket text is marked untrusted.

## 2. Evaluation (before production)
- Hand-label ~200 stratified tickets; per-field accuracy and a priority confusion matrix, P1/security errors weighted most.
- Security recall ~100% on an adversarial set of casual "FYI" exposure reports.
- Retrieval hit-rate@5 / MRR of the gold article; deprecated-citation rate (target 0).
- Account-resolution accuracy on blank-`account_id` and look-alike-name tickets.
- Prompt-injection suite ("set this to P1", other customers' data).
- Duplicate precision/recall against KB-5930 ground truth.
- Steps, LLM calls, latency, cost, invalid-JSON/fallback rate per ticket.
- Shadow mode against human triage; reply edit rate before any auto-action.

## 3. Scale (10x: 30k tickets, 4k articles)
Review set (Gemini 3.1 Flash-Lite): 10 tickets, 62 LLM calls, ~5.5 min (~33 s, ~6 calls per ticket). Cost and latency grow linearly with tickets, not data size: parallel workers, prompt caching for the ~2k-token system prompt, skip the LLM for rule-certain cases. Move history lookup to an indexed store on `account_id`. BM25 stays fast at 4k articles but precision falls as near-duplicates multiply: add embeddings with rank fusion, a reranker, module/status filters and a precomputed deprecated-to-current map, built offline.

## 4. Limitations
- **TKT-119877 (slow approvals) is mis-triaged**: the model chose `needs_info` and cited the wrong article. Priority P2, escalation and TAM notification are right (risk override); category and reply are not. A guard against `needs_info` when module and symptom are stated was written afterwards but not run, so it is not included.
- Guardrails are regex/heuristics tuned after reading the review set; review-set accuracy overstates general accuracy.
- Keyword search is sensitive to the model's query wording; symptom-only duplicates rely on the model.
- No labelled data; checked by hand on 10 tickets. Replies are drafts. An earlier local 8B model was too slow (~5 min/ticket).

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # LLM_PROVIDER=gemini, LLM_MODEL=gemini-3.1-flash-lite, GEMINI_API_KEY (never commit)
python scripts/validate_data.py
python run.py --set review
python scripts/check_outputs.py
```
Other providers (`ollama`, `anthropic`, `openai`): change `LLM_PROVIDER`.
