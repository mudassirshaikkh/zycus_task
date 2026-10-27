"""Check that outputs/ contains a valid result for every review-set ticket.

Run before submitting:  python scripts/check_outputs.py
Reviewers run the same script during pre-screening.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tam_agent.schema import validate  # noqa: E402

ids = [l.strip() for l in (ROOT / "data" / "review_set.txt").read_text().splitlines() if l.strip()]
passed = 0
for tid in ids:
    p = ROOT / "outputs" / f"{tid}.json"
    if not p.exists():
        print(f"[MISSING] {tid}")
        continue
    try:
        errs = validate(json.loads(p.read_text()))
    except json.JSONDecodeError as e:
        errs = [f"invalid JSON: {e}"]
    if errs:
        print(f"[INVALID] {tid}: " + "; ".join(errs))
    else:
        out = json.loads(p.read_text())
        print(f"[OK]      {tid}: {out['category']}/{out['priority']} -> {out['route_to']} "
              f"({len(out['reasoning_trace'])} steps, {len(out['kb_references'])} KB refs)")
        passed += 1
print(f"\n{passed}/{len(ids)} review tickets valid")
sys.exit(0 if passed == len(ids) else 1)
