"""Referential-integrity check for the task dataset.

Run:  python scripts/validate_data.py
Exits non-zero if any integrity rule fails. Realistic noise (blank fields,
module aliases, company-name variants) is expected and reported, not failed.
"""
import csv, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, *p)

def read(name):
    with open(D("data", name), newline="") as f:
        return list(csv.DictReader(f))

def front_matter(path):
    text = open(path).read()
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta

accounts, tickets, usage = read("accounts.csv"), read("tickets.csv"), read("usage_metrics.csv")
review = [l.strip() for l in open(D("data", "review_set.txt")) if l.strip()]
kb = {front_matter(p)["id"]: front_matter(p) for p in glob.glob(D("kb", "*.md"))}

acc_ids = {a["account_id"] for a in accounts}
domains = {a["domain"]: a["account_id"] for a in accounts}
tix_ids = [t["ticket_id"] for t in tickets]
fails = []

def rule(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)

rule("account_id unique", len(acc_ids) == len(accounts))
rule("domain unique", len(domains) == len(accounts))
rule("ticket_id unique", len(set(tix_ids)) == len(tix_ids))
bad = [t["ticket_id"] for t in tickets if t["account_id"] and t["account_id"] not in acc_ids]
rule("every non-blank tickets.account_id exists in accounts", not bad, f"{len(bad)} bad")
bad = [t["ticket_id"] for t in tickets if t["requester_domain"] not in domains]
rule("every tickets.requester_domain maps to one account", not bad, f"{len(bad)} bad")
bad = [t["ticket_id"] for t in tickets if t["account_id"] and domains[t["requester_domain"]] != t["account_id"]]
rule("account_id and requester_domain agree", not bad, f"{len(bad)} conflicts")
bad = [u for u in usage if u["account_id"] not in acc_ids]
rule("every usage_metrics.account_id exists", not bad, f"{len(bad)} bad")
bad = [r for r in review if r not in set(tix_ids)]
rule("every review_set ticket exists", not bad, f"{len(bad)} missing")
bad = [k for k, m in kb.items() if m.get("superseded_by") and m["superseded_by"] not in kb]
rule("every KB superseded_by target exists", not bad, f"{len(bad)} bad")
rel_bad = []
for k, m in kb.items():
    for r in re.findall(r"KB-\d+", m.get("related", "")):
        if r not in kb:
            rel_bad.append((k, r))
rule("every KB related link exists", not rel_bad, f"{len(rel_bad)} bad")

print("\nExpected noise (not failures):")
print(f"  tickets with blank account_id: {sum(1 for t in tickets if not t['account_id'])}")
print(f"  tickets with blank customer_priority: {sum(1 for t in tickets if not t['customer_priority'])}")
print(f"  distinct module labels in tickets: {len({t['module'] for t in tickets})}")
print(f"  KB articles: {len(kb)} ({sum(1 for m in kb.values() if m.get('status') == 'deprecated')} deprecated)")
sys.exit(1 if fails else 0)
