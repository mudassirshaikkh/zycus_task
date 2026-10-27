"""Lookup tools exposed to the agent. Every tool takes JSON-friendly args and returns a JSON-friendly dict.

All joins, matching and search live here; the LLM only sees compact results.
"""
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

from rank_bm25 import BM25Okapi

from .data import AS_OF_DATE

TODAY = date.fromisoformat(AS_OF_DATE)

CANONICAL = ["Contracts", "Integrations", "Platform", "Sourcing", "Spend Analytics",
             "Supplier Management", "eInvoice", "eProcurement"]

# free-text label -> canonical module (matched on lower-cased label)
ALIASES = {
    "clm": "Contracts", "contract": "Contracts", "contract mgmt": "Contracts",
    "sso": "Platform", "login": "Platform", "platform": "Platform",
    "sim": "Supplier Management", "supplier portal": "Supplier Management", "supplier mgmt": "Supplier Management",
    "spend": "Spend Analytics", "analytics": "Spend Analytics", "reports": "Spend Analytics",
    "eproc": "eProcurement", "e-procurement": "eProcurement", "procurement": "eProcurement",
    "rfx": "Sourcing", "esourcing": "Sourcing", "sourcing events": "Sourcing",
    "invoicing": "eInvoice", "ap": "eInvoice", "einv": "eInvoice", "e-invoice": "eInvoice",
    "erp integration": "Integrations", "api": "Integrations", "connector": "Integrations",
}
_CANON_LOWER = {c.lower(): c for c in CANONICAL}


def canonical_module(label: str) -> str | None:
    l = (label or "").strip().lower()
    return _CANON_LOWER.get(l) or ALIASES.get(l)


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())


def _norm_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(inc|ltd|llc|corp|corporation|co|group|plc|gmbh|the)\b", " ", s)
    return " ".join(s.split())


def _days_until(iso: str) -> int | None:
    try:
        return (date.fromisoformat(iso) - TODAY).days
    except ValueError:
        return None


class Toolbox:
    def __init__(self, accounts, tickets, usage, kb):
        self.accounts = accounts.set_index("account_id", drop=False)
        self.by_domain = {r.domain.lower(): r.account_id for r in accounts.itertuples()}
        self.tickets = tickets.copy()
        self.tickets["ts_"] = self.tickets["created_at"].map(datetime.fromisoformat)
        self.tickets["mod_"] = self.tickets["module"].map(canonical_module)
        self.usage = usage
        self.kb = {a["id"]: a for a in kb}
        self._ids = list(self.kb)
        self._bm25 = BM25Okapi([_tok(self._doc(a)) for a in kb])
        self.seen_ticket_ids: set[str] = set()  # ids returned by get_ticket_history (duplicate_of must come from here)

    @staticmethod
    def _doc(a):
        m = a["meta"]
        return f"{m.get('title', '')} {m.get('title', '')} {m.get('module', '')} {' '.join(m.get('error_codes', []))} {a['body']}"

    # ------------------------------------------------------------------ accounts
    def lookup_account(self, domain: str = "", company_name: str = "", account_id: str = "") -> dict:
        """Resolve an account. Domain is authoritative (unique per account); names are fuzzy and ambiguous."""
        by_dom = self.by_domain.get(domain.strip().lower()) if domain else None
        if by_dom:
            r = self.accounts.loc[by_dom]
            res = {"account_id": by_dom, "company_name": r.company_name, "match": "exact domain"}
            if account_id and account_id != by_dom:
                res["warning"] = f"ticket says {account_id} but domain maps to {by_dom}; trust the domain"
            return res
        if account_id and account_id in self.accounts.index:
            return {"account_id": account_id, "company_name": self.accounts.loc[account_id].company_name,
                    "match": "account_id as given (domain not matched)"}
        if company_name:
            q = _norm_name(company_name)
            scored = sorted(((SequenceMatcher(None, q, _norm_name(n)).ratio(), i)
                             for i, n in self.accounts.company_name.items()), reverse=True)[:3]
            return {"account_id": None, "match": "fuzzy name only - NOT reliable, similar names exist",
                    "candidates": [{"account_id": i, "company_name": self.accounts.loc[i].company_name,
                                    "domain": self.accounts.loc[i].domain, "score": round(s, 2)} for s, i in scored]}
        return {"account_id": None, "match": "none"}

    def get_account_profile(self, account_id: str) -> dict:
        if account_id not in self.accounts.index:
            return {"error": f"unknown account {account_id}"}
        r = self.accounts.loc[account_id]
        u = self.usage[self.usage.account_id == account_id].sort_values("month")
        series = [{"month": x.month, "active_users": int(x.active_users), "login_rate_pct": float(x.login_rate_pct)}
                  for x in u.tail(6).itertuples()]
        change = None
        if len(u) >= 4:
            now, then = float(u.iloc[-1].active_users), float(u.iloc[-4].active_users)
            change = round((now - then) / then * 100, 1) if then else None
        days = _days_until(r.renewal_date)
        esc = int(r.open_escalations or 0)
        override = days is not None and 0 <= days <= 90 and ((change is not None and change < -25) or esc >= 2)
        return {
            "account_id": account_id, "company_name": r.company_name, "tier": r.tier, "region": r.region,
            "industry": r.industry, "arr_band": r.arr_band, "renewal_date": r.renewal_date, "days_to_renewal": days,
            "open_escalations": esc, "nps_last": r.nps_last or None, "last_qbr_date": r.last_qbr_date or None,
            "modules_licensed": r.modules_licensed.split(";"), "sso_enabled_date": r.sso_enabled_date or None,
            "active_users_last_6_months": series, "active_users_change_3m_pct": change,
            "risk_override_applies": override,
            "risk_override_rule": "renewal<=90d AND (active users down >25% over 3m OR >=2 open escalations)",
        }

    def get_ticket_history(self, account_id: str, ticket_id: str, module: str = "", error_code: str = "",
                           days: int = 30) -> dict:
        """Tickets of the same account created before this one. Flags duplicate candidates per KB-5930."""
        me = self.tickets[self.tickets.ticket_id == ticket_id]
        ref = me.iloc[0].ts_ if len(me) else datetime.combine(TODAY, datetime.min.time())
        if len(me):  # the ticket's own fields are authoritative when the model leaves them blank
            r0 = me.iloc[0]
            if not error_code:
                m = re.search(r"E-\d{4}", f"{r0.error_code} {r0.subject} {r0.description}")
                error_code = m.group(0) if m else ""
            module = module or r0.module
        cm = canonical_module(module) or module
        h = self.tickets[(self.tickets.account_id == account_id) & (self.tickets.ticket_id != ticket_id)
                         & (self.tickets.ts_ < ref) & (self.tickets.ts_ >= ref - timedelta(days=days))]
        rows = []
        for t in h.sort_values("ts_", ascending=False).head(15).itertuples():
            age = (ref - t.ts_).total_seconds() / 86400
            same_mod = bool(cm and t.mod_ == cm)
            same_err = bool(error_code and (t.error_code == error_code or error_code in t.description))
            self.seen_ticket_ids.add(t.ticket_id)
            rows.append({"ticket_id": t.ticket_id, "created_at": t.created_at, "module": t.mod_ or t.module,
                         "error_code": t.error_code or None, "status": t.status, "customer_priority": t.customer_priority or None,
                         "subject": t.subject[:90], "age_days": round(age, 1),
                         "duplicate_candidate": t.status in ("open", "pending_customer") and age <= 7 and same_mod and same_err,
                         "same_module": same_mod, "same_error": same_err})
        # blank-account tickets from the same domain are invisible to account_id filtering
        return {"window_days": days, "count": len(h), "tickets": rows,
                "note": "duplicate_candidate = open/pending, <=7 days, same module AND same error code (KB-5930). "
                        "Judge symptom similarity yourself when error codes are missing."}

    # ------------------------------------------------------------------ knowledge base
    def search_kb(self, query: str, module: str = "", error_code: str = "", k: int = 6) -> dict:
        cm = canonical_module(module) or (module if module in CANONICAL else None)
        scores = self._bm25.get_scores(_tok(query + " " + error_code))
        out = []
        for i, s in enumerate(scores):
            a = self.kb[self._ids[i]]
            m = a["meta"]
            s = float(s)
            if error_code and error_code in m.get("error_codes", []):
                s += 15
            if cm and m.get("module") in (cm, "All"):
                s += 2.5
            if m.get("type") == "policy":
                s += 1
            if m.get("status") == "active_incident":
                s += 3
            out.append((s, a))
        out.sort(key=lambda x: -x[0])
        return {"results": [self._brief(a, s) for s, a in out[:min(k, 10)]],
                "hint": "check status: deprecated articles name superseded_by; active_incident means a known outage. "
                        "Use read_kb to read an article and follow its 'related' ids."}

    def _brief(self, a, score=None):
        m = a["meta"]
        d = {"id": a["id"], "title": m.get("title"), "module": m.get("module"), "type": m.get("type"),
             "status": m.get("status"), "error_codes": m.get("error_codes"), "last_updated": m.get("last_updated")}
        if m.get("superseded_by"):
            d["superseded_by"] = m["superseded_by"]
        if score is not None:
            d["score"] = round(score, 1)
        return d

    def read_kb(self, article_id: str) -> dict:
        a = self.kb.get(article_id)
        if not a:
            return {"error": f"no such article {article_id}"}
        d = self._brief(a)
        d["related"] = a["meta"].get("related", [])
        d["body"] = a["body"].strip()[:3500]
        if a["meta"].get("status") == "deprecated":
            d["warning"] = f"DEPRECATED - read {a['meta'].get('superseded_by')} instead; do not cite this article"
        return d

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict) -> dict:
        fn = {"lookup_account": self.lookup_account, "get_account_profile": self.get_account_profile,
              "get_ticket_history": self.get_ticket_history, "search_kb": self.search_kb, "read_kb": self.read_kb}.get(name)
        if not fn:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}


TOOL_DOC = """\
lookup_account(domain, company_name="", account_id="")   Resolve the customer account. Prefer requester_domain (authoritative); company names are ambiguous.
get_account_profile(account_id)                          Tier, renewal date/days, open escalations, usage trend, risk_override_applies flag.
get_ticket_history(account_id, ticket_id, module="", error_code="", days=30)   Earlier tickets of this account; flags duplicate candidates.
search_kb(query, module="", error_code="", k=6)          Search the knowledge base. Returns ids/titles/status only.
read_kb(article_id)                                      Full article text plus related ids. Use it to follow related/superseded_by links."""
