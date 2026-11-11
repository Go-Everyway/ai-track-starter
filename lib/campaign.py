"""The campaign layer (Module 12): signals to offers, a scored list, a holdout, suppression,
a sending-setup check, and results you can defend.

    from lib import campaign, pipeline
    world = pipeline.World.load()
    ranked = campaign.scored_list(world)                    # leads with a profile, scored, offers mapped
    groups = campaign.assign_holdout(ranked.lead_id, share=0.5, seed=12)
    campaign.record_before({"meetings_per_week": 3.0}, period="last 8 weeks")   # before anything runs
    stop = campaign.Suppression()                           # opt-outs and do-not-contact
    campaign.send_message(world, stop, draft_id, token, recipient=..., idempotency_key=...)
    report = campaign.results(outcomes)                     # treatment vs holdout, with sample sizes and a range

Nothing here sends real email. `send_message` puts an approved draft in the pipeline's outbox
(Module 9) only if the recipient isn't suppressed. `check_sending_setup` checks the text of
SPF, DKIM and DMARC records you paste in; `lookup_txt` fetches them with `dig` when it's
installed. `simulate_outcomes` makes SIMULATED replies and meetings from stated assumptions,
so the results code can be exercised; it says nothing about what a real campaign would do.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lib import pipeline

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"

# ---------------------------------------------------------------- signals to offers

# Signal text (as it appears in companies.jsonl) -> what Tallyhawk would talk about.
# A signal with no offer is trivia: it doesn't earn a message.
SIGNAL_OFFERS = {
    "is hiring a demand planner": "Forecasting that gives a new planner a head start instead of a spreadsheet",
    "posted two warehouse operations roles this month": "Reorder suggestions that keep a growing team from over-ordering",
    "announced a second distribution center": "Per-warehouse forecasts for the new site from day one",
    "reported stockouts during the last peak season": "Seasonal forecasts and earlier reorders before the next peak",
    "is replacing its ERP": "A forecasting layer that connects to the new ERP instead of living in it",
    "is consolidating suppliers": "Lead-time-aware reorders as supplier terms change",
    "expanded into a new region": "Forecasts per warehouse as demand patterns split by region",
}


def map_signals(company: dict) -> dict:
    """{"offers": [{"signal", "offer"}], "unmapped": [signals with no offer]} for one profile."""
    offers, unmapped = [], []
    for s in company.get("signals", []):
        (offers.append({"signal": s, "offer": SIGNAL_OFFERS[s]}) if s in SIGNAL_OFFERS else unmapped.append(s))
    return {"offers": offers, "unmapped": unmapped}


def scored_list(world: pipeline.World, *, require_offer: bool = False) -> pd.DataFrame:
    """Every lead whose company has a profile, scored with Module 4's model, with its first
    mapped offer. Sorted best first."""
    rows = []
    for lead in world.leads.to_dict("records"):
        profile = world.companies.get(lead["company_id"])
        if profile is None:
            continue
        mapped = map_signals(profile)
        if require_offer and not mapped["offers"]:
            continue
        rows.append({"lead_id": lead["lead_id"], "company_id": lead["company_id"], "company": profile["name"],
                     "industry": profile["industry"], "score": pipeline.score(world, lead),
                     "offer": mapped["offers"][0]["offer"] if mapped["offers"] else None,
                     "signal": mapped["offers"][0]["signal"] if mapped["offers"] else None})
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- holdout

def assign_holdout(ids, *, share: float = 0.5, seed: int = 0) -> dict:
    """Randomly split ids into "treatment" and "holdout" before anything is sent.

    Deterministic for a given seed, and stable per id: adding new ids later doesn't move the
    old ones, because each id's group comes from a hash of (seed, id), not from its position."""
    def u(i):
        h = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
        return int(h[:12], 16) / 16 ** 12
    return {i: ("holdout" if u(i) < share else "treatment") for i in ids}


# ---------------------------------------------------------------- the before number

BEFORE_PATH = LOGS / "before.json"


def record_before(values: dict, *, period: str, path: Path | None = None) -> dict:
    """Write down the before number, with a timestamp, before the system touches anything.
    Refuses to overwrite an existing record: a before number you can edit later isn't one."""
    path = Path(path or BEFORE_PATH)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the before number is recorded once. Delete it on purpose to start over.")
    rec = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "period": period, "values": values}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def read_before(path: Path | None = None) -> dict | None:
    path = Path(path or BEFORE_PATH)
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- suppression and opt-outs

class Suppressed(Exception):
    """Raised when a send is attempted to someone on the suppression list."""


def add_business_days(d: date, n: int) -> date:
    while n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


@dataclass
class Suppression:
    """Who must never get another message. Addresses and whole domains, each with a reason and
    the date it arrived. Check it before every send, including follow-ups already scheduled."""
    entries: dict = field(default_factory=dict)  # key -> {"reason", "received", "honour_by"}

    @staticmethod
    def _key(address: str) -> str:
        return address.strip().lower()

    def add(self, address: str, reason: str, received: date | None = None) -> dict:
        received = received or date.today()
        # US and Canadian law both allow up to 10 business days; Yahoo asks bulk senders for
        # 2 days. Suppressing at once, as this does, is the only setting that's never late.
        rec = {"reason": reason, "received": received.isoformat(),
               "yahoo_by": (received + timedelta(days=2)).isoformat(),
               "legal_by": add_business_days(received, 10).isoformat()}
        self.entries[self._key(address)] = rec
        return rec

    def opt_out(self, address: str, received: date | None = None) -> dict:
        return self.add(address, "opted out", received)

    def is_suppressed(self, address: str) -> bool:
        a = self._key(address)
        domain = a.split("@")[-1]
        return a in self.entries or f"@{domain}" in self.entries

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.entries, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Suppression":
        p = Path(path)
        return cls(json.loads(p.read_text())) if p.exists() else cls()


def send_message(world: pipeline.World, suppression: Suppression, draft_id: str, token: str | None, *,
                 recipient: str, idempotency_key: str) -> dict:
    """Send one approved draft, unless the recipient is suppressed. Suppression is checked
    first, so even a valid approval can't reach someone who opted out."""
    if suppression.is_suppressed(recipient):
        raise Suppressed(f"{recipient} is on the suppression list; nothing sent")
    record = pipeline.send(world, draft_id, token, idempotency_key=idempotency_key)
    record.setdefault("recipient", recipient)
    return record


# ---------------------------------------------------------------- sending setup

def check_spf(record: str) -> list[str]:
    r = record.strip().strip('"')
    problems = []
    if not r.startswith("v=spf1"):
        return ["SPF must start with v=spf1"]
    terms = r.split()[1:]
    alls = [t for t in terms if t.lstrip("+-~?") == "all"]
    if not alls:
        problems.append("no 'all' at the end: say what happens to everyone else (-all or ~all)")
    elif alls[-1] in ("all", "+all"):
        problems.append("+all lets anyone send as you; use -all or ~all")
    elif terms[-1] != alls[-1]:
        problems.append("'all' should be the last term; anything after it is ignored")
    lookups = sum(1 for t in terms if re.match(r"^[+\-~?]?(include:|a\b|a:|mx\b|mx:|ptr|exists:|redirect=)", t))
    if lookups > 10:
        problems.append(f"{lookups} DNS lookups; SPF allows at most 10 (RFC 7208)")
    return problems


def check_dkim(record: str) -> list[str]:
    r = record.strip().strip('"').replace('" "', "")
    tags = dict(t.strip().split("=", 1) for t in r.split(";") if "=" in t)
    problems = []
    if tags.get("v", "DKIM1") != "DKIM1":
        problems.append("v= should be DKIM1")
    p = tags.get("p", None)
    if p is None:
        problems.append("no p= public key tag")
    elif p == "":
        problems.append("p= is empty, which means the key has been revoked")
    else:
        try:
            base64.b64decode(p + "=" * (-len(p) % 4), validate=True)
        except Exception:
            problems.append("p= isn't valid base64")
    return problems


def check_dmarc(record: str) -> list[str]:
    r = record.strip().strip('"')
    if not r.startswith("v=DMARC1"):
        return ["DMARC must start with v=DMARC1"]
    tags = dict(t.strip().split("=", 1) for t in r.split(";") if "=" in t)
    return [] if tags.get("p") in ("none", "quarantine", "reject") else ["p= must be none, quarantine or reject"]


def dmarc_warnings(record: str) -> list[str]:
    """Not failures, but worth fixing."""
    tags = dict(t.strip().split("=", 1) for t in record.strip().strip('"').split(";") if "=" in t)
    return [] if "rua" in tags else ["no rua= address, so you'll never see the reports (Yahoo strongly recommends one)"]


def check_sending_setup(spf: str, dkim: str, dmarc: str) -> dict:
    out = {name: {"ok": not probs, "problems": probs}
           for name, probs in [("spf", check_spf(spf)), ("dkim", check_dkim(dkim)), ("dmarc", check_dmarc(dmarc))]}
    out["dmarc"]["warnings"] = dmarc_warnings(dmarc)
    out["ok"] = all(out[k]["ok"] for k in ("spf", "dkim", "dmarc"))
    return out


def lookup_txt(name: str) -> list[str] | None:
    """TXT records for a DNS name via `dig`, or None if dig isn't installed. No new dependency."""
    if not shutil.which("dig"):
        return None
    res = subprocess.run(["dig", "+short", "TXT", name], capture_output=True, text=True, timeout=15)
    return [line.replace('" "', "").strip('"') for line in res.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------- outcomes and results

def simulate_outcomes(groups: dict, scores: dict, *, seed: int = 12, holdout_reply: float = 0.04,
                      treatment_reply: float = 0.06, meeting_given_reply: float = 0.4,
                      score_weight: float = 0.5) -> pd.DataFrame:
    """SIMULATED replies and meetings, for exercising the results code only.

    Every rate here is an assumption you pass in, not a measurement. Better-scored accounts get
    a little more likely to reply (score_weight), in both groups, so the comparison stays fair."""
    rng = np.random.default_rng(seed)
    rows = []
    mean_score = float(np.mean(list(scores.values()))) or 1.0
    for i, g in sorted(groups.items()):
        base = treatment_reply if g == "treatment" else holdout_reply
        p = min(0.95, base * (1 - score_weight + score_weight * scores[i] / mean_score))
        replied = bool(rng.random() < p)
        meeting = bool(replied and rng.random() < meeting_given_reply)
        rows.append({"lead_id": i, "group": g, "replied": replied, "meeting": meeting, "simulated": True})
    return pd.DataFrame(rows)


def results(outcomes: pd.DataFrame, *, metrics=("replied", "meeting"), n_boot: int = 2000, seed: int = 0) -> dict:
    """Treatment vs holdout for each metric: n, count and rate per group, the difference in
    rates, and a 90% bootstrap range on that difference. If the range includes zero, you
    haven't shown an effect yet, however good the treatment number looks."""
    rng = np.random.default_rng(seed)
    t = outcomes[outcomes.group == "treatment"]
    h = outcomes[outcomes.group == "holdout"]
    report = {"n_treatment": len(t), "n_holdout": len(h),
              "simulated": bool(outcomes.get("simulated", pd.Series([False])).any()), "metrics": {}}
    for m in metrics:
        tv, hv = t[m].to_numpy(dtype=float), h[m].to_numpy(dtype=float)
        diffs = [rng.choice(tv, len(tv)).mean() - rng.choice(hv, len(hv)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(diffs, [5, 95])
        report["metrics"][m] = {"treatment": {"count": int(tv.sum()), "rate": float(tv.mean())},
                                "holdout": {"count": int(hv.sum()), "rate": float(hv.mean())},
                                "difference": float(tv.mean() - hv.mean()), "range_90": [float(lo), float(hi)],
                                "shown": bool(lo > 0)}
    return report


def results_table(report: dict) -> pd.DataFrame:
    rows = []
    for m, r in report["metrics"].items():
        rows.append({"metric": m,
                     f"treatment (n={report['n_treatment']})": f"{r['treatment']['count']} ({r['treatment']['rate']:.1%})",
                     f"holdout (n={report['n_holdout']})": f"{r['holdout']['count']} ({r['holdout']['rate']:.1%})",
                     "difference": f"{r['difference']:+.1%}",
                     "90% range": f"{r['range_90'][0]:+.1%} to {r['range_90'][1]:+.1%}",
                     "effect shown": "yes" if r["shown"] else "not yet"})
    return pd.DataFrame(rows)
