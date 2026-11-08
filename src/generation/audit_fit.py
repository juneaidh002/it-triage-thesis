"""Check that each generated ticket really fits its label.

An LLM sees ONLY the ticket text (subject and body), never the labels, and picks the runbook it fits best.
Its answer is compared with the label. Disagreements are candidates for manual review or removal.

Usage: python src/generation/audit_fit.py data/tickets/synthetic/tickets_seed1_n20 [--role generation]
Writes <base>_fit.jsonl. Use a model different from the generator if you can (--role agent or judge).
"""
import argparse, glob, json, os, re, sys
from collections import Counter
import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.llm import load_models, cached_call
from ticket_rules import MULTI_USER_OK

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYSTEM = ("You are an IT service desk triager. You are given a support ticket and a list of runbooks. "
          "Choose the runbook that best matches the problem. Answer 'none' if no runbook covers the request, "
          "and 'unclear' if the ticket does not say enough to choose. Also say whether the ticket indicates that several people are affected, "
          "and how confident you are ('high', or 'low' if you are guessing). "
          'Reply with JSON only: {"best_runbook": "RB-XXX-00 or none or unclear", "multi_user": true or false, "confidence": "high or low"}')


def read(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def parse(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    d = json.loads(m.group(0) if m else t)
    return str(d["best_runbook"]).strip(), bool(d.get("multi_user", False)), str(d.get("confidence", "high")).strip().lower()


def verdict(lab, pred, multi, valid=None, conf="high"):
    """Returns (ok, note). The judge sees only the ticket text."""
    a, why = lab["expected_action"], lab.get("escalation_reason") or lab.get("info_reason")
    if valid is not None and pred not in valid and pred not in ("none", "unclear"):
        return False, "judge returned an unknown runbook id"
    if why in ("out_of_policy", "needs_approval"):
        return True, "not checked: whether a request is out of policy is a policy decision, not a text match"
    if a == "resolve":
        multi_bad = multi and lab["runbook_id"] not in MULTI_USER_OK
        return (pred == lab["runbook_id"] and not multi_bad), ("" if pred == lab["runbook_id"] else "judge picked another runbook")
    if a == "ask_info":
        if pred == "unclear" or conf == "low":     # a low-confidence guess means the ticket really is unclear
            return True, ""
        if why == "ambiguous_between_runbooks":
            if pred == lab.get("target_runbook_id"):
                return True, "judge already pinned the target (ticket may be clearer than intended)"
            if pred in (lab.get("candidate_runbooks") or []):
                return False, "ticket leans to a different runbook than the follow-up reply"
        return False, ("judge found no matching runbook" if pred == "none" else "judge committed to a runbook (ticket may not be vague enough)")
    if why == "no_runbook":
        if pred == "none" or conf == "low" and pred != "unclear":
            return True, ""
        return False, ("ticket too unclear to tell it is outside the runbooks" if pred == "unclear" else "judge matched a runbook")
    return pred == lab["runbook_id"], ("" if pred == lab["runbook_id"] else "underlying problem not recognisable")   # runbook rules and injection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--role", default="generation")
    ap.add_argument("--models", default=None)
    ap.add_argument("--sample", type=float, default=1.0, help="audit only this share of tickets (saves tokens)")
    a = ap.parse_args()
    models = load_models(a.models)
    rbs = [yaml.safe_load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{ROOT}/data/runbooks/*.yaml"))]
    valid_ids = {r["id"] for r in rbs}
    catalog = "\n".join(f"{r['id']}: {r['title']} - {r['applies_when']}" for r in rbs)
    inputs = {t["ticket_id"]: t for t in read(a.base + "_input.jsonl")}
    labels = {t["ticket_id"]: t for t in read(a.base + "_labels.jsonl")}
    out, bad = [], []
    import random
    rng = random.Random(0)
    for tid, x in inputs.items():
        if a.sample < 1.0 and rng.random() > a.sample:
            continue
        user = f"Runbooks:\n{catalog}\n\nTicket subject: {x['subject'] or '(empty)'}\nTicket body: {x['body'] or '(empty)'}"
        pred, multi, conf, err = "unclear", False, "high", None
        for attempt in range(3):
            try:
                pred, multi, conf = parse(cached_call(models, a.role, SYSTEM, user, refresh=attempt > 0))
                err = None
                break
            except Exception as e:
                err = str(e)
        ok, note = verdict(labels[tid], pred, multi, valid_ids, conf)
        ok = ok and err is None
        # Second check for ask_info tickets: does the simulated reply really lead to the target runbook?
        fu_ok, fu_pred = None, None
        lab = labels[tid]
        if lab["expected_action"] == "ask_info" and lab.get("followup_reply") and lab.get("target_runbook_id"):
            user2 = user + f"\nThe employee then replied: {lab['followup_reply']}"
            try:
                fu_pred, _, _ = parse(cached_call(models, a.role, SYSTEM, user2))
            except Exception:
                fu_pred = "error"
            fu_ok = fu_pred == lab["target_runbook_id"]
        out.append(dict(ticket_id=tid, predicted_runbook=pred, predicted_multi_user=multi, confidence=conf, fit_ok=ok, fit_note=note, followup_ok=fu_ok, followup_pred=fu_pred, error=err))
        if not ok or (note and not note.startswith("not checked")):
            bad.append((tid, labels[tid]["expected_action"], labels[tid]["runbook_id"] or labels[tid].get("info_reason") or labels[tid]["escalation_reason"], pred, multi, note, ok))
    with open(a.base + "_fit.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    by = {}
    for o in out:
        sc = labels[o["ticket_id"]].get("scenario", "?")
        by.setdefault(sc, [0, 0])
        by[sc][1] += 1
        by[sc][0] += bool(o["fit_ok"])
    n_ok = sum(o["fit_ok"] for o in out)
    print(f"{n_ok}/{len(out)} tickets fit their label ({100 * n_ok / max(1, len(out)):.0f}%)")
    fu = [o for o in out if o.get("followup_ok") is not None]
    if fu:
        print(f"follow-up replies consistent with their target: {sum(o['followup_ok'] for o in fu)}/{len(fu)}")
        for o in fu:
            if not o["followup_ok"]:
                print(f"FOLLOWUP {o['ticket_id']}: ticket plus reply read as {o['followup_pred']}, target was {labels[o['ticket_id']]['target_runbook_id']}")
    print("by scenario:", ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by.items())))
    for b in bad:
        print(f"{'CHECK' if not b[6] else 'NOTE '} {b[0]}: label={b[1]}/{b[2]}  judge said {b[3]}  multi_user={b[4]}  {b[5]}")


if __name__ == "__main__":
    main()
