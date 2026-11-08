"""Apply the second-review decisions to final_v1 and write final_v2. final_v1 is never modified.

Usage (from the project folder):
  python src/generation/apply_review.py --base data/tickets/synthetic/final_v1 --out data/tickets/synthetic/final_v2 [--scrub-banned] \
      --review-xlsx data/audits/ticket_audit_with_comments.xlsx \
      --fits data/tickets/synthetic/tickets_seed11_n100_fit.jsonl data/tickets/synthetic/tickets_seed12_n150_fit.jsonl

What it does
  1. Starts from the kept tickets in final_v1 and REINSTATES the tickets the fit audit rejected (the second review
     found none of them unusable). Near-duplicates stay out.
  2. Applies the decisions listed below.
  3. Gives every ticket an `acceptable_actions` list (just the primary action unless a decision widens it).
  4. Renames the reason `out_of_policy` to `needs_approval` (privileged or restricted requests need approval,
     they are not necessarily prohibited).
  5. Re-splits dev/test, stratified by action and reason (nothing has been tuned on the old split yet).
  6. Writes final_v2_input.jsonl, final_v2_labels.jsonl and final_v2_changes.csv (every change, for the thesis).
Then check it:  python src/generation/validate_tickets.py data/tickets/synthetic/final_v2
"""
import argparse, csv, glob, json, os, random, re, sys, warnings
from collections import Counter, defaultdict
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ticket_rules import (priority_rule, banned_hits, BANNED_WORDS, MULTI_USER_OK, AMBIGUOUS_CATEGORY,
                          NEEDS_APPROVAL_CATEGORY, OUT_OF_POLICY_CATEGORY)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------- decisions (from the second review)
# Tickets labelled resolve that say several people are affected: they are incidents, so escalate.
MULTI_USER_TO_ESCALATE = {            # ticket_id: users_affected to record
    "T-S11-0077": "many",             # whole floor, laptops not charging
    "T-S11-0078": "few",              # payroll software unlicensed on several machines
    "T-S12-0062": "many",             # update loop on most lab machines
    "T-S12-0063": "few",              # licence error for the user and three others
    "T-S12-0091": "many",             # whole floor network slow
    "T-S12-0148": "few",              # update loop, a colleague has the same problem
    "T-S12-0139": "many",             # warehouse system timing out, "the rest of the floor" has the same problem
}
# ask_info tickets that are clear enough to act on: primary action becomes resolve (the old target runbook).
PINNED_TO_RESOLVE = ["T-S11-0100", "T-S12-0030", "T-S12-0111", "T-S12-0140", "T-S12-0142"]
# Resolve ticket that really is ambiguous between runbooks: ask for the missing detail.
TO_ASK_INFO = {"T-S12-0053": ["RB-APP-01", "RB-ACC-01", "RB-PWD-01", "RB-PWD-02"]}
# Gray cases where both actions are defensible: the primary label stays, the acceptable set is widened.
BORDERLINE = {
    **{t: ["resolve", "ask_info"] for t in [
        "T-S11-0011", "T-S11-0070", "T-S11-0083", "T-S11-0098", "T-S12-0015",
        "T-S12-0043", "T-S12-0077", "T-S12-0130", "T-S12-0138"]},
    "T-S12-0101": ["resolve", "escalate"],   # update stuck after a forced shutdown
    "T-S11-0019": ["ask_info", "resolve"],   # "audio glitchy during calls" is not very vague
}
# Follow-up replies that leave out a fact the target runbook needs (printer name or location).
DROP_FOLLOWUP = ["T-S12-0023", "T-S12-0079"]
# Tickets to keep but flag, so results can also be reported without them.
WEAK = {"T-S11-0019": "not very vague", "T-S11-0080": "text does not clearly point at the VPN",
        "T-S11-0099": "overlaps the hardware runbooks"}
# -------------------------------------------------------------------------------------------------


def read(path):
    if not os.path.exists(path):
        return []
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="final_v1 base path (without _input.jsonl)")
    ap.add_argument("--out", required=True, help="final_v2 base path")
    ap.add_argument("--review-xlsx", default=None, help="the reviewed workbook; adds second_review_* fields")
    ap.add_argument("--fits", nargs="*", default=[], help="batch _fit.jsonl files, used only to list unreviewed multi-user tickets")
    ap.add_argument("--scrub-banned", action="store_true",
                    help="remove priority words (asap, urgent...) left in ticket text; every edit is logged in the changes file")
    ap.add_argument("--keep-split", action="store_true", help="keep the v1 dev/test split instead of re-splitting")
    ap.add_argument("--dev-share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rbs = {}
    for f in glob.glob(f"{ROOT}/data/runbooks/*.yaml"):
        r = yaml.safe_load(open(f, encoding="utf-8"))
        rbs[r["id"]] = r

    inputs = {t["ticket_id"]: t for t in read(a.base + "_input.jsonl")}
    labels = {t["ticket_id"]: dict(t) for t in read(a.base + "_labels.jsonl")}
    origin = {t: "kept" for t in inputs}
    skipped = []
    for r in read(a.base + "_rejected.jsonl"):
        tid = r["input"]["ticket_id"]
        if r["reason"].startswith("near-duplicate"):
            skipped.append((tid, r["reason"]))
            continue
        inputs[tid] = r["input"]
        l = dict(r["labels"])
        l["fit_ok"] = False
        l["fit_note"] = r["reason"].split(": ", 1)[-1]
        labels[tid] = l
        origin[tid] = "reinstated"

    second = {}
    if a.review_xlsx:
        import pandas as pd
        warnings.filterwarnings("ignore")
        df = pd.read_excel(a.review_xlsx, sheet_name="Ticket Audit")
        df = df.astype(object).where(df.notna(), "")
        second = {r.ticket_id: (r.reviewed_action, r.review_outcome) for r in df.itertuples()}

    changes = []
    def log(tid, change, detail):
        changes.append({"ticket_id": tid, "change": change, "detail": detail})

    for tid, l in labels.items():
        l.setdefault("fit_ok", None)
        l["acceptable_actions"] = [l["expected_action"]]
        l["flags"] = []
        l["label_version"] = "v2"
        if tid in second:
            l["second_review_action"], l["second_review_outcome"] = second[tid]
        if origin[tid] == "reinstated":
            log(tid, "reinstated", "rejected by the fit audit in v1; reinstated because the second review dropped none of them")

    def need(ids, what):
        missing = [t for t in ids if t not in labels]
        if missing:
            print(f"WARNING: {what}: ticket ids not found: {missing}")
        return [t for t in ids if t in labels]

    for tid in need(MULTI_USER_TO_ESCALATE, "multi-user"):
        l = labels[tid]
        old = f"{l['expected_action']}/{l.get('escalation_reason') or l.get('info_reason')}"
        l.update(expected_action="escalate", escalation_reason="multi_user_incident", info_reason=None,
                 candidate_runbooks=[], target_runbook_id=None, followup_reply=None,
                 users_affected=MULTI_USER_TO_ESCALATE[tid], acceptable_actions=["escalate"])
        l["priority"] = priority_rule(l["true_category"], l["blocking"], l["users_affected"])
        log(tid, "multi_user_to_escalate", f"{old} -> escalate/multi_user_incident, users_affected={l['users_affected']}")

    for tid in need(PINNED_TO_RESOLVE, "pinned"):
        l = labels[tid]
        tgt = l.get("target_runbook_id")
        if not tgt or tgt not in rbs:
            print(f"WARNING: {tid} has no usable target runbook, left unchanged")
            continue
        l.update(expected_action="resolve", runbook_id=tgt, true_category=rbs[tgt]["category"], escalation_reason=None,
                 info_reason=None, candidate_runbooks=[], target_runbook_id=None, followup_reply=None,
                 acceptable_actions=["resolve", "ask_info"])
        l["priority"] = priority_rule(l["true_category"], l["blocking"], l["users_affected"])
        log(tid, "ask_info_to_resolve", f"ask_info -> resolve with {tgt}; ask_info stays acceptable")

    for tid in need(TO_ASK_INFO, "to ask_info"):
        l = labels[tid]
        tgt = l.get("runbook_id")
        l.update(expected_action="ask_info", runbook_id=None, true_category=AMBIGUOUS_CATEGORY, escalation_reason=None,
                 info_reason="ambiguous_between_runbooks", candidate_runbooks=TO_ASK_INFO[tid], target_runbook_id=tgt,
                 followup_reply=None, acceptable_actions=["ask_info"])
        l["priority"] = priority_rule(l["true_category"], l["blocking"], l["users_affected"])
        log(tid, "resolve_to_ask_info", f"resolve/{tgt} -> ask_info between {' '.join(TO_ASK_INFO[tid])}; no follow-up stored")

    for tid in need(BORDERLINE, "borderline"):
        labels[tid]["acceptable_actions"] = BORDERLINE[tid]
        log(tid, "acceptable_widened", f"primary {labels[tid]['expected_action']}, acceptable {BORDERLINE[tid]}")

    for tid in need(DROP_FOLLOWUP, "follow-up"):
        labels[tid]["followup_reply"] = None
        labels[tid]["followup_dropped"] = True
        log(tid, "followup_dropped", "reply omitted a fact the target runbook needs")

    for tid in need(WEAK, "weak"):
        labels[tid]["flags"].append("weak_label")
        log(tid, "flag_weak_label", WEAK[tid])

    if a.scrub_banned:
        for tid, x in inputs.items():
            old = (x["subject"], x["body"])
            hits = banned_hits(old[0] + " " + old[1])
            if not hits:
                continue
            for field in ("subject", "body"):
                t = x[field]
                for w in BANNED_WORDS:
                    t = re.sub(r"\s*\b" + re.escape(w) + r"\b", "", t, flags=re.I)
                t = re.sub(r"\s+([,.!?])", r"\1", t)
                t = re.sub(r"[ \t]{2,}", " ", t).strip()
                x[field] = t
            log(tid, "text_scrubbed", f"removed {hits}. before: {old[1][:120]!r} after: {x['body'][:120]!r}")

    n_ren = 0
    for tid, l in labels.items():
        if l.get("escalation_reason") == "out_of_policy":
            l["escalation_reason"] = "needs_approval"
            n_ren += 1
        if l.get("true_category") == OUT_OF_POLICY_CATEGORY:
            l["true_category"] = NEEDS_APPROVAL_CATEGORY
    if n_ren:
        log("(all)", "renamed_reason", f"out_of_policy -> needs_approval on {n_ren} tickets")

    if not a.keep_split:
        rng = random.Random(a.seed)
        groups = defaultdict(list)
        for tid, l in labels.items():
            groups[(l["expected_action"], l.get("escalation_reason") or l.get("info_reason") or "resolve")].append(tid)
        for key in sorted(groups):
            ids = sorted(groups[key])
            rng.shuffle(ids)
            cut = round(len(ids) * a.dev_share)
            for i, tid in enumerate(ids):
                labels[tid]["split"] = "dev" if i < cut else "test"

    order = sorted(labels)
    with open(a.out + "_input.jsonl", "w", encoding="utf-8") as fi, open(a.out + "_labels.jsonl", "w", encoding="utf-8") as fl:
        for tid in order:
            fi.write(json.dumps(inputs[tid], ensure_ascii=False) + "\n")
            fl.write(json.dumps(labels[tid], ensure_ascii=False) + "\n")
    with open(a.out + "_changes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["ticket_id", "change", "detail"])
        w.writeheader()
        w.writerows(changes)

    L = list(labels.values())
    print(f"final_v2: {len(L)} tickets ({sum(o == 'kept' for o in origin.values())} kept from v1, "
          f"{sum(o == 'reinstated' for o in origin.values())} reinstated); not reinstated: {skipped or 'none'}")
    print("changes:", dict(Counter(c["change"] for c in changes)))
    print("expected_action:", dict(Counter(l["expected_action"] for l in L)))
    print("acceptable_actions:", dict(Counter(" / ".join(l["acceptable_actions"]) for l in L)))
    print("escalation_reason:", dict(Counter(l["escalation_reason"] for l in L if l["escalation_reason"])))
    print("split:", dict(Counter(l["split"] for l in L)))

    # Resolve tickets the audit judge thought were multi-user but nobody reviewed or relabelled: check by hand.
    sus = []
    for fp in a.fits:
        for f in read(fp):
            tid = f["ticket_id"]
            l = labels.get(tid)
            if (l and f.get("predicted_multi_user") and l["expected_action"] == "resolve"
                    and l.get("runbook_id") not in MULTI_USER_OK and tid not in MULTI_USER_TO_ESCALATE):
                sus.append(tid)
    if sus:
        print(f"\nCHECK BY HAND: {len(sus)} resolve tickets the audit judge read as multi-user and no decision covers: {sorted(sus)}")


if __name__ == "__main__":
    main()