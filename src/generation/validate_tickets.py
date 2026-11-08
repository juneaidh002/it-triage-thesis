"""Validate a ticket set stored as two files: <base>_input.jsonl and <base>_labels.jsonl.

Usage: python src/generation/validate_tickets.py data/tickets/synthetic/pilot_20
Exit code 1 if any ERROR is found. WARNINGs need a human look.
"""
import glob, json, os, re, sys
from collections import Counter
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ticket_rules import (priority_rule, banned_hits, ESCALATION_REASONS, ACTIONS, INFO_REASONS, multi_user_categories, MULTI_USER_OK,
                          NO_RUNBOOK_CATEGORY, AMBIGUOUS_CATEGORY, OUT_OF_POLICY_CATEGORY, NEEDS_APPROVAL_CATEGORY, USERS_AFFECTED)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_KEYS = {"ticket_id", "subject", "body"}          # the agent may see ONLY these
LABEL_REQUIRED = ["ticket_id", "true_category", "runbook_id", "expected_action", "escalation_reason",
                  "blocking", "users_affected", "priority", "subject_style", "body_style", "source", "split"]


def shingles(text, n):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def read(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        print(f"NOTE: {os.path.basename(path)} is not UTF-8 (probably written by an older run on Windows). Re-run the generator to rewrite it as UTF-8.")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def main(base):
    runbooks = {}
    for f in glob.glob(f"{ROOT}/data/runbooks/*.yaml"):
        r = yaml.safe_load(open(f, encoding="utf-8"))
        runbooks[r["id"]] = r
    cats = {r["category"] for r in runbooks.values()} | {NO_RUNBOOK_CATEGORY, AMBIGUOUS_CATEGORY, OUT_OF_POLICY_CATEGORY, NEEDS_APPROVAL_CATEGORY}
    multi_cats = multi_user_categories(runbooks)
    inputs = {t["ticket_id"]: t for t in read(base + "_input.jsonl")}
    labels = {t["ticket_id"]: t for t in read(base + "_labels.jsonl")}
    errors, warns = [], []
    if not inputs and labels:
        errors.append("the _input file is empty or unreadable but the _labels file has rows; re-run the generator (cached responses make this free)")

    for tid in sorted(set(inputs) ^ set(labels)):
        errors.append(f"{tid}: present in only one of the two files")

    for tid, x in inputs.items():
        extra = set(x) - INPUT_KEYS
        if extra:
            errors.append(f"{tid}: input file has extra fields {sorted(extra)} (leak risk)")
        text = (x.get("subject", "") + " " + x.get("body", "")).strip()
        if not text:
            errors.append(f"{tid}: empty subject and body")
        if banned_hits(text):
            errors.append(f"{tid}: priority words in text {banned_hits(text)}")
        n = len(x.get("body", "").split())
        if n > 220:
            warns.append(f"{tid}: body is {n} words")

    for tid, t in labels.items():
        miss = [k for k in LABEL_REQUIRED if k not in t]
        if miss:
            errors.append(f"{tid}: missing label fields {miss}")
            continue
        if t["true_category"] not in cats:
            errors.append(f"{tid}: unknown category {t['true_category']!r}")
        if t["users_affected"] not in USERS_AFFECTED:
            errors.append(f"{tid}: bad users_affected")
        rb = runbooks.get(t["runbook_id"]) if t["runbook_id"] else None
        if t["runbook_id"] and not rb:
            errors.append(f"{tid}: runbook {t['runbook_id']} not found")
        if rb and rb["category"] != t["true_category"]:
            errors.append(f"{tid}: category does not match runbook")
        info_fields = [t.get("info_reason"), t.get("candidate_runbooks"), t.get("target_runbook_id"), t.get("followup_reply")]
        if t["expected_action"] == "ask_info":
            if t["runbook_id"] is not None:
                errors.append(f"{tid}: ask_info ticket must not have a runbook_id")
            if t["escalation_reason"] is not None:
                errors.append(f"{tid}: ask_info ticket has an escalation reason")
            if t.get("info_reason") not in INFO_REASONS:
                errors.append(f"{tid}: bad info_reason {t.get('info_reason')!r}")
            tgt = runbooks.get(t.get("target_runbook_id"))
            if not tgt or not tgt["auto_resolvable"]:
                errors.append(f"{tid}: target_runbook_id must be an auto-resolvable runbook")
            cands = t.get("candidate_runbooks") or []
            if t.get("info_reason") == "ambiguous_between_runbooks":
                if len(cands) < 2 or t.get("target_runbook_id") not in cands:
                    errors.append(f"{tid}: candidates must have 2+ runbooks including the target")
                if any(c not in runbooks for c in cands):
                    errors.append(f"{tid}: unknown runbook in candidates")
            if not t.get("followup_reply"):
                warns.append(f"{tid}: no follow-up reply stored")
            elif banned_hits(t["followup_reply"]):
                errors.append(f"{tid}: priority words in follow-up reply")
        elif any(info_fields):
            errors.append(f"{tid}: info fields set on a non-ask_info ticket")
        if t["expected_action"] == "resolve":
            if t["escalation_reason"] is not None:
                errors.append(f"{tid}: resolve ticket has an escalation reason")
            if not rb or not rb["auto_resolvable"]:
                errors.append(f"{tid}: resolve needs an auto-resolvable runbook")
        elif t["expected_action"] == "escalate":
            if t["escalation_reason"] not in ESCALATION_REASONS:
                errors.append(f"{tid}: bad escalation reason {t['escalation_reason']!r}")
            if t["escalation_reason"] == "runbook_always_escalate" and rb and rb["auto_resolvable"]:
                errors.append(f"{tid}: always-escalate reason on an auto-resolvable runbook")
        elif t["expected_action"] != "ask_info":
            errors.append(f"{tid}: expected_action must be one of {sorted(ACTIONS)}")
        if t["expected_action"] in ("resolve", "ask_info") and t["users_affected"] != "one":
            tcats = set()
            if rb:
                tcats = {rb["category"]}
            elif t.get("target_runbook_id") in runbooks:
                tcats = {runbooks[c]["category"] for c in (t.get("candidate_runbooks") or [t["target_runbook_id"]]) if c in runbooks}
            if (tcats & multi_cats) and not (rb and rb["id"] in MULTI_USER_OK):
                errors.append(f"{tid}: label conflict: multi-user impact on a category where several affected users means escalate")
        acc = t.get("acceptable_actions")
        if acc is not None and (not isinstance(acc, list) or t["expected_action"] not in acc or not set(acc) <= ACTIONS):
            errors.append(f"{tid}: acceptable_actions must be a list of valid actions that includes expected_action")
        if t["priority"] != priority_rule(t["true_category"], t["blocking"], t["users_affected"]):
            errors.append(f"{tid}: priority does not follow the rule table")
        x = inputs.get(tid)
        if x:
            if t["body_style"] == "empty" and x["body"]:
                errors.append(f"{tid}: body_style is empty but body has text")
            if t["subject_style"] == "empty" and x["subject"]:
                errors.append(f"{tid}: subject_style is empty but subject has text")
            if not x["body"] and not x["subject"]:
                pass
            if t["expected_action"] == "resolve" and t["subject_style"] in ("empty", "generic") and len(x["body"].split()) < 4:
                warns.append(f"{tid}: resolve ticket may be unidentifiable (generic subject, tiny body)")
            if rb:
                steps = " ".join(rb["steps"])
                overlap = shingles(x["subject"] + " " + x["body"], 5) & shingles(steps, 5)
                if overlap:
                    warns.append(f"{tid}: copies runbook step wording {sorted(overlap)[:2]}")

    ids = list(inputs)
    sh = [(i, shingles(inputs[i]["subject"] + " " + inputs[i]["body"], 3)) for i in ids]
    for a in range(len(sh)):
        for b in range(a + 1, len(sh)):
            x, y = sh[a][1], sh[b][1]
            if x and y and len(x & y) / len(x | y) > 0.6:
                warns.append(f"{sh[a][0]} and {sh[b][0]} look like near duplicates")

    L = list(labels.values())
    print(f"{len(inputs)} tickets")
    for k in ["expected_action", "escalation_reason", "info_reason", "priority", "subject_style", "body_style", "true_category"]:
        print(f"{k}:", dict(Counter(t.get(k) for t in L)))
    for w in warns:
        print("WARNING", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} errors, {len(warns)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1])
