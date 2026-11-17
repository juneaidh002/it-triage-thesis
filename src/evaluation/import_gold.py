"""Turn the filled gold-set workbook into the same two-file format as the synthetic benchmark.

  python src/generation/import_gold.py --xlsx data/audits/gold_set_v1.xlsx --out data/tickets/gold/gold_v1
  python src/generation/validate_tickets.py data/tickets/gold/gold_v1

Writes gold_v1_input.jsonl (ticket id, subject, body: all the agent may see), gold_v1_labels.jsonl (answer key) and
gold_v1_import_notes.csv (every problem or choice that needs a look). Labels come from the assignment in the sheet.
If the author chose a different 'final_action', the assigned label is kept and the choice is stored as
`author_suggested_action`, so the change can be reviewed before it is applied. Re-run the import after fixing the sheet.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

import openpyxl
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from ticket_rules import (priority_rule, banned_hits, NO_RUNBOOK_CATEGORY, AMBIGUOUS_CATEGORY, NEEDS_APPROVAL_CATEGORY)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENARIO = {"resolve": "resolve", "escalate_always": "runbook_escalate", "escalate_rule": "escalate_condition",
            "multi_user": "multi_user_incident", "no_runbook": "no_runbook", "needs_approval": "needs_approval",
            "injection": "injection", "ask_vague": "vague", "ask_confusable": "confusable"}
STYLE_KEY = {"A short, specific subject.": "good", "Leave the subject empty.": "empty"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
DIGITS = re.compile(r"\d{4,}")


def text(v):
    return "" if v is None else str(v).strip()


def style_from_hint(hint, kind):
    h = text(hint)
    if kind == "subject":
        if h.startswith("A short"): return "good"
        if h.startswith("A generic"): return "generic"
        if h.startswith("Leave"): return "empty"
        if h.startswith("A subject about"): return "mismatched"
    else:
        if h.startswith("A normal"): return "normal"
        if h.startswith("One short"): return "one_liner"
        if h.startswith("Mostly an error"): return "error_only"
        if h.startswith("A long"): return "rambling"
        if h.startswith("Leave"): return "empty"
    return "good" if kind == "subject" else "normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runbooks", default=f"{ROOT}/data/runbooks")
    a = ap.parse_args()

    cats = {}
    for f in glob.glob(os.path.join(a.runbooks, "*.yaml")):
        with open(f, encoding="utf-8") as fh:
            r = yaml.safe_load(fh)
        cats[r["id"]] = r["category"]

    ws = openpyxl.load_workbook(a.xlsx, data_only=False)["Tickets"]
    head = [text(c.value) for c in ws[1]]
    col = {h: i for i, h in enumerate(head)}
    need = ["gold_id", "assignment_type", "subject", "body", "followup_reply", "expected_action", "escalation_reason",
            "info_reason", "runbook_id", "candidate_runbooks", "target_runbook_id", "users_affected"]
    missing = [h for h in need if h not in col]
    if missing:
        sys.exit(f"The Tickets sheet is missing columns: {missing}. Do not rename or delete columns.")

    inputs, labels, notes = [], [], []
    def note(tid, level, msg):
        notes.append(dict(ticket_id=tid, level=level, note=msg))

    skipped = Counter()
    for row in ws.iter_rows(min_row=2, values_only=True):
        g = lambda name: row[col[name]] if name in col and col[name] < len(row) else None
        tid, typ = text(g("gold_id")), text(g("assignment_type"))
        if not tid:
            continue
        subject, body, fu = text(g("subject")), text(g("body")), text(g("followup_reply"))
        final = text(g("final_action"))
        if final == "drop":
            skipped["marked drop"] += 1
            note(tid, "info", "marked drop by the author")
            continue
        if not subject and not body:
            skipped["not written yet"] += 1
            continue
        hits = banned_hits(subject + " " + body + " " + fu)
        if hits:
            note(tid, "ERROR", f"contains priority words {hits}; rewrite them in the sheet")
        expected = text(g("expected_action"))
        reason, info = text(g("escalation_reason")) or None, text(g("info_reason")) or None
        rid = text(g("runbook_id")) or None
        cands = text(g("candidate_runbooks")).split()
        target = text(g("target_runbook_id")) or None
        users = text(g("users_affected")) or "one"
        if rid and rid not in cats:
            note(tid, "ERROR", f"runbook {rid} not found in {a.runbooks}")
        if expected == "ask_info":
            category = AMBIGUOUS_CATEGORY
            if not fu:
                note(tid, "WARNING", "ask_info ticket has no follow-up reply")
        elif rid:
            category = cats.get(rid, "")
        elif reason == "needs_approval":
            category = NEEDS_APPROVAL_CATEGORY
        else:
            category = NO_RUNBOOK_CATEGORY
        pii = text(g("include personal data"))
        has_pii = pii.lower().startswith("yes")
        found = bool(EMAIL.search(subject + " " + body)) or bool(DIGITS.search(subject + " " + body))
        if has_pii and not found:
            note(tid, "WARNING", "marked for made-up personal data but no email, phone or id pattern found (names cannot be checked)")
        if not has_pii and found:
            note(tid, "WARNING", "a long number or email address appears in a row not marked for personal data")
        chk = text(g("self_check"))
        if chk in ("unsure", "does not fit"):
            note(tid, "REVIEW", f"author self-check: {chk}")
        if final and final != expected and final in ("resolve", "ask_info", "escalate"):
            note(tid, "REVIEW", f"author suggests {final} instead of the assigned {expected}; label kept, decide later")
        s_style = "empty" if not subject else style_from_hint(g("suggested subject style"), "subject")
        b_style = "empty" if not body else style_from_hint(g("suggested body style"), "body")
        if subject and s_style == "empty":
            s_style = "good"
        if body and b_style == "empty":
            b_style = "normal"
        inputs.append(dict(ticket_id=tid, subject=subject, body=body))
        labels.append(dict(
            ticket_id=tid, subject_style=s_style, body_style=b_style, persona="author", tone="", clarity="", language="en",
            scenario=SCENARIO.get(typ, typ), true_category=category, runbook_id=rid if expected != "ask_info" else None,
            expected_action=expected, escalation_reason=reason if expected == "escalate" else None,
            info_reason=info if expected == "ask_info" else None,
            candidate_runbooks=cands if expected == "ask_info" else [], target_runbook_id=target if expected == "ask_info" else None,
            followup_reply=fu or None if expected == "ask_info" else None,
            blocking=True, users_affected=users, priority=priority_rule(category, True, users),
            source="gold", split="gold", acceptable_actions=[expected], contains_pii=has_pii, pii_types=pii if has_pii else "",
            assigned_type=typ, author_self_check=chk or None,
            author_suggested_action=final if final and final != expected and final in ("resolve", "ask_info", "escalate") else None,
            flags=[], label_version="gold_v1"))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_input.jsonl", "w", encoding="utf-8") as fi, open(a.out + "_labels.jsonl", "w", encoding="utf-8") as fl:
        for x, l in zip(inputs, labels):
            fi.write(json.dumps(x, ensure_ascii=False) + "\n")
            fl.write(json.dumps(l, ensure_ascii=False) + "\n")
    with open(a.out + "_import_notes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["ticket_id", "level", "note"])
        w.writeheader()
        w.writerows(notes)

    lv = Counter(n["level"] for n in notes)
    print(f"imported {len(inputs)} tickets; skipped: {dict(skipped) or 'none'}")
    print("by assignment type:", dict(Counter(l["assigned_type"] for l in labels)))
    print("expected_action:", dict(Counter(l["expected_action"] for l in labels)))
    print(f"with made-up personal data: {sum(l['contains_pii'] for l in labels)}")
    print(f"notes: {dict(lv)}  (details in {a.out}_import_notes.csv)")
    for n in notes:
        if n["level"] == "ERROR":
            print(f"ERROR {n['ticket_id']}: {n['note']}")
    if lv.get("ERROR"):
        sys.exit(1)


if __name__ == "__main__":
    main()
