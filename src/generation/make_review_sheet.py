"""Build a CSV of the tickets you need to read, and later summarize your verdicts.

Make the sheet:
  python src/generation/make_review_sheet.py --base data/tickets/synthetic/final_v1 --out data/audits/review_final_v1.csv

Open the CSV in Excel, read each row, and fill the 'verdict' column with one of:
  OK           label and text are fine
  WRONG_LABEL  text is fine but the label is wrong (write the right one in 'correct_label')
  BAD_TEXT     text is unrealistic, echoes the prompt, or contradicts itself
  DROP         ticket should be removed

Summarize after you have filled it in:
  python src/generation/make_review_sheet.py --summary data/audits/review_final_v1.csv

Groups in the sheet:
  A_rejected  tickets the fit audit rejected (read all)
  B_ask_info  every kept ask_info ticket (read all, there are few)
  C_flagged   kept tickets the audit had a note about
  D_random    a random sample of ALL kept tickets (this one gives the error rate for your thesis)
"""
import argparse, csv, json, math, os, random
from collections import Counter, defaultdict

COLS = ["group", "ticket_id", "split", "scenario", "subject", "body", "expected_action", "reason",
        "runbook_or_target", "candidates", "followup_reply", "judge_note", "verdict", "correct_label", "notes"]


def read(path):
    if not os.path.exists(path):
        return []
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def row(group, x, l, note=""):
    return {"group": group, "ticket_id": x["ticket_id"], "split": l.get("split", ""), "scenario": l.get("scenario", ""),
            "subject": x["subject"], "body": x["body"], "expected_action": l["expected_action"],
            "reason": l.get("escalation_reason") or l.get("info_reason") or "",
            "runbook_or_target": l.get("runbook_id") or l.get("target_runbook_id") or "",
            "candidates": " ".join(l.get("candidate_runbooks") or []),
            "followup_reply": l.get("followup_reply") or "", "judge_note": note,
            "verdict": "", "correct_label": "", "notes": ""}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - m) / d), min(1, (c + m) / d)


def build(a):
    inputs = {t["ticket_id"]: t for t in read(a.base + "_input.jsonl")}
    labels = {t["ticket_id"]: t for t in read(a.base + "_labels.jsonl")}
    rejected = read(a.base + "_rejected.jsonl")
    rng = random.Random(a.seed)
    kept = [t for t in inputs if t in labels]

    # D: stratified random sample of all kept tickets
    by_sc = defaultdict(list)
    for t in kept:
        by_sc[labels[t]["scenario"]].append(t)
    d_ids = set()
    for sc, ids in by_sc.items():
        n = max(1, round(a.sample * len(ids)))
        d_ids.update(rng.sample(ids, min(n, len(ids))))

    rows = [row("A_rejected", r["input"], r["labels"], r["reason"]) for r in rejected]
    seen = set()
    for t in kept:
        l = labels[t]
        if t in d_ids:
            continue
        if l["expected_action"] == "ask_info":
            rows.append(row("B_ask_info", inputs[t], l, l.get("fit_note", "")))
        elif l.get("fit_note"):
            rows.append(row("C_flagged", inputs[t], l, l["fit_note"]))
    rows += [row("D_random", inputs[t], labels[t], labels[t].get("fit_note", "")) for t in sorted(d_ids)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["group"] for r in rows)
    print(f"wrote {len(rows)} rows to {a.out}: {dict(sorted(c.items()))}")
    print("Fill the 'verdict' column with OK, WRONG_LABEL, BAD_TEXT or DROP.")


def summary(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    done = [r for r in rows if r["verdict"].strip()]
    print(f"{len(done)}/{len(rows)} rows have a verdict")
    for g in sorted({r["group"] for r in rows}):
        gr = [r for r in done if r["group"] == g]
        print(f"{g}: {dict(Counter(r['verdict'].strip().upper() for r in gr))}  (n={len(gr)})")
    d = [r for r in done if r["group"] == "D_random"]
    if d:
        bad = sum(r["verdict"].strip().upper() != "OK" for r in d)
        lo, hi = wilson(bad, len(d))
        print(f"\nRandom sample: {bad}/{len(d)} tickets have a problem = {100 * bad / len(d):.0f}% "
              f"(95% interval {100 * lo:.0f}% to {100 * hi:.0f}%). This is the label/text error rate to report.")
        for sc in sorted({r['scenario'] for r in d}):
            s = [r for r in d if r["scenario"] == sc]
            print(f"  {sc}: {sum(r['verdict'].strip().upper() != 'OK' for r in s)}/{len(s)} problems")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--sample", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    elif a.base and a.out:
        build(a)
    else:
        ap.error("use --base and --out to build, or --summary to summarize")
