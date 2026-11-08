"""Merge several generated batches into one cleaned dataset.

Usage:
  python src/generation/build_dataset.py data/tickets/synthetic/tickets_seed11_n200 data/tickets/synthetic/tickets_seed12_n200 \
      --out data/tickets/synthetic/final_v1 --resplit

Steps: merge -> drop near-duplicates -> drop tickets the fit audit rejected (if a _fit.jsonl exists) -> optional stratified dev/test split.
Writes <out>_input.jsonl, <out>_labels.jsonl and <out>_rejected.jsonl (with reasons, for manual review).
"""
import argparse, json, os, random, re
from collections import Counter, defaultdict


def read(path):
    if not os.path.exists(path):
        return []
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def shingles(text, n=3):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dup", type=float, default=0.6, help="Jaccard similarity above which two tickets count as duplicates")
    ap.add_argument("--keep-failed", action="store_true", help="keep tickets the fit audit rejected")
    ap.add_argument("--resplit", action="store_true", help="re-assign dev/test, stratified by action and scenario")
    ap.add_argument("--dev-share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for b in a.bases:
        inputs = {t["ticket_id"]: t for t in read(b + "_input.jsonl")}
        labels = {t["ticket_id"]: t for t in read(b + "_labels.jsonl")}
        fits = {t["ticket_id"]: t for t in read(b + "_fit.jsonl")}
        for tid in inputs:
            if tid in labels:
                rows.append((inputs[tid], labels[tid], fits.get(tid)))
    ids = [r[0]["ticket_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ticket_ids across batches: use different seeds for each batch")

    kept, rejected = [], []
    seen = []   # (id, shingles) of kept tickets
    for x, l, f in rows:
        if f and not f["fit_ok"] and not a.keep_failed:
            rejected.append((x, l, f"fit audit: {f.get('fit_note') or f.get('error') or 'rejected'}"))
            continue
        sh = shingles(x["subject"] + " " + x["body"])
        dup = next((i for i, s in seen if sh and s and len(sh & s) / len(sh | s) > a.dup), None)
        if dup:
            rejected.append((x, l, f"near-duplicate of {dup}"))
            continue
        seen.append((x["ticket_id"], sh))
        if f:
            l = dict(l, fit_ok=f["fit_ok"], fit_note=f.get("fit_note", ""), followup_ok=f.get("followup_ok"))
            if f.get("followup_ok") is False:      # keep the ticket for the first-turn decision, drop the unreliable follow-up
                l["followup_reply"] = None
                l["followup_dropped"] = True
        kept.append((x, l))

    if a.resplit:
        rng = random.Random(a.seed)
        groups = defaultdict(list)
        for x, l in kept:
            groups[(l["expected_action"], l["scenario"])].append(l)
        for g in groups.values():
            rng.shuffle(g)
            cut = round(len(g) * a.dev_share)
            for i, l in enumerate(g):
                l["split"] = "dev" if i < cut else "test"

    with open(a.out + "_input.jsonl", "w", encoding="utf-8") as fi, open(a.out + "_labels.jsonl", "w", encoding="utf-8") as fl:
        for x, l in kept:
            fi.write(json.dumps(x, ensure_ascii=False) + "\n")
            fl.write(json.dumps(l, ensure_ascii=False) + "\n")
    with open(a.out + "_rejected.jsonl", "w", encoding="utf-8") as fr:
        for x, l, why in rejected:
            fr.write(json.dumps(dict(reason=why, input=x, labels=l), ensure_ascii=False) + "\n")

    L = [l for _, l in kept]
    print(f"read {len(rows)} | kept {len(kept)} | rejected {len(rejected)}")
    print("NOTE: do not discard rejected tickets blindly. Dropping the ones the judge found hard makes the benchmark easier.")
    print("      Read *_rejected.jsonl, fix wrong labels, and drop only tickets that are really mislabeled or broken.")
    print("rejected reasons:", dict(Counter(r[2].split(":")[0].split(" of ")[0] for r in rejected)))
    for k in ["expected_action", "scenario", "split"]:
        print(f"{k}:", dict(Counter(l.get(k) for l in L)))


if __name__ == "__main__":
    main()
