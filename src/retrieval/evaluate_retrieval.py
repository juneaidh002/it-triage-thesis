"""Measure how often retrieval finds the right runbook for a ticket.

  python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2                 # dev tickets only (default)
  python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2 --tag aliases   # name the saved result file
  python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2 --split test --final

The query is built from the ticket's subject and body only (see make_query). The labels are used only for scoring.
The test split is protected: it needs --final, so it is not touched while you are still tuning.

Query modes (compared side by side by default):
  plain       subject + body
  body_first  the body alone when it has at least --min-body-words words, otherwise subject + body

Reports
  1. Recall@k for each method and query mode, for tickets that have a correct runbook (resolve, runbook escalations, injection,
     incidents). A ticket is a hit if its runbook is among the top-k runbooks (each runbook is ranked by its best chunk).
  2. For ambiguous ask_info tickets: a hit if ANY candidate runbook is in the top-k.
  3. Recall@3 of the best variant broken down by scenario and by how the ticket was written.
  4. The misses of the best variant, with what it returned instead.
  5. Whether the best raw score separates tickets that have a runbook from no-runbook tickets (AUC; 0.5 = no signal).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from chunking import runbook_hash
from retriever import Retriever, make_query

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def read(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def auc(pos, neg):
    """Chance that a random positive scores higher than a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def kind_and_truth(lab):
    if lab["expected_action"] == "ask_info":
        if lab.get("info_reason") == "ambiguous_between_runbooks":
            return "ambiguous", set(lab.get("candidate_runbooks") or [])
        return "skip", set()
    if lab.get("runbook_id"):
        return "runbook", {lab["runbook_id"]}
    if lab.get("escalation_reason") == "no_runbook":
        return "none", set()
    return "skip", set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ticket set, e.g. data/tickets/synthetic/final_v2")
    ap.add_argument("--index", default=f"{ROOT}/data/index")
    ap.add_argument("--runbooks", default=f"{ROOT}/data/runbooks")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--query-modes", nargs="*", default=["plain", "body_first"], choices=["plain", "body_first"])
    ap.add_argument("--min-body-words", type=int, default=4)
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5])
    ap.add_argument("--show-misses", type=int, default=8)
    ap.add_argument("--tag", default="", help="added to the saved result file name, e.g. aliases")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    if a.split != "dev" and not a.final:
        sys.exit("The test split is for final results only. If this is the final run, add --final.")

    r = Retriever(a.index)
    if os.path.isdir(a.runbooks) and r.meta.get("runbook_hash") != runbook_hash(a.runbooks):
        print("WARNING: the runbooks changed since this index was built. Rebuild it with build_index.py.\n")
    methods = a.methods or r.methods
    modes = a.query_modes
    inputs = {t["ticket_id"]: t for t in read(a.base + "_input.jsonl")}
    labels = {t["ticket_id"]: t for t in read(a.base + "_labels.jsonl")}
    ids = [t for t in inputs if t in labels and (a.split == "all" or labels[t].get("split") == a.split)]
    kmax = max(a.ks)

    rows = []
    for tid in ids:
        lab = labels[tid]
        kind, truth = kind_and_truth(lab)
        if kind == "skip":
            continue
        s, b = inputs[tid]["subject"], inputs[tid]["body"]
        rows.append(dict(id=tid, kind=kind, truth=truth, scenario=lab.get("scenario", ""),
                         subject_style=lab.get("subject_style", ""), body_style=lab.get("body_style", ""),
                         queries={m: make_query(s, b, m, a.min_body_words) for m in modes}))
    n_by = {k: sum(x["kind"] == k for x in rows) for k in ("runbook", "ambiguous", "none")}
    print(f"split={a.split}: {len(ids)} tickets, scored {len(rows)} "
          f"(with a runbook {n_by['runbook']}, ambiguous ask_info {n_by['ambiguous']}, no runbook {n_by['none']})")
    print(f"index: {r.meta['n_chunks']} chunks, embedder={r.meta.get('embedder') or 'none'}, methods={methods}, query modes={modes}\n")

    variants = [(m, q) for m in methods for q in modes]
    ranked = {v: {x["id"]: [rid for rid, _ in r.rank_runbooks(x["queries"][v[1]], k=kmax, method=v[0])] for x in rows}
              for v in variants}

    def recall(v, kind, k, subset=None):
        sel = [x for x in rows if x["kind"] == kind and (subset is None or subset(x))]
        if not sel:
            return None, 0
        hits = sum(bool(x["truth"] & set(ranked[v][x["id"]][:k])) for x in sel)
        return hits / len(sel), len(sel)

    name = lambda v: f"{v[0]}/{v[1]}"
    results = {}
    for kind, title in (("runbook", "Tickets with a correct runbook"), ("ambiguous", "Ambiguous ask_info (any candidate in top-k)")):
        if not n_by[kind]:
            continue
        print(title + f"  (n={n_by[kind]})")
        print("  variant            " + "  ".join(f"R@{k:<3}" for k in a.ks))
        for v in variants:
            vals = [recall(v, kind, k)[0] for k in a.ks]
            results[f"{kind}/{name(v)}"] = dict(zip(map(str, a.ks), vals))
            print(f"  {name(v):<18} " + "  ".join(f"{x:5.2f}" for x in vals))
        print()

    if not n_by["runbook"]:
        print("No tickets with a correct runbook in this split, so nothing more to report.")
        return
    best = max(variants, key=lambda v: (recall(v, "runbook", 3)[0] or 0, recall(v, "runbook", 1)[0] or 0))
    print(f"Best variant on Recall@3: {name(best)}\n")

    for field, label in (("scenario", "scenario"), ("body_style", "how the body was written"), ("subject_style", "how the subject was written")):
        print(f"Recall@3 by {label} ({name(best)}):")
        for v in sorted({x[field] for x in rows if x["kind"] == "runbook"}):
            rec, n = recall(best, "runbook", 3, lambda x, v=v, f=field: x[f] == v)
            print(f"  {v or '(none)':<20} {rec:5.2f}  n={n}")
        print()

    misses = [x for x in rows if x["kind"] == "runbook" and not (x["truth"] & set(ranked[best][x["id"]][:3]))]
    if misses and a.show_misses:
        print(f"Misses at top-3 for {name(best)} ({len(misses)} tickets, showing {min(len(misses), a.show_misses)}):")
        for x in misses[:a.show_misses]:
            print(f"  {x['id']} [{x['scenario']}] wanted {sorted(x['truth'])}  got {ranked[best][x['id']][:3]}")
            print(f"      {x['queries'][best[1]][:150]!r}")
        print()

    pos = [x for x in rows if x["kind"] == "runbook"]
    neg = [x for x in rows if x["kind"] == "none"]
    if neg:
        print(f"Does the best raw score separate runbook tickets (n={len(pos)}) from no-runbook tickets (n={len(neg)})? (query mode: {best[1]})")
        for m in [m for m in methods if m != "hybrid"]:
            sp = [r.top_raw_score(m, x["queries"][best[1]]) for x in pos]
            sn = [r.top_raw_score(m, x["queries"][best[1]]) for x in neg]
            print(f"  {m:<8} AUC={auc(sp, sn):.2f}   mean top score: runbook {np.mean(sp):.3f}  no-runbook {np.mean(sn):.3f}")
        print("  (0.5 means the score says nothing, 1.0 means it separates them perfectly)\n")

    suffix = ("_" + a.tag) if a.tag else ""
    out = a.save or f"{ROOT}/experiments/retrieval_{a.split}_{r.meta.get('embedder') or 'keyword'}{suffix}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(split=a.split, n_tickets=len(rows), index_meta=r.meta, recall=results, best=name(best)), f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
