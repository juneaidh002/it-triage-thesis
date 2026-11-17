"""Run the agent over a ticket set and store the raw model outputs (no condition is applied here).

  python src/agent/run_agent.py --base data/tickets/synthetic/final_v2 --split dev --limit 10    # smoke test
  python src/agent/run_agent.py --base data/tickets/synthetic/final_v2 --split dev
  python src/agent/run_agent.py --base data/tickets/gold/gold_v1 --split gold --final

Every model call is cached, so an interrupted or budget-stopped run continues where it stopped when you run it again.
The test and gold sets need --final: use them once, with thresholds already tuned on dev.
"""
import argparse
import json
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from agent import TriageAgent  # noqa: E402  (also puts retrieval and common on the path)
from chunking import load_runbooks  # noqa: E402
from llm import load_models  # noqa: E402
from retriever import Retriever  # noqa: E402


def read(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv=None, call_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "gold", "all"])
    ap.add_argument("--final", action="store_true", help="required for test, gold and all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-verify", action="store_true", help="skip call 2 (conditions c3 and c4 then cannot be evaluated)")
    ap.add_argument("--no-guardrails", action="store_true")
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--config", default=f"{ROOT}/configs/agent.yaml")
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.split != "dev" and not a.final:
        sys.exit("Only the dev split can be run freely. For test or gold, tune the thresholds on dev first, then add --final.")

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.no_guardrails:
        cfg["guardrails"] = {"pii_redaction": False, "injection_screen": False}
    runbooks = {r["id"]: r for r in load_runbooks(os.path.join(ROOT, "data", "runbooks"))}
    retriever = Retriever(os.path.join(ROOT, cfg["retrieval"].get("index_dir", "data/index")))
    models = load_models(a.models)
    agent = TriageAgent(models, retriever, runbooks, cfg, call_fn=call_fn)

    inputs = {t["ticket_id"]: t for t in read(a.base + "_input.jsonl")}
    labels = {t["ticket_id"]: t for t in read(a.base + "_labels.jsonl")}
    ids = [t for t in inputs if t in labels and (a.split == "all" or labels[t].get("split") == a.split)]
    if a.limit:
        ids = ids[:a.limit]
    out = a.out or os.path.join(ROOT, "experiments", "agent_runs", f"{os.path.basename(a.base)}_{a.split}_{a.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"{len(ids)} tickets from {a.base} ({a.split}); agent model: {models['agent']['model']}; verify={not a.no_verify}")

    n_done = n_bad = n_vbad = n_flag = tok = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for i, tid in enumerate(ids, 1):
            try:
                rec = agent.run_ticket(inputs[tid], verify=not a.no_verify)
            except RuntimeError as e:                       # daily budget guard from the model module
                print(f"\nSTOPPED: {e}")
                break
            except Exception as e:                          # network or provider error: keep what we have, resume later
                print(f"\nSTOPPED on {tid}: {type(e).__name__}: {str(e)[:200]}\nRe-run the same command to continue (finished calls are cached).")
                break
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            n_bad += not rec["agent_ok"]
            n_vbad += (rec["agent_ok"] and bool(rec["agent_out"]["runbook_id"]) and not rec["verify_ok"] and not a.no_verify)
            n_flag += bool(rec["injection_flags"])
            for u in rec["usage"].values():
                tok += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
            o = rec["agent_out"]
            print(f"[{i}/{len(ids)}] {tid} -> " + (f"{o['decision']} {o['runbook_id']} conf {o['resolution_confidence']}" if o else "no usable output")
                  + (f" flags {rec['injection_flags']}" if rec["injection_flags"] else ""))
    print(f"\ndone {n_done}/{len(ids)} in {time.time() - t0:.0f}s | unusable agent outputs {n_bad} | failed evidence checks {n_vbad} | "
          f"injection flagged {n_flag} | tokens {tok}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
