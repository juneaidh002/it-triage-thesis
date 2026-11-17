# Retrieval module

Chunks the 40 runbooks, builds a searchable index, and measures whether the right runbook is found.

## Install (small)
    pip install rank_bm25 scikit-learn numpy pyyaml
    pip install fastembed          # dense vectors; small ONNX model (~67 MB download), no torch
    pip install chromadb           # optional, only if you want a Chroma vector database

## Basic steps (run from the project root)
    python src/retrieval/build_index.py --embedder fastembed
    python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2
    python src/retrieval/retriever.py "printer queue stuck" --k 3
    python src/retrieval/retriever.py "printer queue stuck" --k 3 --exclude RB-PRT-01     # evidence-removal test

## Experiments, in this order (change one thing at a time)
A. Query mode. Needs no rebuild; the evaluation compares `plain` and `body_first` side by side:
       python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2 --tag base
B. Aliases (how employees word each problem), then rebuild and compare:
       python src/retrieval/add_aliases.py
       python src/retrieval/build_index.py --embedder fastembed
       python src/retrieval/evaluate_retrieval.py --base data/tickets/synthetic/final_v2 --tag aliases
   Undo with `python src/retrieval/add_aliases.py --remove` (then rebuild). A backup of the runbooks is kept in
   data/runbooks_backup_before_aliases.
Results are saved in experiments/ with the tag in the file name.

## Rules
- Evaluate on the dev split only. The test split needs --final, which you use once, at the end.
- Rebuild the index whenever a runbook changes (the evaluation warns you if you forget).
- Aliases were written after the dev misses were seen, so dev recall after step B is partly tuned. Say so in the thesis
  and keep the test split untouched until the final run.
- In code: `Retriever("data/index").search(make_query(subject, body, "body_first"), k=5, method="hybrid", exclude_runbooks=[...])`.

## What goes into the index
Three chunks per runbook (overview, steps, escalation). The overview also carries the aliases, if any.
`allowed_actions`, `auto_resolvable`, category and owner are stored as metadata with each chunk.
Test, dev and gold tickets are never indexed.
