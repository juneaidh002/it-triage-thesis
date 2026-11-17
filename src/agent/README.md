# Triage agent v0

Per ticket: redact personal data -> screen for hostile text and security topics -> retrieve the top 3 runbooks (frozen hybrid setup)
-> call 1 (structured decision and draft) -> call 2 (evidence check). The raw outputs are stored once; the conditions are applied
afterwards, so tuning a threshold costs no API calls.

## Run order (from the project root)
1. Smoke test, about 10 calls:
       python src/agent/run_agent.py --base data/tickets/synthetic/final_v2 --split dev --limit 5 --tag smoke
   Read experiments/agent_runs/final_v2_dev_smoke.jsonl: is the decision sensible, is the JSON usable, are tokens recorded?
2. The dev set, about 200 calls (about 15 minutes at one call every 4.5 seconds):
       python src/agent/run_agent.py --base data/tickets/synthetic/final_v2 --split dev
3. Score it and tune the confidence thresholds on dev:
       python src/agent/evaluate_agent.py --base data/tickets/synthetic/final_v2 --tune
4. Change prompts or rules only on dev, and test changes on a slice (--limit 30 --tag p2). A changed prompt means new calls.
5. When dev is final, freeze prompts, rules and thresholds, then run each reserved set once:
       python src/agent/run_agent.py --base data/tickets/synthetic/final_v2 --split test --final
       python src/agent/evaluate_agent.py --base data/tickets/synthetic/final_v2 --split test --final
       python src/agent/run_agent.py --base data/tickets/gold/gold_v1 --split gold --final
       python src/agent/evaluate_agent.py --base data/tickets/gold/gold_v1 --split gold --final

If a run stops (daily budget, network, rate limit) run the same command again later: finished calls are cached.

## Conditions
c1 standard RAG (always acts) | c2 confidence escalation | c3 evidence verification plus policy checks | c4 c3 plus the ask outcome.
The headline comparison is c1, c2, c3. c4 is the extension with the third outcome.
`evaluate_agent.py --no-sensitive-screen` is the ablation that switches off the security-topic rules in c3 and c4.

## Metrics
coverage, error among handled (wrong or unsafe among resolved), unsafe rate, strict and lenient accuracy, escalation recall and
precision, ask rate, over-asking, injection and no-runbook tickets resolved (lower is better), tokens and seconds per ticket.

## What is stored
Run files hold the redacted query, counts of personal data found per type, guardrail flags, retrieved runbooks, the model's
structured outputs and token use. They never hold the original personal data.

## Not in v0
The second turn of the ask-then-resolve flow, redaction scoring on the gold personal-data tickets, a cost table in money,
and name detection beyond simple rules.
