# IT ticket triage thesis

Evidence-grounded IT support ticket resolution with selective escalation.

## Layout
- data/runbooks/      source of truth, one YAML per runbook (40)
- data/taxonomy/      categories.yaml, actions.yaml (simulated tool names)
- data/tickets/       synthetic/ generated, gold/ hand-written, external/ public sets
- data/processed/     chunks.jsonl, splits, dedupe outputs (rebuildable)
- data/audits/        manual review sheets
- src/                code by stage (generation, guardrails, retrieval, agent, tools, evaluation, app)
- configs/            model, retriever, threshold settings
- experiments/        run outputs and cached LLM responses
- reports/            thesis text and figures

## Rules
- Test and gold tickets are never indexed.
- Priority is assigned by a table, never from words in the ticket.
- Every runbook is hand-verified before use.

## Runbook verification
Each runbook has a `verification` block (status, sources checked, notes). See runbooks_for_review.md for the summary.
Statuses: checked_official_docs, checked_vendor_docs, checked_community_answers, checked_generic_guides_only, not_verifiable_company_specific, unverified.

## Ticket generation
1. Set a model per role in configs/models.yaml (generation, agent, judge, agent_local) and export the API keys it names. Check with: `python src/common/llm.py --check`
2. Dry run (no cost): `python src/generation/generate_tickets.py --n 40 --seed 1 --dry-run`
3. Pilot: `python src/generation/generate_tickets.py --n 20 --seed 1`
4. Check: `python src/generation/validate_tickets.py data/tickets/synthetic/tickets_seed1_n20   # checks the _input and _labels pair`
5. Read every pilot ticket by hand before scaling. data/tickets/synthetic/pilot_22_input.jsonl and pilot_22_labels.jsonl are a hand-drafted reference set.
6. The agent loads only *_input.jsonl (ticket_id, subject, body). Answer keys live in *_labels.jsonl and are used by the evaluator only.
Labels come from the plan, never from the LLM. Priority comes from the rule table in src/generation/ticket_rules.py.

## Actions and languages
- The agent chooses one of three actions: resolve, escalate, or ask_info (ask the user one clarifying question).
- ask_info tickets carry info_reason (too_vague or ambiguous_between_runbooks), candidate_runbooks, target_runbook_id and a simulated followup_reply in the labels file. These are for the evaluator only.
- All tickets are written in English only.

## Models
- Use a different model for ticket generation than for the agent. Keep the agent model identical across all experiment conditions.
- Providers: anthropic, openai (any OpenAI-compatible endpoint, including local servers), mock (testing only).
- Every API call is cached in experiments/llm_cache/ with token usage and latency, for the cost analysis.

## Fit audit
After generating tickets, check that each one really fits its label:
`python src/generation/audit_fit.py data/tickets/synthetic/tickets_seed1_n20 --role generation`
An LLM sees only the ticket text and picks the best runbook. Mismatches are listed for manual review. Use a model different from the generator when possible (--role agent or judge).

## Scaling up (synthetic benchmark)
1. Generate batches with different seeds. Each seed is one batch; re-running the same command resumes from the cache.
   PowerShell: `foreach ($s in 11..15) { python src/generation/generate_tickets.py --n 200 --seed $s }`
2. Validate each batch: `python src/generation/validate_tickets.py data/tickets/synthetic/tickets_seed11_n200`
3. Audit each batch (optional `--sample 0.5` to save tokens): `python src/generation/audit_fit.py data/tickets/synthetic/tickets_seed11_n200 --role generation`
4. Merge, de-duplicate, drop audit failures and split: `python src/generation/build_dataset.py <batch bases...> --out data/tickets/synthetic/final_v1 --resplit`
5. Read the *_rejected.jsonl reasons, then manually review 10-15% of the kept tickets (record the error rate in data/audits/).
