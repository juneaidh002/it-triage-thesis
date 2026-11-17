"""Triage agent v0: collects model outputs once per ticket. Conditions and thresholds are applied afterwards (conditions.py),
so tuning a threshold costs no extra API calls.

Per ticket:  redact -> screen -> retrieve top runbooks -> call 1 (structured decision and draft) -> call 2 (evidence check).
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "retrieval"))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.dirname(__file__))

import guardrails
import prompts
from retriever import make_query

DECISIONS = {"resolve", "ask_info", "escalate"}
SECTION_ORDER = {"overview": 0, "steps": 1, "escalation": 2}


class ParseFailure(Exception):
    pass


def _json_obj(text):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ParseFailure("no JSON object in the reply")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ParseFailure(f"invalid JSON: {e}")


def normalize_agent(d, runbooks):
    dec = str(d.get("decision", "")).strip().lower()
    if dec not in DECISIONS:
        raise ParseFailure(f"bad decision {dec!r}")
    rid = d.get("runbook_id")
    rid = rid if isinstance(rid, str) and rid in runbooks else None
    steps = [str(s) for s in (d.get("draft_steps") or []) if str(s).strip()]
    calls = []
    for c in d.get("tool_calls") or []:
        if isinstance(c, str):
            calls.append({"tool": c, "args": {}})
        elif isinstance(c, dict) and c.get("tool"):
            calls.append({"tool": str(c["tool"]), "args": c.get("args") if isinstance(c.get("args"), dict) else {}})
    try:
        conf = max(0, min(100, int(float(d.get("resolution_confidence", d.get("confidence", 50))))))
    except (TypeError, ValueError):
        conf = 50
    q = d.get("question")
    return dict(decision=dec, reason_code=str(d.get("reason_code", "")), runbook_id=rid, draft_steps=steps, tool_calls=calls,
                citations=[str(c) for c in (d.get("citations") or [])], question=q if isinstance(q, str) and q.strip() else None,
                resolution_confidence=conf)


def normalize_verify(d):
    steps = []
    for s in d.get("steps") or []:
        if isinstance(s, dict):
            steps.append(dict(step=str(s.get("step", "")), supported=bool(s.get("supported")), evidence=s.get("evidence")))
    all_supported = bool(steps) and all(s["supported"] for s in steps)
    match, cond = bool(d.get("ticket_matches_runbook")), bool(d.get("escalation_condition_met"))
    return dict(steps=steps, ticket_matches_runbook=match, escalation_condition_met=cond,
                escalation_condition=d.get("escalation_condition"),
                overall_supported=bool(d.get("overall_supported")) and all_supported and match and not cond)


def _cache_usage(models, role, system, user):
    """Read token use and latency of a call from the cache file (same key as common/llm.py)."""
    try:
        r = models[role]
        key = hashlib.sha256(json.dumps([role, r["provider"], r["model"], r["temperature"], r["max_tokens"], system, user]).encode()).hexdigest()
        with open(os.path.join(ROOT, "experiments", "llm_cache", key + ".json"), encoding="utf-8") as f:
            c = json.load(f)
        u = c.get("usage") or {}
        return dict(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"), latency_s=c.get("latency_s"))
    except Exception:
        return dict(input_tokens=None, output_tokens=None, latency_s=None)


class TriageAgent:
    def __init__(self, models, retriever, runbooks, cfg, call_fn=None):
        self.models, self.retriever, self.runbooks, self.cfg = models, retriever, runbooks, cfg
        self.call_fn = call_fn          # for tests: call_fn(role, system, user) -> (text, usage)
        self.by_rb = {}
        for c in retriever.chunks:
            self.by_rb.setdefault(c["runbook_id"], []).append(c)
        for v in self.by_rb.values():
            v.sort(key=lambda c: SECTION_ORDER.get(c["section"], 9))

    # ---- model calls -------------------------------------------------------------------------
    def _ask(self, role, system, user, normalize, tries=3):
        last = None
        for attempt in range(tries):
            if self.call_fn:
                text, usage = self.call_fn(role, system, user)
            else:
                from llm import cached_call
                text = cached_call(self.models, role, system, user, refresh=attempt > 0)
                usage = _cache_usage(self.models, role, system, user)
            try:
                return normalize(_json_obj(text)), usage
            except ParseFailure as e:
                last = e
        raise last

    # ---- steps ---------------------------------------------------------------------------------
    def prepare(self, ticket):
        g = self.cfg.get("guardrails", {})
        subj, body = ticket["subject"], ticket["body"]
        counts = {}
        if g.get("pii_redaction", True):
            subj, body, red = guardrails.redact(subj, body)
            counts = red.counts
        flags = guardrails.screen_injection(f"{subj}\n{body}") if g.get("injection_screen", True) else []
        sens = guardrails.screen_sensitive(f"{subj}\n{body}") if g.get("sensitive_screen", True) else []
        rc = self.cfg.get("retrieval", {})
        query = make_query(subj, body, rc.get("query_mode", "plain"))
        method = rc.get("method", "hybrid")
        ranked = self.retriever.rank_runbooks(query, k=rc.get("k_runbooks", 3), method=method)
        top_method = "dense" if self.retriever.has_dense else "bm25"
        blocks = []
        for rid, _ in ranked:
            rb = self.runbooks[rid]
            blocks.append(prompts.runbook_block(rid, rb["title"], rb["category"], rb.get("allowed_actions", []), self.by_rb[rid]))
        return dict(subject=subj, body=body, pii=counts, flags=flags, sens=sens, query=query, ranked=ranked, blocks=blocks,
                    top_score=self.retriever.top_raw_score(top_method, query) if query else 0.0)

    def run_ticket(self, ticket, verify=True):
        p = self.prepare(ticket)
        rec = dict(ticket_id=ticket["ticket_id"], pii_found=p["pii"], injection_flags=p["flags"], sensitive_flags=p["sens"], query=p["query"],
                   retrieved=[dict(runbook_id=r, score=round(s, 5)) for r, s in p["ranked"]], top_score=round(p["top_score"], 5),
                   agent_ok=False, agent_out=None, verify_ok=False, verify_out=None, usage={}, errors=[], prompt_version=prompts.PROMPT_VERSION)
        if not p["ranked"]:
            rec["errors"].append("retrieval returned nothing")
            return rec
        try:
            out, usage = self._ask("agent", prompts.SYSTEM_AGENT, prompts.agent_user_prompt(p["subject"], p["body"], p["blocks"]),
                                   lambda d: normalize_agent(d, self.runbooks))
            rec.update(agent_ok=True, agent_out=out)
            rec["usage"]["agent"] = usage
        except ParseFailure as e:
            rec["errors"].append(f"agent: {e}")
        if verify and rec["agent_ok"] and rec["agent_out"]["runbook_id"]:
            rid = rec["agent_out"]["runbook_id"]
            rb = self.runbooks[rid]
            block = prompts.runbook_block(rid, rb["title"], rb["category"], rb.get("allowed_actions", []), self.by_rb[rid])
            try:
                v, usage = self._ask("agent", prompts.SYSTEM_VERIFY,
                                     prompts.verify_user_prompt(p["subject"], p["body"], block, rec["agent_out"]["draft_steps"], rec["agent_out"]["tool_calls"]),
                                     normalize_verify)
                rec.update(verify_ok=True, verify_out=v)
                rec["usage"]["verify"] = usage
            except ParseFailure as e:
                rec["errors"].append(f"verify: {e}")
        return rec
