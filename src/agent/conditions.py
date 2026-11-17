"""The experimental conditions, applied AFTER the model calls so thresholds can be tuned without new API calls.

  c1  standard RAG             always acts on the draft (never escalates)
  c2  confidence escalation    acts if resolution_confidence >= c2_confidence, otherwise escalates
  c3  evidence verification    acts only if the evidence check passes AND the policy checks pass (a runbook was chosen, it is
                               not an always-human runbook, no injection rule and no sensitive-topic rule matched, every tool is
                               on its allowed list; the sensitive-topic rule can be switched off to measure its effect)
                               AND resolution_confidence >= c3_confidence (default 0 = verification only); otherwise escalates
  c4  c3 plus ask              like c3, but if the model decided ask_info and wrote a question, the ticket asks instead of escalating

Headline comparison: c1, c2, c3. c4 is the extension with the third outcome (ask).
"""

CONDITIONS = ("c1", "c2", "c3", "c4")
DEFAULT_THRESHOLDS = {"c2_confidence": 70, "c3_confidence": 0}


def policy_ok(rec, runbooks, env, opts=None):
    out = rec["agent_out"]
    rid = out["runbook_id"]
    if rid is None:
        return False, "no_runbook"
    rb = runbooks[rid]
    if not rb.get("auto_resolvable", True):
        return False, "always_human_runbook"
    if rec["injection_flags"]:
        return False, "injection_flag"
    if rec.get("sensitive_flags") and (opts or {}).get("use_sensitive_screen", True):
        return False, "sensitive_topic"
    ok, bad = env.check(rb, out["tool_calls"])
    if not ok:
        return False, "tool_not_allowed"
    return True, ""


def decide(rec, cond, thr, runbooks, env):
    """Return dict(action, why, runbook_id, tool_calls)."""
    if not rec["agent_ok"]:
        # a failed model call cannot be acted on; the baseline would have produced nothing either
        return dict(action="escalate", why="agent_output_unusable", runbook_id=None, tool_calls=[])
    out = rec["agent_out"]
    base = dict(runbook_id=out["runbook_id"], tool_calls=out["tool_calls"])
    if cond == "c1":
        return dict(action="resolve", why="always_act", **base)
    if cond == "c2":
        if out["resolution_confidence"] >= thr["c2_confidence"]:
            return dict(action="resolve", why="confidence_ok", **base)
        return dict(action="escalate", why="low_confidence", **base)
    if cond in ("c3", "c4"):
        ok, why = policy_ok(rec, runbooks, env, thr)
        v = rec["verify_out"] if rec["verify_ok"] else None
        supported = bool(v and v["overall_supported"])
        if ok and supported and out["resolution_confidence"] >= thr["c3_confidence"]:
            return dict(action="resolve", why="verified", **base)
        if cond == "c4" and out["decision"] == "ask_info" and out["question"] and not rec["injection_flags"]:
            return dict(action="ask_info", why="model_asked", **base)
        return dict(action="escalate", why=why or ("not_verified" if not supported else "low_confidence"), **base)
    raise ValueError(cond)
