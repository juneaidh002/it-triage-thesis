"""Shared rules for ticket generation and validation.

Priority is computed from facts about the incident (category, whether it blocks work,
how many people are affected). It is never read from the ticket text, so the text
cannot leak the label.
"""
import re

BANNED_WORDS = [
    "urgent", "asap", "critical", "emergency",
    "high priority", "medium priority", "low priority", "p1", "p2", "p3",
]

ESCALATION_REASONS = {
    "runbook_always_escalate",   # runbook is auto_resolvable: false
    "runbook_escalation_rule",   # one of the runbook's escalate_if conditions is met
    "no_runbook",                # no runbook covers the request
    "injection",                 # ticket text tries to instruct the system
    "out_of_policy",             # request the agent must never grant (v1 name)
    "needs_approval",            # privileged or restricted request that needs human approval (v2 name for out_of_policy)
    "multi_user_incident",       # several people affected, so it is an incident for a human (v2)
}

ACTIONS = {"resolve", "escalate", "ask_info"}

INFO_REASONS = {
    "too_vague",                    # nothing identifiable, ask what is wrong
    "ambiguous_between_runbooks",   # fits several runbooks, ask a question that tells them apart
}

NO_RUNBOOK_CATEGORY = "No matching runbook"
AMBIGUOUS_CATEGORY = "Ambiguous request"
OUT_OF_POLICY_CATEGORY = "Out of policy request"
NEEDS_APPROVAL_CATEGORY = "Approval-required request"

USERS_AFFECTED = ["one", "few", "many"]


def priority_rule(category: str, blocking: bool, users_affected: str) -> str:
    if category == "Security reports":
        return "high"
    if users_affected == "many":
        return "high"
    if blocking:
        return "high" if users_affected == "few" else "medium"
    return "medium" if users_affected == "few" else "low"


def banned_hits(text: str):
    low = text.lower()
    return [w for w in BANNED_WORDS if w in low]


# A ticket that says several people are affected can turn a "resolve" into an "escalate" for many runbooks
# (their escalate_if mentions several users, an outage, the same floor...). So tickets for those categories
# must stay single-user unless the runbook is one where other people being involved is normal.
MULTI_USER_RE = re.compile(r"(several|many|multiple)\s+(users|desks|people)|all users|everyone|outage|same floor|same area|same time", re.I)
MULTI_USER_OK = {"RB-DRV-02", "RB-DRV-03"}   # restoring a file or getting folder access naturally involves a team


def multi_user_categories(runbooks):
    return {r["category"] for r in runbooks.values()
            if r["id"] not in MULTI_USER_OK and any(MULTI_USER_RE.search(c) for c in r["escalate_if"])}
