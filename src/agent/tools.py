"""Simulated enterprise tools. Nothing here touches a real system: every call only writes to an in-memory log.

The agent may only call tools that the chosen runbook lists in `allowed_actions`. `escalate_to_human` is always available
and is not a tool call (escalation is a decision).
"""
import os

import yaml


class ToolEnv:
    def __init__(self, actions_path):
        with open(actions_path, encoding="utf-8") as f:
            self.tools = {t["name"]: t for t in yaml.safe_load(f)}
        self.log = []

    def check(self, runbook, calls):
        """Return (ok, bad tool names). A call is bad if the tool is unknown or not allowed for this runbook."""
        allowed = set(runbook.get("allowed_actions", [])) if runbook else set()
        bad = [c["tool"] for c in calls if c["tool"] not in self.tools or c["tool"] not in allowed]
        return not bad, bad

    def execute(self, ticket_id, calls):
        effects = []
        for c in calls:
            t = self.tools.get(c["tool"])
            if not t:
                continue
            status = "pending_approval" if t.get("requires_approval") else "done"
            e = dict(ticket_id=ticket_id, tool=c["tool"], args=c.get("args", {}), status=status)
            self.log.append(e)
            effects.append(e)
        return effects
