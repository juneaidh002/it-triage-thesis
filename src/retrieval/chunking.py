"""Turn runbook YAML files into retrieval chunks.

Each runbook becomes three chunks, so the agent (and the verifier) can retrieve "how to fix it" and
"when NOT to act" separately:

  overview    title, category, when it applies, what to check first
  steps       the numbered fix steps
  escalation  when to hand the ticket to a human (plus the escalation note)

Every chunk text starts with the runbook id, title and category, so a chunk still makes sense on its own.
If a runbook has an `aliases` list (how people tend to word the problem), it is added to the overview chunk.
Only what a service desk reader would see is embedded. Fields such as `allowed_actions`, `auto_resolvable`,
`must_mention` and `verification` stay in metadata or are left out: they are not searchable text.
"""
import glob
import hashlib
import os

import yaml

SECTIONS = ("overview", "steps", "escalation")
REQUIRED = ("id", "title", "category", "owner", "applies_when", "pre_checks", "steps", "escalate_if")


def load_runbooks(runbook_dir):
    runbooks = []
    for path in sorted(glob.glob(os.path.join(runbook_dir, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            rb = yaml.safe_load(f)
        missing = [k for k in REQUIRED if k not in rb]
        if missing:
            raise ValueError(f"{os.path.basename(path)} is missing required fields: {missing}")
        runbooks.append(rb)
    return runbooks


def runbook_hash(runbook_dir):
    """Fingerprint of the runbook files, stored with the index so a stale index is noticed."""
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(runbook_dir, "*.yaml"))):
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + f.read())
    return h.hexdigest()[:12]


def chunks_for(rb):
    rid = rb["id"]
    head = f"Runbook {rid}: {rb['title']} ({rb['category']})."
    overview = f"{head}\nApplies when: {rb['applies_when']}."
    if rb.get("aliases"):
        overview += f"\nPeople often describe this as: {'; '.join(rb['aliases'])}."
    overview += f"\nCheck first: {'; '.join(rb['pre_checks'])}."
    steps = head + "\nSteps:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(rb["steps"], 1))
    esc = f"{head}\nEscalate to a human if: {'; '.join(rb['escalate_if'])}."
    if rb.get("escalation_note"):
        esc += f"\nWhen escalating, {rb['escalation_note'][0].lower()}{rb['escalation_note'][1:]}."
    if not rb.get("auto_resolvable", True):
        esc += "\nThis kind of problem is never resolved automatically. A human always handles it."
    meta = dict(runbook_id=rid, title=rb["title"], category=rb["category"], owner=rb["owner"],
                allowed_actions=rb.get("allowed_actions", []), auto_resolvable=bool(rb.get("auto_resolvable", True)))
    return [dict(chunk_id=f"{rid}#{sec}", section=sec, text=text, **meta)
            for sec, text in zip(SECTIONS, (overview, steps, esc))]


def build_chunks(runbook_dir):
    chunks = []
    for rb in load_runbooks(runbook_dir):
        chunks.extend(chunks_for(rb))
    return chunks


if __name__ == "__main__":
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cs = build_chunks(sys.argv[1] if len(sys.argv) > 1 else f"{root}/data/runbooks")
    print(f"{len(cs)} chunks from {len({c['runbook_id'] for c in cs})} runbooks")
    for c in cs[:3]:
        print("\n---", c["chunk_id"], "\n" + c["text"])
