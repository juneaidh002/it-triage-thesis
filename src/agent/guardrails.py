"""Guardrails that run BEFORE any text reaches the model.

  redact(subject, body)  -> replaces personal data with placeholders such as [EMAIL_1]
  screen_injection(text) -> names of the rules that matched text aimed at the assistant

Redaction is rule based. It is good at emails, phone numbers and employee ids and only partial for names (names are
caught after phrases such as 'this is', 'my name is', 'regards' and in front of an id or email). The original values are
kept in memory only (Redactor.map) and are never written to the run files, which store counts per type.
"""
import re

NAME_STOP = {"team", "support", "it", "help", "desk", "please", "hi", "hello", "sir", "madam", "all", "locked",
             "unable", "not", "stuck", "having", "facing", "new", "back", "here", "writing", "trying", "getting"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
EMPID_A = re.compile(r"(?i)\b((?:emp(?:loyee)?|staff)\s*(?:id|no\.?|number|#)?\s*[:#-]?\s*)([A-Z]{0,4}-?\d{3,8})\b")
EMPID_B = re.compile(r"(?i)\b(EMP-?\d{3,8})\b")
EMPID_C = re.compile(r"(?i)\b(id\s*[:#-]?\s*)(\d{4,8})\b")
PHONE_MARK = re.compile(r"(?i)\b((?:phone|mobile|cell|tel|ph|ext(?:ension)?)\.?\s*(?:no\.?|number|#|:)?\s*)(\+?\d[\d\s-]{2,13}\d)")
PHONES = [re.compile(p) for p in (
    r"\+\d{1,3}[\s-]?\d[\d\s-]{7,13}\d",
    r"(?<![\w.])[6-9]\d{9}(?![\w.])",
    r"(?<![\w.])\d{3}[\s-]\d{3}[\s-]\d{4}(?![\w.])",
    r"(?<![\w.])\d{5}[\s-]\d{5}(?![\w.])",
    r"(?<![\w.])\d{3}-\d{4}(?![\w.])",
    r"(?<![\w.])\d{2,5}-\d{6,8}(?![\w.])",
    r"(?<![\w.])\d{7,12}(?![\w.])")]
NAME_AFTER = re.compile(r"(?i:this is|my name is|i am|i'm|regards,?|thanks,?|thank you,?|cheers,?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
NAME_BEFORE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s*[\(,]\s*(?i:emp|employee|id\b|[\w.+-]+@))")


class Redactor:
    def __init__(self):
        self.map = {}          # (kind, lowercase value) -> placeholder; stays in memory
        self.counts = {}

    def _tag(self, kind, value):
        key = (kind, value.lower())
        if key not in self.map:
            n = sum(1 for k in self.map if k[0] == kind) + 1
            self.map[key] = f"[{kind}_{n}]"
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return self.map[key]

    def text(self, s):
        if not s:
            return s
        s = EMAIL.sub(lambda m: self._tag("EMAIL", m.group(0)), s)
        s = EMPID_A.sub(lambda m: m.group(1) + self._tag("EMP_ID", m.group(2)), s)
        s = EMPID_B.sub(lambda m: self._tag("EMP_ID", m.group(1)), s)
        s = EMPID_C.sub(lambda m: m.group(1) + self._tag("EMP_ID", m.group(2)), s)
        s = PHONE_MARK.sub(lambda m: m.group(1) + self._tag("PHONE", m.group(2)), s)
        for p in PHONES:
            s = p.sub(lambda m: self._tag("PHONE", m.group(0)), s)

        def name_after(m):
            first = m.group(1).split()[0].lower()
            return m.group(0) if first in NAME_STOP else m.group(0).replace(m.group(1), self._tag("NAME", m.group(1)), 1)
        s = NAME_AFTER.sub(name_after, s)
        s = NAME_BEFORE.sub(lambda m: self._tag("NAME", m.group(1)), s)
        return s


def redact(subject, body):
    r = Redactor()
    return r.text(subject or ""), r.text(body or ""), r


INJECTION_RULES = [
    ("ignore_instructions", r"(?i)\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|your|the)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|polic(?:y|ies))\b"),
    ("note_to_assistant", r"(?i)\b(?:note|message|instruction)s?\s+(?:to|for)\s+(?:the\s+)?(?:support\s+|ai\s+|it\s+)?(?:assistant|bot|agent|ai|system|model)\b"),
    ("system_note", r"(?i)\bsystem\s*(?:note|message|prompt|override|instruction)"),
    ("role_play", r"(?i)\byou are (?:now|no longer)\b|\bpretend (?:to be|you)\b|\bact as (?:an?|the)\b"),
    ("reveal_prompt", r"(?i)\b(?:reveal|print|show|display|repeat|output|list)\b.{0,40}\b(?:system prompt|internal instructions|your instructions|your rules|runbooks?)\b"),
    ("redirect_credentials", r"(?i)\b(?:send|email|forward|mail)\b.{0,50}\b(?:reset link|password|credentials|codes?)\b.{0,40}\b(?:personal|my own|gmail|yahoo|hotmail|outlook\.com)"),
    ("skip_verification", r"(?i)\b(?:skip|bypass)\b.{0,25}\b(?:verification|verify|mfa|approval|authentication)\b"),
    ("authority_claim", r"(?i)\bi am the (?:it )?(?:director|cio|ciso|admin(?:istrator)?|manager)\b.{0,80}\b(?:order|authori[sz]e|instruct|command|require)"),
    ("bulk_approve", r"(?i)\bapprove (?:all|every)\b"),
    ("grant_privilege", r"(?i)\b(?:grant|give)\b.{0,30}\b(?:admin|administrator|root|privileged)\b.{0,30}\b(?:role|access|rights)\b"),
    ("erase_logs", r"(?i)\b(?:clear|delete|wipe|erase)\b.{0,30}\b(?:audit|logs?)\b"),
    ("unlock_other", r"(?i)\bunlock\b.{0,40}\b(?:svc_?\w*|service account|admin\w*|root)\b"),
]
_COMPILED = [(n, re.compile(p)) for n, p in INJECTION_RULES]

# Topics a human must always handle (security incidents). Kept separate from injection so the effect can be measured on its own.
SENSITIVE_RULES = [
    ("phishing", r"(?i)\b(?:phishing|phished|scam|suspicious (?:e-?mail|message|link)|fake (?:e-?mail|message))\b|\bclicked (?:on )?(?:a|the|that|this)\b.{0,20}\blink\b|\b(?:typed|entered|gave) my (?:work )?password\b"),
    ("lost_or_stolen_device", r"(?i)\b(?:stolen|theft)\b|\b(?:lost|misplaced|missing)\b.{0,15}\b(?:laptop|phone|device|mobile)\b|\bleft my (?:work )?(?:laptop|phone)\b|\bcan(?:no|')?t find my (?:company |work )?(?:laptop|phone)\b"),
    ("suspicious_account_activity", r"(?i)\bunexpected\b.{0,20}\b(?:sign-?in|log-?in|mfa|approval)\b|\b(?:prompts?|requests?|alerts?)\b.{0,25}\b(?:i did not|i didn'?t|never)\b.{0,15}\b(?:start|request|make|initiate)|\bsomeone (?:else )?(?:is|may be|might be) using my account"),
]
_SENS = [(n, re.compile(p)) for n, p in SENSITIVE_RULES]


def screen_sensitive(text):
    return [n for n, p in _SENS if p.search(text or "")]


def screen_injection(text):
    return [n for n, p in _COMPILED if p.search(text or "")]
