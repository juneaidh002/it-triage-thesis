"""Role-based LLM access with caching.

  models = load_models()
  text = cached_call(models, "generation", system, user)

Every non-cached call stores text, token usage and latency in experiments/llm_cache/,
so cost and latency can be reported later without extra logging.

Check your setup (no API calls):  python src/common/llm.py --check
"""
import argparse, hashlib, json, os, re, time
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = f"{ROOT}/experiments/llm_cache"


def _load_dotenv(path=f"{ROOT}/.env"):
    """Read KEY=VALUE lines from .env (git-ignored) without overriding variables already set."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def load_models(path=None):
    return yaml.safe_load(open(path or f"{ROOT}/configs/models.yaml", encoding="utf-8"))


def _call_anthropic(r, system, user):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get(r.get("api_key_env", "ANTHROPIC_API_KEY")))
    m = client.messages.create(model=r["model"], max_tokens=r["max_tokens"], temperature=r["temperature"],
                               system=system, messages=[{"role": "user", "content": user}])
    return m.content[0].text, {"input_tokens": m.usage.input_tokens, "output_tokens": m.usage.output_tokens}


def _call_openai(r, system, user):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get(r.get("api_key_env", "OPENAI_API_KEY"), "none"), base_url=r.get("base_url"))
    kwargs = dict(model=r["model"], max_tokens=r["max_tokens"], temperature=r["temperature"],
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if r.get("extra_body"):
        kwargs["extra_body"] = r["extra_body"]
    m = client.chat.completions.create(**kwargs)
    u = m.usage
    return m.choices[0].message.content or "", {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}


def _call_mock(r, system, user):
    return json.dumps({"subject": "mock subject", "body": "mock body", "reply": "mock reply", "best_runbook": "unclear", "multi_user": False, "confidence": "low"}), {"input_tokens": 0, "output_tokens": 0}


PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai, "mock": _call_mock}


def clean_text(text):
    """Remove reasoning blocks that some models put before the answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _with_retry(fn, r, system, user, tries=5):
    for n in range(tries):
        try:
            return fn(r, system, user)
        except Exception as e:  # retry only on rate limits and temporary errors
            msg = str(e).lower()
            if n == tries - 1 or not any(k in msg for k in ("429", "rate limit", "timeout", "temporarily", "503", "502")):
                raise
            time.sleep(2 ** (n + 1))


def cached_call(models, role, system, user, refresh=False):
    r = models[role]
    if r["model"] == "CHANGE_ME" and r["provider"] != "mock":
        raise RuntimeError(f"Set a model for role '{role}' in configs/models.yaml")
    key = hashlib.sha256(json.dumps([role, r["provider"], r["model"], r["temperature"], r["max_tokens"], system, user]).encode()).hexdigest()
    path = f"{CACHE}/{key}.json"
    if refresh and os.path.exists(path):
        os.remove(path)          # drop a cached answer that turned out to be unusable, then ask again
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["text"]
    t0 = time.time()
    raw, usage = _with_retry(PROVIDERS[r["provider"]], r, system, user)
    text = clean_text(raw)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"role": role, "provider": r["provider"], "model": r["model"], "text": text, "raw_text": raw, "usage": usage,
               "latency_s": round(time.time() - t0, 3), "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False)
    return text


def check(models):
    print(f"{'role':<12}{'provider':<11}{'model':<22}{'key env':<20}key set?")
    for name, r in models.items():
        env = r.get("api_key_env", "-")
        ok = "n/a" if r["provider"] == "mock" or (r.get("base_url") and "localhost" in str(r["base_url"])) else ("yes" if os.environ.get(env) else "NO")
        print(f"{name:<12}{r['provider']:<11}{r['model']:<22}{env:<20}{ok}")
        if r["model"] == "CHANGE_ME":
            print(f"  -> set a model for '{name}'")
    g, a, j = models.get("generation"), models.get("agent"), models.get("judge")
    if g and a and g["model"] != "CHANGE_ME" and (g["model"], g["provider"]) == (a["model"], a["provider"]):
        print("WARNING: generation and agent use the same model; results may look better than they are.")
    if g and a and g["model"] != "CHANGE_ME" and g["provider"] == a["provider"] and g["model"] != a["model"]:
        print("NOTE: generation and agent share a provider; different model families are better if you can afford it.")
    if a and j and a["model"] != "CHANGE_ME" and (a["model"], a["provider"]) == (j["model"], j["provider"]):
        print("WARNING: judge is the same model as the agent; validate it against human labels.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    check(load_models(a.models))
