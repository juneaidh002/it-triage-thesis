"""Adds two guards to src/common/llm.py for tight free-tier limits. Safe to run twice; keeps a backup.

  python patch_llm_limits.py            (run from the project root)

1. min_interval_s (per role in configs/models.yaml): wait at least this many seconds between real API calls.
   Cached calls are not delayed. For 15 requests per minute use 4.5.
2. max_requests_per_day (per role): count real API calls per calendar day in experiments/usage_<date>.json and stop with a
   clear message before the limit is hit. Cached results are kept, so tomorrow's run continues where today's stopped.
   The provider's day may not end at your local midnight, so treat the counter as a safety margin, not an exact meter.
"""
import os
import shutil
import sys

PATH = os.path.join("src", "common", "llm.py")
if not os.path.exists(PATH):
    sys.exit(f"{PATH} not found. Run this from the project root.")
s = open(PATH, encoding="utf-8").read()
if "_last_call" in s:
    sys.exit("llm.py already has the limit guards. Nothing to do.")

A1 = 'CACHE = f"{ROOT}/experiments/llm_cache"\n'
A2 = '    raw, usage = _with_retry(PROVIDERS[r["provider"]], r, system, user)\n'
for a in (A1, A2):
    if a not in s:
        sys.exit("llm.py does not look like the expected version, so it was not changed. Send me the file and I will adapt the patch.")

helpers = A1 + '''
_last_call = {}


def _count_call(role, limit):
    """Count one real API call for today; stop before the daily limit is exceeded."""
    path = f"{ROOT}/experiments/usage_{time.strftime('%Y-%m-%d')}.json"
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    n = data.get(role, 0)
    if limit and n >= limit:
        raise RuntimeError(f"Daily budget reached for role '{role}': {n} real API calls today (limit {limit}). "
                           "Cached results are kept. Run the same command again tomorrow.")
    data[role] = n + 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
'''
guard = '''    _count_call(role, r.get("max_requests_per_day"))
    gap = r.get("min_interval_s", 0)
    if gap:
        wait = _last_call.get(role, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
    raw, usage = _with_retry(PROVIDERS[r["provider"]], r, system, user)
    _last_call[role] = time.time()
'''
shutil.copy(PATH, PATH + ".bak")
s = s.replace(A1, helpers, 1).replace(A2, guard, 1)
open(PATH, "w", encoding="utf-8").write(s)
print("patched src/common/llm.py (backup: llm.py.bak). Add min_interval_s and max_requests_per_day to the roles in configs/models.yaml.")
