"""Bounded no-reasoning compatibility checks for the requested bulk run."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from augment_litellm import credentials, request_json, ENV, PROMPTS, reasoning_status, accepted_text

OUT = Path("artifacts/augmentation200")
OUT.mkdir(parents=True, exist_ok=True)
records = [json.loads(s) for s in Path(
    "artifacts/augmentation_no_reasoning_preflight/responses.jsonl").read_text().splitlines()]
profiles = {}
pending = []
for r in records:
    if r["accepted"]:
        profiles[r["target_label"]] = {
            "api_model": r["requested_model"], "options": {"reasoning_effort": "none"},
            "reported_model": r["reported_model"], "reasoning_check": r["reasoning_check"]}
    else:
        pending.append(r)
secrets, endpoint = credentials(ENV)


def check(r):
    model = r["requested_model"]
    if model.startswith("claude"):
        variants = [{"thinking": {"type": "disabled"}},
                    {"reasoning_effort": "disable", "thinking": {"type": "disabled"}}]
    elif model.startswith("gemini"):
        variants = [{"reasoning_effort": "disable"},
                    {"thinking": {"type": "disabled"}}]
    elif model.startswith("glm"):
        variants = [{"thinking": {"type": "disabled"}},
                    {"enable_thinking": False}]
    else:
        variants = [{"reasoning_effort": "disable"},
                    {"reasoning_effort": "minimal"}]
    attempts = []
    for options in variants:
        payload = {"model": model, "messages": [{"role": "user", "content": PROMPTS[0]}],
                   "temperature": .7, "max_tokens": 1024, **options}
        response, error = request_json(endpoint, secrets, payload, 90)
        response = response if isinstance(response, dict) else {}
        count, status = reasoning_status(response)
        reported = response.get("model")
        # Explicit observed gateway namespace; never strip version numbers.
        allowed = {model, "openai_api_plus/" + model}
        text, reason = accepted_text(response, reported) if reported in allowed else (None, "model_mismatch")
        if error:
            text, reason = None, error
        if status == "nonzero":
            text, reason = None, "reasoning_not_disabled"
        if options.get("reasoning_effort") == "minimal":
            # Minimal is not off. This diagnostic may establish unsupported disable,
            # but never authorize it for a no-reasoning bulk run.
            text, reason = None, "minimal_not_disabled"
        attempts.append({"label": r["target_label"], "api_model": model, "options": options,
                         "reported_model": reported, "reasoning_tokens": count,
                         "reasoning_check": status, "error": reason, "usage": response.get("usage"),
                         "accepted": text is not None})
        if text is not None:
            return r["target_label"], {"api_model": model, "options": options,
                    "reported_model": reported, "reasoning_check": status}, attempts
    return r["target_label"], None, attempts


if (OUT / "profile_attempts.jsonl").exists():
    raise SystemExit("Checks already attempted; refusing duplicate billed calls.")
with (OUT / "profile_attempts.jsonl").open("w") as f, ThreadPoolExecutor(max_workers=4) as pool:
    for label, profile, attempts in pool.map(check, pending):
        for attempt in attempts:
            f.write(json.dumps(attempt) + "\n")
        f.flush()
        if profile:
            profiles[label] = profile
        print(label, "compatible" if profile else "no verified disabling profile", flush=True)
(OUT / "profiles.json").write_text(json.dumps(profiles, indent=2))
print("Usable profiles:", len(profiles), flush=True)
