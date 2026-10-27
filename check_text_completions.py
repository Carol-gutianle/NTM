"""Bounded gateway text-completion probe; not augmentation or native-backend proof."""
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener

from augment_litellm import credentials, ENV, NoRedirect, resolve_models, request_json, reasoning_status

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/text_completions_probe"


def main():
    if OUT.exists():
        raise SystemExit("Probe output exists; refusing duplicate billed calls.")
    OUT.mkdir()
    secrets, chat = credentials(ENV)
    base = chat[:-len("/chat/completions")]
    parsed = urlsplit(base)
    routes = list(dict.fromkeys([
        base + "/completions",
        urlunsplit((parsed.scheme, parsed.netloc, "/v1/completions", "", "")),
    ]))
    listing, error = request_json(base + "/models", secrets, timeout=30)
    if error:
        raise SystemExit("Model listing failed: " + error)
    labels = sorted({r["model"] for r in csv.DictReader((ROOT / "train.csv").open())})
    mapping = resolve_models(labels, {r["id"] for r in listing.get("data", [])})
    all_results = []

    def redact(value):
        text = str(value)
        for secret in secrets.values():
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = re.sub(r'https?://[^\s"\']+', "[URL]", text)
        return text[:1200]

    def probe(label, route):
        payload = {
            "model": mapping[label],
            "prompt": "Continue this sentence in one short sentence: The little boat reached the shore and",
            "max_tokens": 96, "stream": False,
        }
        headers = {"Authorization": "Bearer " + secrets["OPENAI_API_KEY"],
                   "Content-Type": "application/json"}
        if secrets.get("LLM_COOKIE"):
            headers["Cookie"] = secrets["LLM_COOKIE"]
        result = {"label": label, "requested_model": mapping[label],
                  "route": urlsplit(route).path, "request_keys": sorted(payload),
                  "max_tokens_requested": 96,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
        start = time.monotonic()
        try:
            req = Request(route, data=json.dumps(payload).encode(), headers=headers, method="POST")
            with build_opener(NoRedirect()).open(req, timeout=45) as response:
                result["http_status"] = response.status
                body = json.loads(response.read())
            choices = body.get("choices") or []
            choice = choices[0] if choices else {}
            text = choice.get("text")
            result.update(
                reported_model=body.get("model"), object=body.get("object"),
                response_keys=list(body), choice_keys=list(choice),
                finish_reason=choice.get("finish_reason"),
                text_completion_schema=isinstance(text, str),
                nonempty_text=isinstance(text, str) and bool(text.strip()),
                text_preview=redact(text) if isinstance(text, str) else None,
                usage=body.get("usage"),
                reasoning_status=reasoning_status(body)[1],
            )
            if body.get("error"):
                result["error"] = redact(body["error"])
        except HTTPError as exc:
            result["http_status"] = exc.code
            # Read only a bounded body; remove credentials/URLs before storage.
            raw = exc.read(8192).decode(errors="replace")
            try:
                data = json.loads(raw)
                err = data.get("error", data.get("detail", {}))
                result["error"] = redact(err)
            except ValueError:
                result["error"] = "Non-JSON error body omitted"
        except (URLError, TimeoutError):
            result["error"] = "network_error_or_timeout"
        except (ValueError, OSError):
            result["error"] = "invalid_response_or_io"
        result["seconds"] = round(time.monotonic()-start, 2)
        return result

    def record(r, log):
        all_results.append(r)
        log.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.flush()
        print(r["requested_model"], r["route"], r.get("http_status"),
              "text_schema=" + str(r.get("text_completion_schema")),
              r.get("error", ""), flush=True)

    with (OUT / "results.jsonl").open("w") as log:
        # Two model families per route before declaring route globally missing.
        seeds = [x for x in ["gpt-5.5", "claude-opus-4-8"] if x in mapping]
        active = []
        for route in routes:
            results = [probe(label, route) for label in seeds]
            for result in results:
                record(result, log)
            # 400/422 may be model incompatibility; check all IDs on that route.
            # Auth failures, missing routes and transport failures don't justify
            # a large set of identical probes.
            if any(r.get("http_status") in {200, 400, 422} for r in results):
                active.append(route)
        for route in active:
            remaining = [x for x in mapping if x not in seeds]
            with ThreadPoolExecutor(max_workers=3) as pool:
                for result in pool.map(lambda label: probe(label, route), remaining):
                    record(result, log)
    summary = {
        "requests": len(all_results), "advertised_target_models": mapping,
        "missing_target_models": sorted(set(labels)-set(mapping)),
        "successful_text_responses": [
            {"label": r["label"], "route": r["route"], "finish_reason": r.get("finish_reason")}
            for r in all_results if r.get("http_status") == 200 and r.get("nonempty_text")],
        "note": "Minimal prompt-format probes, no explicit reasoning controls: this checks "
                "endpoint/model compatibility, NOT disabled reasoning or native raw completion. "
                "No automatic retries, no training texts transmitted, no responses merged.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
