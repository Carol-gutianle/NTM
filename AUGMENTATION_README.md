# RubricLab gateway augmentation pilot

> Archive update (2026-10-27): the 200-per-class batch has stopped and its
> evaluation is complete, with 2,025 accepted rows across 12 of 18 classes
> (not full completion). Holdout accuracy decreased from 31.85% to 28.33%.
> See `results/README.md` and `results/evidence/augmentation200/`.
> In this repository, credentials default to the local ignored `.env`;
> set `RUBRICLAB_ENV=/path/to/your/.env` to select another file explicitly.
> The notes below preserve the original pilot/running-batch context.

`augment_litellm.py` reads ONLY the endpoint, API key and optional gateway cookie
from the user-designated RubricLab `.env`. It never executes shell dotenv code,
prints credentials, uploads credentials, or modifies that project.
HTTP redirects are refused to avoid forwarding credentials elsewhere.

```bash
python augment_litellm.py --discover
python augment_litellm.py --prompts 2 --max-requests 30 --max-tokens 768 --workers 2
```

Actual generation consumes API quota. Limits are requested output tokens, NOT
a guaranteed bill cap: gateways may account for reasoning or ignore limits.
There are no automatic retries or silent repeat runs.

New runs explicitly request `reasoning_effort="none"` by default. This is a
gateway request, not proof that every backend supports disabling reasoning.
Unsupported requests fail without retrying/removing the parameter. If the gateway
specifically supports the alternate value, use `--reasoning-effort disable`.
No backend-specific mapping is assumed automatically.
Responses reporting positive reasoning tokens are quarantined as
`reasoning_not_disabled`. Missing reasoning usage is recorded as `unknown`,
not zero; accepted candidates with unknown usage still require audit.
Only final `message.content` is retained, not separate reasoning content.
The first pilot below predates this explicit no-reasoning setting.

Original prompts are shared across the available exact-match target IDs.
No original competition text is sent to the gateway. No imitation is relabeled
as another model. Model identities are gateway-reported, not independently verified.
The model list currently lacks three of the 18 target labels.

Responses with a different/missing model name, non-stop finish, empty/short
answer, or exact duplicate are excluded from `augmentation_candidates.csv`;
local response records are preserved for audit. Candidate CSV is NOT merged into
training automatically, and candidate coverage need not be balanced after filtering.
Novel prompt generation alone does not prove semantic separation from existing
validation prompts; audit before using it. External-data competition compliance
must also be confirmed before using these data for a submission.

## First pilot, 2026-10-23

- 15 exact-name models advertised, 2 original prompts, 30 generation calls.
- 14 provisional accepted responses across 8 labels.
- 16 non-stop/incomplete outputs excluded. Reported usage often spends most of
  the 768-token request budget on reasoning, leaving no complete final answer.
- Gateway reported 20,500 total tokens and `cost` sum 1.31726982 (currency unknown).
  Some responses report more completion tokens than the requested limit.
- No original dataset modified, no retraining or Kaggle submission.
- Results: `artifacts/augmentation_pilot_first/`.

Before scaling: resolve missing model IDs, audit backend identity, configure
reasoning/output budgets per model, rerun a bounded pilot, verify complete
answers and balance, then perform a frozen-split training comparison.

## 200-per-class run

`generate_augmentation200.py` uses an original 25-scenario × 8-task grid shared
across all eligible models. It loads explicit per-model options from
`artifacts/augmentation200/profiles.json`. Exact Claude punctuation aliases were
added after the first pilot. Of the 18 classes, 12 passed bounded compatibility
checks, 5 failed no-reasoning checks and one has no corresponding advertised ID.
These gaps are never filled with another version's text.

The running batch aims at 200 answers per eligible class, not a promise of
200 valid responses: timeouts are not blindly repeated, accepted outputs are
deduplicated, and repeated reasoning/model/parameter failures trip a circuit
breaker. Reservations are journaled before requests. An interrupted request
without a recorded response is ambiguous and is not automatically billed again.
The full manifest is immutable across resumes, and a file lock prevents overlap.

```bash
tail -f artifacts/augmentation200/bulk.log
cat artifacts/augmentation200/progress.json
cat artifacts/augmentation200/evaluation_status.json
```

`finish_augmentation_eval.py` waits for generation to release its lock, snapshots
the candidate CSV, then runs `evaluate_augmentation.py`: same original grouped
validation split and character-SVM settings, original-only vs original+external.
It explicitly labels missing classes and partial completion, screens duplicate
answers, and verifies the original baseline predictions are reproduced.
Generation costs have unspecified currency and are not a guaranteed bill cap.
No Kaggle submission is made.
