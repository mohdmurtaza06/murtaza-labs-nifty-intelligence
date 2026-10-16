# NIFTY 50 Intelligence

The active application consists of a FastAPI backend in `backend/`, a React
dashboard in `web/`, and the production prediction pipeline in `notebooks/`.

## Setup

Create a virtual environment, install `requirements.txt`, then start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

In another terminal, configure and start the dashboard:

```powershell
Set-Location web
Copy-Item .env.example .env.local
npm.cmd install
npm.cmd run dev
```

The frontend reads `VITE_API_BASE_URL` at build/start time. The checked-in
`web/.env.example` points to the local API at
`http://127.0.0.1:8000/api`. For any future hosted frontend, configure this
variable with the explicitly approved API URL; do not point a release build at
an unverified service.

## Daily research-signal flow

`notebooks/27_build_production_model.py` creates a **research candidate** at
`models/nifty50_research_candidate.pkl`. It does not create an approved
production model.

`notebooks/19_daily_pipeline.py` runs the research predictor
(`notebooks/15_live_prediction_production.py`), records the same signal in
`data/prediction_history.csv`, verifies prior predictions against the next
session's open-to-close outcome, and refreshes evaluation results. The
in-process API scheduler is **disabled by default**. Set
`NIFTY_ENABLE_SCHEDULER=true` only for a single-process local research run; it
checks weekdays at 16:00 Asia/Kolkata time. Do not use this process-local
scheduler as a production multi-worker job service.

Research directions are displayed for analysis only. API responses identify
`trade_status=NO_TRADE`, `trading_mode=RESEARCH_ONLY`,
`production_eligible=false`, and `live_orders_enabled=false`. The separate
production-prediction endpoint intentionally returns unavailable until the
appropriate release gates are satisfied.

Machine-readable probabilities (`probability_up`, `probability_down`, and
`confidence`) use the 0–1 range. The frontend converts them to percentages.

## Safety checks and current trading status

The API has separate liveness and release-readiness reporting. Open
`GET /api/health` to confirm the process responds and `GET /api/readiness` to
inspect evidence on disk. The readiness inspector checks current-root market
data, the canonical feature manifest/model metadata, calibration metadata,
the one-time holdout report, the structured prospective paper ledger, and
NIFTYBEES reconciliation. Missing, malformed, stale, or incompatible artifacts
are reported as failed/unverified. The endpoint always leaves
`production_eligible=false` and `live_orders_enabled=false`; an evidence
report cannot itself authorize real-money execution.

The project now includes shared fail-closed checks in `src/readiness_checks.py`:

- next-session labels with missing/infinite outcomes stay missing and are excluded from training;
- production prediction generation stops if the live-data refresh fails;
- the latest market observation is checked for a future date or excessive calendar age;
- probability values must be finite, in `[0, 1]`, and sum to approximately one;
- the API refuses to serve stale or internally inconsistent prediction artifacts.

Production freshness checks require the last completed NSE equity session to
match the observation date resolved by `pandas-market-calendars`. The package
calendar is shipped with the installed dependency and is not a live NSE feed:
keep the dependency updated and verify ad-hoc/exceptional exchange holidays.
If the calendar cannot be loaded or the latest bar does not match the expected
completed session, prediction publication is blocked.

The research training target is defined as the **next session's open-to-close return direction**, because the prediction is generated after today's close and cannot be executed at today's already-known closing price. Retrain models after pulling this change; models trained under the previous target definition are not interchangeable.

### Tests

From the repository root, run:

```powershell
python -m pip install numpy pandas fastapi pandas-market-calendars
python -m unittest discover -s tests -v
python -m compileall -q backend src notebooks tests
```

GitHub Actions runs these regression tests and Python syntax checks on pushes to the safety-fix branch and on pull requests.

### Manual pipeline trigger security

The `POST /api/run-pipeline` endpoint launches a subprocess and now requires a server-side `PIPELINE_TRIGGER_TOKEN` environment variable plus a matching `X-Pipeline-Token` header. If no token is configured, manual triggering is disabled. The dashboard prompts for the token per action and does not write it to local storage or bundle it into the Vite assets.

On a local PowerShell session, set the value **before** starting the backend:

```powershell
$env:PIPELINE_TRIGGER_TOKEN = "replace-with-a-long-random-secret"
python -m uvicorn backend.main:app --reload
```

Set the equivalent private environment variable in the backend hosting provider for a deployed API. Do not prefix it with `VITE_`, commit it, put it into frontend source, or share it in screenshots. Use HTTPS for remote requests. A production-grade user identity/authorization flow is preferable to a shared trigger token.

### Trading safety

- Live order placement remains disabled and must not be enabled by this branch.
- A directional probability is not a profit guarantee or a calibrated probability unless measured and validated separately.
- The project must pass an untouched chronological holdout, realistic cost-aware strategy evaluation, prospective paper trading, and independent instrument reconciliation before any real-money decision.
- A successful software build or passing unit tests do not certify a profitable strategy.

### Chronological model validation

Refresh data first, then run this one-shot evaluation from the repository root:

```powershell
python notebooks/28_time_split_validation.py
```

The script reserves the last 252 labeled rows as a chronological holdout,
selects features and imputations using the training segment only, calibrates
probabilities on a separate earlier segment, and scores the holdout with a
configurable round-trip cost assumption (default 20 basis points). It compares
a model-directed long-only next-open-to-close strategy with an always-long
strategy on the same horizon. It writes a JSON report and holdout prediction
CSV under `reports/validation/`, never overwrites the report by default, and
can only return `PASS_FOR_PAPER_TRADING_ONLY` or `HOLD_NO_TRADE`. Even a
passing holdout is not production approval. Do not inspect the holdout repeatedly
while tuning models; register a new holdout for a genuine retest.

`NIFTY_ROUND_TRIP_COST_BPS` can override the illustrative 20-basis-point cost
assumption. Replace this with costs measured for the actual intended instrument
and broker before interpreting trading returns.

## Consolidated source release

On a successful GitHub Actions run for `fix/production-readiness-gates`, the
`source-package` job builds one ZIP plus a SHA-256 sidecar and uploads them
as the `MURTAZA_LABS_NIFTY_INTELLIGENCE_SOURCE_RELEASE` workflow artifact.
The package has a `RELEASE_MANIFEST.json` with commit identity, per-file
hashes, artifact availability, and the trading/hosting state. The archive
preserves `StockMarket_UI_Updated` under `legacy/` so it cannot be confused
with the active root application.

The package is labelled a source release because the Git repository does not
contain active-root current market data, the canonical 172-feature manifest,
or a promoted model binary. The historical snapshot is not a substitute. See
[`RELEASE_READINESS.md`](RELEASE_READINESS.md) before running model training.
Automatic Vercel deployments are disabled in the active repository and web
root configuration until explicitly re-enabled by the owner.
