# NIFTY 50 Intelligence

The active application consists of a FastAPI backend in `backend/`, a React
dashboard in `web/`, and the production prediction pipeline in `notebooks/`.

## Setup

Create a virtual environment, install `requirements.txt`, then start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

In another terminal, start the dashboard:

```powershell
Set-Location web
npm.cmd install
npm.cmd run dev
```

## Daily prediction flow

`notebooks/19_daily_pipeline.py` runs the production predictor, records that
same prediction in `data/prediction_history.csv`, verifies prior predictions,
and updates evaluation results. The API scheduler triggers this pipeline on
weekdays at 15:45 Asia/Kolkata time.

Machine-readable probabilities (`probability_up`, `probability_down`, and
`confidence`) use the 0–1 range. The frontend converts them to percentages.

## Safety checks and current trading status

The project now includes shared fail-closed checks in `src/readiness_checks.py`:

- next-session labels with missing/infinite outcomes stay missing and are excluded from training;
- production prediction generation stops if the live-data refresh fails;
- the latest market observation is checked for a future date or excessive calendar age;
- probability values must be finite, in `[0, 1]`, and sum to approximately one;
- the API refuses to serve stale or internally inconsistent prediction artifacts.

The freshness guard currently uses a conservative four-calendar-day maximum. It is **not** a substitute for an authoritative NSE session calendar. Long holiday closures may require a deliberate fail-closed review. Integrating a maintained official session calendar remains a readiness task.

The production training target is defined as the **next session's open-to-close return direction**, because the prediction is generated after today's close and cannot be executed at today's already-known closing price. Retrain models after pulling this change; models trained under the previous target definition are not interchangeable.

### Tests

From the repository root, run:

```powershell
python -m pip install numpy pandas
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
