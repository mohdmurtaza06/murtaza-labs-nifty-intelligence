# Release readiness and handoff

**Package type:** source release / research-only  
**Live trading:** disabled  
**Trading verdict:** `NO_TRADE`  
**Hosting:** automatic Vercel deployments disabled by repository configuration  
**Real-money approval:** `NO`

## What this package contains

The active source tree is at repository root: `backend/`, `notebooks/`, `src/`, `tests/`, and `web/`. `StockMarket_UI_Updated/` is an earlier project snapshot and is preserved under `legacy/StockMarket_UI_Updated/` in the release archive for reference only. Do not run its model or data as though they belong to the active API.

A passing GitHub Actions run covers the automated Python safety tests, Python syntax compilation, and frontend production build. It does **not** prove current data collection works on the owner's Windows machine or establish that the strategy is profitable.

## Known runtime requirements / release blockers

The repository itself does not contain a verified, current active-root market dataset, the canonical ordered 172-feature manifest, a compatible promoted model artifact, a preserved clean holdout report, or 200 closed prospective paper-trading outcomes. Legacy/sample market data and a legacy model exist only in the preserved snapshot and are not evidence for the active model.

The owner must supply licensed/current input files, including:
- `data/live_market_data.csv` through the latest completed NSE equity session.
- `data/nifty50_external_features.csv` with point-in-time aligned external feeds for all required historical sessions.
- The canonical feature manifest and a model candidate whose ordered feature names and hash match that manifest.
- A newly registered, unexposed chronological holdout and a calibration report that uses prior chronological data only.
- An append-only prospective paper-trading ledger and independent NIFTYBEES reconciliation.

The available audits recorded stale market data ending 2026-09-22, historical external-session alignment gaps, a retired/compromised prior holdout, and zero verified prospective paper observations. These are not repaired by a source-code build. Do not reuse a compromised holdout or count historical/backfilled rows toward the prospective 200-signal gate.

## Local setup

Use a supported Python environment, preferably Python 3.11 to match CI.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q backend src notebooks tests
```

In another PowerShell terminal:

```powershell
Set-Location web
Copy-Item .env.example .env.local
npm ci
npm run dev
```

Start the backend from the repository root:

```powershell
python -m uvicorn backend.main:app --reload
```

The API defaults to research-only behavior. The in-process daily scheduler is disabled unless `NIFTY_ENABLE_SCHEDULER=true` is set in the environment. Keep it disabled by default; do not use the process-local scheduler as a multi-worker production service.

The dashboard defaults to `http://127.0.0.1:8000/api`. `GET /api/health` reports process liveness, while `GET /api/readiness` reports trading release readiness. A working API response is not approval to trade.

## Required release verdict

Until the blockers above are independently resolved, the only supported operating state is:

```json
{
  "trading_mode": "RESEARCH_ONLY",
  "trade_status": "NO_TRADE",
  "production_eligible": false,
  "live_orders_enabled": false,
  "real_money_approval": "NO"
}
```

No broker order-placement adapter is included in this release. No Vercel production deployment is authorized. This package is a consolidated source release, not a claim of investment readiness, calibrated 70% accuracy, profitability, or operational approval for real-money trades.
