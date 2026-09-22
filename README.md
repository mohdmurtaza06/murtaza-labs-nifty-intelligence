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
