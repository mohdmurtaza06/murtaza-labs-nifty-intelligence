from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import subprocess
import sys
import threading

app = FastAPI(title="Murtaza Labs NIFTY Intelligence API")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_bool(value):
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return True

    if value in ["false", "0", "no"]:
        return False

    return None


def get_dashboard_data():
    prediction_file = DATA_DIR / "live_prediction.csv"
    history_file = DATA_DIR / "prediction_history.csv"

    if not prediction_file.exists():
        raise HTTPException(
            status_code=404,
            detail="live_prediction.csv not found"
        )

    prediction = pd.read_csv(prediction_file)

    if prediction.empty:
        raise HTTPException(
            status_code=404,
            detail="No prediction available"
        )

    latest = prediction.iloc[-1]

    history = pd.DataFrame()

    if history_file.exists():
        history = pd.read_csv(history_file)

    correct_values = []

    if not history.empty and "Correct" in history.columns:
        for value in history["Correct"]:
            parsed = parse_bool(value)
            if parsed is not None:
                correct_values.append(parsed)

    verified = len(correct_values)
    correct = sum(correct_values)

    accuracy = None

    if verified > 0:
        accuracy = (correct / verified) * 100

    return {
        "date": str(latest.get("Prediction_Date", "")),
        "nifty_close": float(latest.get("NIFTY_Close", 0)),
        "prediction": str(latest.get("Prediction", "UNKNOWN")),
        "probability_up": float(latest.get("Probability_UP", 0)),
        "probability_down": float(latest.get("Probability_DOWN", 0)),
        "confidence": float(latest.get("Confidence", 0)),
        "signal": str(latest.get("Signal", "UNKNOWN")),
        "total_predictions": int(len(history)),
        "verified_predictions": int(verified),
        "correct": int(correct),
        "accuracy": accuracy,
        "backend": "online",
    }


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Murtaza Labs NIFTY Intelligence API"
    }


@app.get("/api/dashboard")
def dashboard():
    return get_dashboard_data()


@app.get("/api/history")
def history():
    file = DATA_DIR / "prediction_history.csv"

    if not file.exists():
        return []

    df = pd.read_csv(file)

    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


@app.get("/api/market")
def market():
    file = DATA_DIR / "live_market_data.csv"

    if not file.exists():
        raise HTTPException(
            status_code=404,
            detail="live_market_data.csv not found"
        )

    df = pd.read_csv(file)

    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)

    df = df.tail(180)

    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient="records")


@app.get("/api/download/prediction")
def download_prediction():
    file = DATA_DIR / "live_prediction.csv"

    if not file.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction file not found"
        )

    return FileResponse(
        path=file,
        filename="murtaza-labs-live-prediction.csv",
        media_type="text/csv"
    )


@app.get("/api/download/history")
def download_history():
    file = DATA_DIR / "prediction_history.csv"

    if not file.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction history not found"
        )

    return FileResponse(
        path=file,
        filename="murtaza-labs-prediction-history.csv",
        media_type="text/csv"
    )


pipeline_lock = threading.Lock()


def run_pipeline_background():
    try:
        script = PROJECT_DIR / "notebooks" / "19_daily_pipeline.py"

        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )

    except Exception as e:
        print("Pipeline error:", e)

    finally:
        try:
            pipeline_lock.release()
        except RuntimeError:
            pass


@app.post("/api/run-pipeline")
def run_pipeline():
    if not pipeline_lock.acquire(blocking=False):
        return {
            "status": "already_running",
            "message": "Daily pipeline is already running."
        }

    thread = threading.Thread(
        target=run_pipeline_background,
        daemon=True
    )

    thread.start()

    return {
        "status": "started",
        "message": "Daily pipeline started."
    }