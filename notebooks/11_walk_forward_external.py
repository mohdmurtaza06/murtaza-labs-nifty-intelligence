import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# ==================================================
# LOAD DATA
# ==================================================

data = pd.read_csv(
    DATA_DIR / "nifty50_external_features.csv",
    index_col="Date",
    parse_dates=True
)


# ==================================================
# TECHNICAL FEATURES
# ==================================================

technical_features = [
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Price_to_MA5",
    "Price_to_MA20",
    "Price_to_MA50",
    "High_Low_Range",
    "Open_Close_Return",
    "Volatility_10",
    "Volatility_20",
    "Volume_Change",
    "Relative_Volume",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Position",
    "ATR_Percent"
]


# ==================================================
# EXTERNAL FEATURES
# ==================================================

external_features = [
    "India_VIX_Return_1D",
    "India_VIX_Return_5D",

    "NIFTY_Bank_Return_1D",
    "NIFTY_Bank_Return_5D",

    "SP500_Return_1D",
    "SP500_Return_5D",

    "USD_INR_Return_1D",
    "USD_INR_Return_5D",

    "Gold_Return_1D",
    "Gold_Return_5D",

    "Crude_Oil_Return_1D"
]


all_features = (
    technical_features +
    external_features
)


# ==================================================
# CLEAN
# ==================================================

model_data = data[
    all_features + ["Target", "Close"]
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()

model_data["Target"] = (
    model_data["Target"].astype(int)
)


# ==================================================
# MODEL FUNCTION
# ==================================================

def train_model(
    train_data,
    test_data,
    features
):

    X_train = train_data[features]
    y_train = train_data["Target"]

    X_test = test_data[features]
    y_test = test_data["Target"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy


# ==================================================
# WALK-FORWARD SETTINGS
# ==================================================

initial_train_size = int(
    len(model_data) * 0.60
)

test_size = 252

results = []

start = initial_train_size


# ==================================================
# WALK-FORWARD
# ==================================================

while start < len(model_data):

    train_end = start
    test_end = min(
        start + test_size,
        len(model_data)
    )

    train_data = model_data.iloc[
        :train_end
    ]

    test_data = model_data.iloc[
        start:test_end
    ]

    print("\n========================================")
    print("WALK-FORWARD PERIOD")
    print("========================================")

    print(
        "Training:",
        train_data.index.min(),
        "to",
        train_data.index.max()
    )

    print(
        "Testing:",
        test_data.index.min(),
        "to",
        test_data.index.max()
    )


    # ==================================================
    # TECHNICAL MODEL
    # ==================================================

    technical_accuracy = train_model(
        train_data,
        test_data,
        technical_features
    )


    # ==================================================
    # EXTERNAL MODEL
    # ==================================================

    external_accuracy = train_model(
        train_data,
        test_data,
        all_features
    )


    improvement = (
        external_accuracy -
        technical_accuracy
    )


    print(
        f"\nTechnical accuracy: "
        f"{technical_accuracy * 100:.2f}%"
    )

    print(
        f"External accuracy: "
        f"{external_accuracy * 100:.2f}%"
    )

    print(
        f"Improvement: "
        f"{improvement * 100:+.2f} percentage points"
    )


    results.append({
        "Test_Start": test_data.index.min(),
        "Test_End": test_data.index.max(),
        "Technical_Accuracy": technical_accuracy,
        "External_Accuracy": external_accuracy,
        "Improvement": improvement
    })


    start = test_end


# ==================================================
# RESULTS
# ==================================================

results_df = pd.DataFrame(results)


print("\n\n========================================")
print("WALK-FORWARD RESULTS")
print("========================================")

print(
    results_df.to_string(index=False)
)


# ==================================================
# OVERALL
# ==================================================

technical_mean = (
    results_df[
        "Technical_Accuracy"
    ].mean()
)

external_mean = (
    results_df[
        "External_Accuracy"
    ].mean()
)

mean_improvement = (
    external_mean -
    technical_mean
)


print("\n========================================")
print("OVERALL RESULTS")
print("========================================")


print(
    f"\nAverage technical accuracy: "
    f"{technical_mean * 100:.2f}%"
)

print(
    f"Average external accuracy: "
    f"{external_mean * 100:.2f}%"
)

print(
    f"\nAverage improvement: "
    f"{mean_improvement * 100:+.2f} percentage points"
)


# ==================================================
# SAVE
# ==================================================

output_file = (
    DATA_DIR /
    "walk_forward_external_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved to:")
print(output_file)