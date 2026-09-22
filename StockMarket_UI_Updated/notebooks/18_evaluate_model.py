import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TRACKER_FILE = DATA_DIR / "prediction_history.csv"
REPORT_FILE = DATA_DIR / "model_evaluation.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("NIFTY 50 MODEL EVALUATION")
print("=" * 60)

history = pd.read_csv(
    TRACKER_FILE
)

print(
    f"\nTotal predictions recorded: {len(history)}"
)


# ============================================================
# KEEP ONLY VERIFIED PREDICTIONS
# ============================================================

verified = history[
    history["Actual_Next_Day"].isin(
        ["UP", "DOWN"]
    )
].copy()


if verified.empty:

    print("\nNo verified predictions yet.")

    print(
        "\nRun this script again after "
        "some predictions have been verified."
    )

    exit()


print(
    f"Verified predictions: {len(verified)}"
)


# ============================================================
# PREPARE LABELS
# ============================================================

y_true = verified[
    "Actual_Next_Day"
]

y_pred = verified[
    "Prediction"
]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label="UP",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label="UP",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label="UP",
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["DOWN", "UP"]
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)


print("\nConfusion Matrix")
print(
    "Rows = Actual"
)
print(
    "Columns = Predicted"
)

print(
    "          DOWN    UP"
)

print(
    f"DOWN      {cm[0][0]:5d}   {cm[0][1]:5d}"
)

print(
    f"UP        {cm[1][0]:5d}   {cm[1][1]:5d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print("=" * 60)
print("CONFIDENCE ANALYSIS")
print("=" * 60)


verified["Confidence_Percent"] = (
    verified["Confidence"] * 100
)


confidence_groups = [

    ("50-55%", 50, 55),

    ("55-60%", 55, 60),

    ("60-70%", 60, 70),

    ("70%+", 70, 101)

]


for name, lower, upper in confidence_groups:

    group = verified[
        (
            verified["Confidence_Percent"]
            >= lower
        )
        &
        (
            verified["Confidence_Percent"]
            < upper
        )
    ]


    if len(group) == 0:

        print(
            f"\n{name}: No predictions"
        )

        continue


    group_accuracy = (
        group["Prediction"]
        == group["Actual_Next_Day"]
    ).mean()


    print(
        f"\n{name}"
    )

    print(
        f"Predictions: {len(group)}"
    )

    print(
        f"Accuracy: "
        f"{group_accuracy * 100:.2f}%"
    )


# ============================================================
# DIRECTION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    verified["Prediction"]
    .value_counts()
)


print("\nActual distribution:")

print(
    verified["Actual_Next_Day"]
    .value_counts()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Verified Predictions",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        len(verified),
        accuracy,
        precision,
        recall,
        f1
    ]

})


summary.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\nEvaluation saved to:")

print(
    REPORT_FILE
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)