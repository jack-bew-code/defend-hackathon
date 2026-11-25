
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score
from threadpoolctl import threadpool_limits


TARGET = "is_fraud"
ID = "transaction_id"

HOLDOUT_SIZE = 2000

CATEGORICAL = [
    "account_type",
    "transaction_type",
    "merchant_country",
]

NUMERIC = [
    "typical_monthly_spend_gbp",
    "amount_gbp",
    "amount_vs_typical",
    "hour_of_day",
    "foreign_transaction",
    "unknown_device",
    "failed_logins",
    "new_payee",
    "travel_notice",
]

FEATURES = CATEGORICAL + NUMERIC

THRESHOLDS = np.arange(0.05, 0.81, 0.01)


def prepare_data(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    # Time-based features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["is_weekend"] = (
        df["timestamp"].dt.dayofweek >= 5
    ).astype(int)

    # Transaction behaviour
    df["is_transfer"] = (
        df["transaction_type"] == "BANK_TRANSFER"
    ).astype(int)

    df["is_online"] = (
        df["transaction_type"] == "CARD_ONLINE"
    ).astype(int)

    df["is_card"] = (
        df["transaction_type"] == "CARD_IN_STORE"
    ).astype(int)

    df["high_amount"] = (
        df["amount_vs_typical"] >= 1
    ).astype(int)

    df["very_high_amount"] = (
        df["amount_vs_typical"] >= 2
    ).astype(int)

    df["multiple_failed_logins"] = (
        df["failed_logins"] >= 2
    ).astype(int)

    df["suspicious_transfer"] = (
        (df["is_transfer"] == 1)
        & (df["new_payee"] == 1)
        & (df["amount_vs_typical"] >= 1)
    ).astype(int)

    df["device_and_login_risk"] = (
        (df["unknown_device"] == 1)
        & (df["failed_logins"] >= 1)
    ).astype(int)

    df["foreign_without_notice"] = (
        (df["foreign_transaction"] == 1)
        & (df["travel_notice"] == 0)
    ).astype(int)

    df["night_transaction"] = (
        (df["hour_of_day"] <= 5)
        | (df["hour_of_day"] >= 23)
    ).astype(int)

    # Customer-level features
    # Only transaction information available in this dataset is used.
    df["customer_transaction_count"] = (
        df.groupby("customer_id")["customer_id"]
        .transform("count")
    )

    return df


def build_model():
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessing = ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("num", numeric, NUMERIC + [
            "day_of_week",
            "day_of_month",
            "is_weekend",
            "is_transfer",
            "is_online",
            "is_card",
            "high_amount",
            "very_high_amount",
            "multiple_failed_logins",
            "suspicious_transfer",
            "device_and_login_risk",
            "foreign_without_notice",
            "night_transaction",
            "customer_transaction_count",
        ]),
    ])

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=2,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def build_boosting():
    preprocessing = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
        ("num", SimpleImputer(strategy="median"),
         NUMERIC + [
             "day_of_week",
             "day_of_month",
             "is_weekend",
             "is_transfer",
             "is_online",
             "is_card",
             "high_amount",
             "very_high_amount",
             "multiple_failed_logins",
             "suspicious_transfer",
             "device_and_login_risk",
             "foreign_without_notice",
             "night_transaction",
             "customer_transaction_count",
         ]),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=2.0,
            random_state=42,
        )),
    ])


def score_predictions(actual, flags):
    actual = np.asarray(actual).astype(bool)
    flags = np.asarray(flags).astype(bool)

    caught = np.sum(actual & flags)
    false_alarms = np.sum(~actual & flags)
    missed = np.sum(actual & ~flags)

    points = (
        10 * caught
        - 5 * false_alarms
        - 10 * missed
    )

    return points, caught, false_alarms, missed


def find_best_threshold(actual, risk):
    results = []

    for threshold in THRESHOLDS:
        flags = risk >= threshold

        points, caught, false_alarms, missed = (
            score_predictions(actual, flags)
        )

        results.append({
            "threshold": threshold,
            "points": points,
            "caught": caught,
            "false_alarms": false_alarms,
            "missed": missed,
        })

    results = pd.DataFrame(results)

    best = results.sort_values(
        ["points", "false_alarms"],
        ascending=[False, True],
    ).iloc[0]

    print("\nThreshold optimisation:")
    print(results.sort_values(
        "points", ascending=False
    ).head(10).to_string(index=False))

    return float(best["threshold"])


def validate(train):
    train = train.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    if len(train) <= HOLDOUT_SIZE:
        raise ValueError(
            "Not enough training rows for the chosen holdout. "
            "Reduce HOLDOUT_SIZE."
        )

    learn = train.iloc[:-HOLDOUT_SIZE].copy()
    holdout = train.iloc[-HOLDOUT_SIZE:].copy()

    print("\nValidation period:")
    print(holdout["timestamp"].min())
    print("to")
    print(holdout["timestamp"].max())

    learn = prepare_data(learn)
    holdout = prepare_data(holdout)

    model1 = build_model()
    model2 = build_boosting()

    model1.fit(learn, learn[TARGET])
    model2.fit(learn, learn[TARGET])

    risk1 = model1.predict_proba(holdout)[:, 1]
    risk2 = model2.predict_proba(holdout)[:, 1]

    risk = 0.5 * risk1 + 0.5 * risk2

    threshold = find_best_threshold(
        holdout[TARGET],
        risk,
    )

    flags = risk >= threshold

    points, caught, false_alarms, missed = (
        score_predictions(holdout[TARGET], flags)
    )

    print("\nValidation results")
    print("------------------")
    print(f"Threshold: {threshold:.2f}")
    print(f"Points: {points}")
    print(f"Frauds caught: {caught}")
    print(f"False alarms: {false_alarms}")
    print(f"Frauds missed: {missed}")
    print(
        f"Precision: {precision_score(holdout[TARGET], flags, zero_division=0):.3f}"
    )
    print(
        f"Recall: {recall_score(holdout[TARGET], flags, zero_division=0):.3f}"
    )

    return threshold


def explain_transaction(row):
    reasons = []

    if row["amount_vs_typical"] >= 1:
        reasons.append("Amount unusually high")

    if row["new_payee"] == 1:
        reasons.append("New payee")

    if row["failed_logins"] >= 2:
        reasons.append("Multiple failed logins")

    if row["unknown_device"] == 1:
        reasons.append("Unknown device")

    if (
        row["foreign_transaction"] == 1
        and row["travel_notice"] == 0
    ):
        reasons.append("Foreign transaction without travel notice")

    if row["transaction_type"] == "BANK_TRANSFER":
        reasons.append("Bank transfer")

    if row["hour_of_day"] <= 5:
        reasons.append("Overnight transaction")

    return " | ".join(reasons) if reasons else (
        "Model identified elevated combined risk"
    )


def make_submission(train, test, threshold):
    train = prepare_data(train)
    test = prepare_data(test)

    model1 = build_model()
    model2 = build_boosting()

    print("\nTraining final models...")

    model1.fit(train, train[TARGET])
    model2.fit(train, train[TARGET])

    risk1 = model1.predict_proba(test)[:, 1]
    risk2 = model2.predict_proba(test)[:, 1]

    risk = 0.5 * risk1 + 0.5 * risk2

    flags = (risk >= threshold).astype(int)

    output = pd.DataFrame({
        "transaction_id": test[ID],
        "risk_score": risk,
        "reasons": [
            explain_transaction(row)
            for _, row in test.iterrows()
        ],
        "is_flagged": flags,
    })

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("train.csv"),
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=Path("test.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_predictions.csv"),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        f"Training fraud rate: {train[TARGET].mean():.2%}"
    )

    with threadpool_limits(limits=2):

        if args.threshold is None:
            threshold = validate(train)
        else:
            threshold = args.threshold

        output = make_submission(
            train,
            test,
            threshold,
        )

    output.to_csv(args.output, index=False)

    print("\nSubmission saved!")
    print(f"File: {args.output}")
    print(f"Transactions: {len(output):,}")
    print(f"Flagged: {output['is_flagged'].sum():,}")

    print("\nHighest-risk transactions:")
    print(
        output.sort_values(
            "risk_score",
            ascending=False,
        ).head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
