import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.data import ROOT, load_test, load_train

THRESHOLD = 0.20
HOLDOUT_SIZE = 2_000
TARGET = "is_fraud"
SIGNALS = {
    "account_type": "Account type",
    "typical_monthly_spend_gbp": "Typical monthly spend (GBP)",
    "transaction_type": "Transaction type",
    "amount_gbp": "Amount (GBP)",
    "amount_vs_typical": "Amount / typical monthly spend",
    "hour_of_day": "Transaction hour",
    "merchant_country": "Merchant country",
    "foreign_transaction": "Foreign transaction",
    "unknown_device": "Unknown device",
    "failed_logins": "Failed logins",
    "new_payee": "New payee",
    "travel_notice": "Travel notice",
}
BASELINE_FEATURES = [
    "amount_gbp", "amount_vs_typical", "hour_of_day", "foreign_transaction",
    "unknown_device", "failed_logins", "new_payee", "travel_notice",
]


def encode(transactions, columns=None):
    encoded = pd.get_dummies(transactions[list(SIGNALS)])
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded


def fit_boosted(transactions):
    features = encode(transactions)
    model = HistGradientBoostingClassifier(
        max_iter=170, max_leaf_nodes=15, random_state=0
    )
    model.fit(features, transactions[TARGET])
    return model, features.columns


def baseline_features(transactions):
    return transactions[BASELINE_FEATURES].assign(
        is_transfer=transactions["transaction_type"].eq("BANK_TRANSFER"),
        is_online=transactions["transaction_type"].eq("CARD_ONLINE"),
    )


def report(label, actual, flags):
    fraud = actual.to_numpy() == 1
    caught = (fraud & flags).sum()
    false_alarms = (~fraud & flags).sum()
    missed = (fraud & ~flags).sum()
    points = 10 * caught - 5 * false_alarms - 10 * missed
    print(f"{label}: {points:,} points | {caught} caught | "
          f"{false_alarms} false alarms | {missed} missed")


def validate_models(train):
    ordered = train.sort_values("timestamp", kind="stable")
    learn, holdout = ordered.iloc[:-HOLDOUT_SIZE], ordered.iloc[-HOLDOUT_SIZE:]
    print(f"Holdout: {len(holdout):,} transactions, "
          f"{holdout['timestamp'].iloc[0]} to {holdout['timestamp'].iloc[-1]}")

    baseline = DecisionTreeClassifier(max_depth=3, random_state=0)
    baseline.fit(baseline_features(learn), learn[TARGET])
    baseline_risk = baseline.predict_proba(baseline_features(holdout))[:, 1]
    report("Starter tree (30% cutoff)", holdout[TARGET], baseline_risk >= 0.30)

    model, columns = fit_boosted(learn)
    risk = model.predict_proba(encode(holdout, columns))[:, 1]
    report("Gradient boosting (20% cutoff)", holdout[TARGET], risk >= THRESHOLD)


def explain_flags(model, columns, genuine, flagged, risk):
    # Show observed values when replacing the feature with a common genuine
    # value lowers the model's score. This is a local score comparison.
    decreases = np.empty((len(flagged), len(SIGNALS)))
    for index, feature in enumerate(SIGNALS):
        reference = genuine[feature].mode().iloc[0]
        candidate = flagged.copy()
        candidate[feature] = reference
        alternate_risk = model.predict_proba(encode(candidate, columns))[:, 1]
        decreases[:, index] = risk - alternate_risk

    reasons = []
    features = list(SIGNALS)
    for row, values in enumerate(decreases):
        top = np.argsort(-values, kind="stable")[:3]
        details = [
            f"{SIGNALS[features[i]]}: {flagged[features[i]].iat[row]}"
            for i in top if values[i] > 0.001
        ]
        reasons.append(" | ".join(details) or f"Model score {risk[row]:.1%} meets review cutoff")
    return reasons


def make_submission(train, test):
    model, columns = fit_boosted(train)
    risk = model.predict_proba(encode(test, columns))[:, 1]
    flags = risk >= THRESHOLD
    output = test[["transaction_id"]].assign(reasons="", is_flagged=flags.astype(int))
    if flags.any():
        genuine = train.loc[train[TARGET].eq(0)]
        output.loc[flags, "reasons"] = explain_flags(
            model, columns, genuine, test.loc[flags], risk[flags]
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Validate the fraud model and export predictions.")
    parser.add_argument("--output", type=Path, default=ROOT / "test_predictions.csv")
    args = parser.parse_args()
    train = load_train()
    test = load_test()
    with threadpool_limits(limits=2):
        validate_models(train)
        output = make_submission(train, test)
    output.to_csv(args.output, index=False)
    print(f"Saved {args.output}: {len(output):,} rows, {output['is_flagged'].sum()} flagged")


if __name__ == "__main__":
    main()
