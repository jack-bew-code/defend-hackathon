import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.data import ROOT, load_test, load_train
from experiments.jack_tuned_boosting.features import HISTORY_FEATURES, RED_FLAG_FEATURES, add_features

# Flag when P(fraud) > 0.2: flagging is worth 10p - 5(1-p), not flagging is -10p.
THRESHOLD = 0.20
TARGET = "is_fraud"
N_FOLDS = 4
FOLD_SIZE = 1_500
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
WITH_RED_FLAGS = {**SIGNALS, **RED_FLAG_FEATURES}   # used for the submission
WITH_HISTORY = {**SIGNALS, **HISTORY_FEATURES}      # adds per-customer history (didn't help)
# Original settings from fraud/boosted-submission, and the tuned ones: smaller
# trees + slower learning scored higher in every validation fold.
ORIGINAL_PARAMS = dict(max_iter=170, max_leaf_nodes=15)
TUNED_PARAMS = dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=7, l2_regularization=1.0)
BASELINE_FEATURES = [
    "amount_gbp", "amount_vs_typical", "hour_of_day", "foreign_transaction",
    "unknown_device", "failed_logins", "new_payee", "travel_notice",
]


def encode(transactions, signals, columns=None):
    encoded = pd.get_dummies(transactions[list(signals)])
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0)
    return encoded


def fit_boosted(transactions, signals, params=TUNED_PARAMS):
    features = encode(transactions, signals)
    model = HistGradientBoostingClassifier(random_state=0, **params)
    model.fit(features, transactions[TARGET])
    return model, features.columns


def boosted_risk(learn, holdout, signals, params=TUNED_PARAMS):
    model, columns = fit_boosted(learn, signals, params)
    return model.predict_proba(encode(holdout, signals, columns))[:, 1]


def baseline_features(transactions):
    return transactions[BASELINE_FEATURES].assign(
        is_transfer=transactions["transaction_type"].eq("BANK_TRANSFER"),
        is_online=transactions["transaction_type"].eq("CARD_ONLINE"),
    )


def baseline_risk(learn, holdout):
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(baseline_features(learn), learn[TARGET])
    return tree.predict_proba(baseline_features(holdout))[:, 1]


def score(actual, flags):
    fraud = actual.to_numpy() == 1
    caught = (fraud & flags).sum()
    false_alarms = (~fraud & flags).sum()
    missed = (fraud & ~flags).sum()
    return np.array([10 * caught - 5 * false_alarms - 10 * missed, caught, false_alarms, missed])


def time_folds(train):
    """Expanding-window folds: train on everything before each block, test on the block."""
    ordered = train.sort_values("timestamp", kind="stable")
    for k in range(N_FOLDS, 0, -1):
        end = len(ordered) - (k - 1) * FOLD_SIZE
        yield ordered.iloc[:end - FOLD_SIZE], ordered.iloc[end - FOLD_SIZE:end]


def validate_models(train):
    models = {
        "Starter tree": baseline_risk,
        "Boosting, original (teammate's)": lambda a, b: boosted_risk(a, b, SIGNALS, ORIGINAL_PARAMS),
        "Boosting, tuned": lambda a, b: boosted_risk(a, b, SIGNALS),
        "Boosting, tuned + red flags": lambda a, b: boosted_risk(a, b, WITH_RED_FLAGS),
        "Boosting, tuned + all history": lambda a, b: boosted_risk(a, b, WITH_HISTORY),
    }
    totals = {name: np.zeros(4, dtype=int) for name in models}
    print(f"Time-based validation: {N_FOLDS} folds of {FOLD_SIZE:,}, all at {THRESHOLD:.0%} cutoff")
    for fold, (learn, holdout) in enumerate(time_folds(train), 1):
        print(f"  Fold {fold}: {holdout['timestamp'].iloc[0][:10]} to {holdout['timestamp'].iloc[-1][:10]}")
        for name, risk_fn in models.items():
            result = score(holdout[TARGET], risk_fn(learn, holdout) >= THRESHOLD)
            totals[name] += result
            print(f"    {name:32s} {result[0]:>6,} pts")
    perfect = 10 * sum(holdout[TARGET].sum() for _, holdout in time_folds(train))
    print(f"Totals across folds (perfect score would be {perfect:,}):")
    for name, (points, caught, false_alarms, missed) in totals.items():
        print(f"  {name:32s} {points:>6,} pts | {caught} caught | "
              f"{false_alarms} false alarms | {missed} missed")


def explain_flags(model, columns, signals, genuine, flagged, risk):
    # Swap each feature for a typical genuine value (median for numbers, most
    # common value for categories) and see how much the fraud score drops.
    features = list(signals)
    decreases = np.empty((len(flagged), len(features)))
    for index, feature in enumerate(features):
        values = genuine[feature]
        reference = values.median() if pd.api.types.is_numeric_dtype(values) else values.mode().iloc[0]
        candidate = flagged.copy()
        candidate[feature] = reference
        alternate_risk = model.predict_proba(encode(candidate, signals, columns))[:, 1]
        decreases[:, index] = risk - alternate_risk

    reasons = []
    for row, values in enumerate(decreases):
        top = np.argsort(-values, kind="stable")[:3]
        details = []
        for i in top:
            if values[i] <= 0.001:
                continue
            value = flagged[features[i]].iat[row]
            if isinstance(value, float):
                value = round(value, 2)
            details.append(f"{signals[features[i]]}: {value}")
        reasons.append(" | ".join(details) or f"Model score {risk[row]:.1%} meets review cutoff")
    return reasons


def make_submission(train, test, signals):
    model, columns = fit_boosted(train, signals)
    risk = model.predict_proba(encode(test, signals, columns))[:, 1]
    flags = risk >= THRESHOLD
    output = test[["transaction_id"]].assign(reasons="", is_flagged=flags.astype(int))
    if flags.any():
        genuine = train.loc[train[TARGET].eq(0)]
        output.loc[flags, "reasons"] = explain_flags(
            model, columns, signals, genuine, test.loc[flags], risk[flags]
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Validate the fraud model and export predictions.")
    parser.add_argument("--output", type=Path, default=ROOT / "test_predictions.csv")
    parser.add_argument("--features", choices=["red-flags", "original", "all-history"],
                        default="red-flags", help="Feature set for the submission model.")
    args = parser.parse_args()

    train, test = load_train(), load_test()
    # Build history on train + test together so September rows can see each
    # customer's earlier transactions. Features only look backwards in time.
    both = add_features(pd.concat([train, test], keys=["train", "test"]))
    train, test = both.loc["train"], both.loc["test"].drop(columns=[TARGET])
    train[TARGET] = train[TARGET].astype(int)

    signals = {"red-flags": WITH_RED_FLAGS, "original": SIGNALS, "all-history": WITH_HISTORY}[args.features]
    with threadpool_limits(limits=2):
        validate_models(train)
        output = make_submission(train, test, signals)
    output.to_csv(args.output, index=False)
    print(f"Saved {args.output}: {len(output):,} rows, {output['is_flagged'].sum()} flagged "
          f"({args.features} features)")


if __name__ == "__main__":
    main()
