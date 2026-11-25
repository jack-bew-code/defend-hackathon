import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.data import ROOT, load_test, load_train

TARGET = "is_fraud"
HOLDOUT_SIZE = 2_000

# Scoring rules from the README.
CATCH_REWARD = 10        # flag a fraud
FALSE_ALARM_COST = 5     # flag a genuine transaction
MISS_PENALTY = 10        # fail to flag a fraud
# Flag when expected points from flagging beat not flagging:
#   p*CATCH - (1-p)*FALSE_ALARM > -p*MISS   =>   p > FALSE_ALARM / (CATCH + MISS + FALSE_ALARM)
THRESHOLD = FALSE_ALARM_COST / (CATCH_REWARD + MISS_PENALTY + FALSE_ALARM_COST)  # 0.20

NIGHT_HOURS = range(0, 6)

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=0,
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(transactions):
    """Turn raw columns into model inputs, including explicit fraud patterns.

    Deliberately left out: exact hour (fraud is flat across 06:00-23:59, only
    the night band matters), typical monthly spend on its own (it only matters
    through amount_vs_typical) and account type (fraud rate is ~8% for all
    three). Dropping these no-signal columns scored better in both the monthly
    backtest and cross-validation, because the model stops fitting noise.
    """
    t = transactions
    foreign_no_notice = (t["foreign_transaction"] == 1) & (t["travel_notice"] == 0)
    return pd.DataFrame({
        # Channel
        "is_online": t["transaction_type"].eq("CARD_ONLINE"),
        "is_transfer": t["transaction_type"].eq("BANK_TRANSFER"),
        "is_in_store": t["transaction_type"].eq("CARD_IN_STORE"),
        # Time
        "night": t["hour_of_day"].isin(NIGHT_HOURS),
        # Money
        "amount": t["amount_gbp"],
        "amount_vs_typical": t["amount_vs_typical"],
        # Location
        "foreign_no_notice": foreign_no_notice,
        "travel_notice": t["travel_notice"],
        # Login / device
        "unknown_device": t["unknown_device"],
        "failed_logins": t["failed_logins"],
        "takeover": (t["unknown_device"] == 1) & (t["failed_logins"] >= 3),
        "clean_login": (t["unknown_device"] == 0) & (t["failed_logins"] == 0),
        # Payee
        "new_payee_transfer": (t["new_payee"] == 1) & t["transaction_type"].eq("BANK_TRANSFER"),
    }, index=t.index).astype(float)


def fit_model(transactions):
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(build_features(transactions), transactions[TARGET])
    return model


def predict_risk(model, transactions):
    return model.predict_proba(build_features(transactions))[:, 1]


# ---------------------------------------------------------------------------
# Previous models, kept so validation shows the improvement
# ---------------------------------------------------------------------------
PREVIOUS_SIGNALS = [
    "account_type", "typical_monthly_spend_gbp", "transaction_type", "amount_gbp",
    "amount_vs_typical", "hour_of_day", "merchant_country", "foreign_transaction",
    "unknown_device", "failed_logins", "new_payee", "travel_notice",
]
STARTER_FEATURES = [
    "amount_gbp", "amount_vs_typical", "hour_of_day", "foreign_transaction",
    "unknown_device", "failed_logins", "new_payee", "travel_notice",
]


def previous_model_risk(learn, score):
    encode = lambda d: pd.get_dummies(d[PREVIOUS_SIGNALS])
    learn_x = encode(learn)
    model = HistGradientBoostingClassifier(max_iter=170, max_leaf_nodes=15, random_state=0)
    model.fit(learn_x, learn[TARGET])
    return model.predict_proba(encode(score).reindex(columns=learn_x.columns, fill_value=0))[:, 1]


def starter_tree_risk(learn, score):
    feats = lambda d: d[STARTER_FEATURES].assign(
        is_transfer=d["transaction_type"].eq("BANK_TRANSFER"),
        is_online=d["transaction_type"].eq("CARD_ONLINE"),
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(feats(learn), learn[TARGET])
    return tree.predict_proba(feats(score))[:, 1]


# ---------------------------------------------------------------------------
# Scoring and validation
# ---------------------------------------------------------------------------
def score_flags(actual, flags):
    fraud = np.asarray(actual) == 1
    flags = np.asarray(flags, dtype=bool)
    caught = int((fraud & flags).sum())
    false_alarms = int((~fraud & flags).sum())
    missed = int((fraud & ~flags).sum())
    points = CATCH_REWARD * caught - FALSE_ALARM_COST * false_alarms - MISS_PENALTY * missed
    return points, caught, false_alarms, missed


def fmt(result):
    points, caught, false_alarms, missed = result
    return f"{points:>6,} pts ({caught} caught, {false_alarms} false alarms, {missed} missed)"


def validation_splits(train):
    """Monthly backtest (Jun, Jul, Aug) plus the original last-2,000 holdout."""
    ordered = train.sort_values("timestamp", kind="stable")
    month = pd.to_datetime(ordered["timestamp"]).dt.to_period("M")
    for period in month.unique()[2:]:
        yield str(period), ordered[month < period], ordered[month == period]
    yield f"last {HOLDOUT_SIZE:,}", ordered.iloc[:-HOLDOUT_SIZE], ordered.iloc[-HOLDOUT_SIZE:]


def validate_models(train):
    candidates = {
        "Starter tree (30% cutoff)": (starter_tree_risk, 0.30),
        "Previous model (20% cutoff)": (previous_model_risk, 0.20),
        "New model (20% cutoff)": (lambda l, s: predict_risk(fit_model(l), s), THRESHOLD),
    }
    totals = {name: np.zeros(4, dtype=int) for name in candidates}
    for split_name, learn, score in validation_splits(train):
        print(f"\n{split_name}: {len(score):,} transactions, {int(score[TARGET].sum())} fraud")
        for name, (risk_fn, cutoff) in candidates.items():
            risk = risk_fn(learn, score)
            result = score_flags(score[TARGET], risk >= cutoff)
            print(f"  {name:<28} {fmt(result)}")
            if not split_name.startswith("last"):
                totals[name] += result

    print("\nBacktest total (Jun + Jul + Aug):")
    for name, total in totals.items():
        print(f"  {name:<28} {fmt(tuple(total))}")


def cutoff_table(train, repeats=2):
    """Show why 20% is the right cutoff and what 'catch every fraud' would cost.

    Uses 5-fold cross-validation over all 10,000 training rows (every row is
    scored by a model that never saw it), which is far less noisy than three
    months of backtest. Points are averaged over the repeats.
    """
    from sklearn.model_selection import StratifiedKFold

    actual = train[TARGET].to_numpy()
    risks = []
    for seed in range(repeats):
        risk = np.zeros(len(train))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for learn_idx, score_idx in folds.split(train, actual):
            model = fit_model(train.iloc[learn_idx])
            risk[score_idx] = predict_risk(model, train.iloc[score_idx])
        risks.append(risk)

    print(f"\nNew model, effect of the cutoff ({repeats}x 5-fold cross-validation, "
          f"{len(train):,} transactions, {int(actual.sum())} fraud):")
    for cutoff in [0.50, 0.30, THRESHOLD, 0.10, 0.05, 0.01, 0.001]:
        results = np.mean([score_flags(actual, r >= cutoff) for r in risks], axis=0)
        marker = "  <- chosen (break-even from scoring rules)" if cutoff == THRESHOLD else ""
        print(f"  cutoff {cutoff:>6.1%}: {fmt(tuple(int(round(x)) for x in results))}{marker}")


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def _switch_off(transactions, pattern):
    """Copy of the transactions with one risk pattern replaced by a normal value."""
    t = transactions.copy()
    if pattern == "login":
        t["unknown_device"] = 0
        t["failed_logins"] = 0
    elif pattern == "abroad":
        t["merchant_country"] = "GB"
        t["foreign_transaction"] = 0
    elif pattern == "night":
        t["hour_of_day"] = 14
    elif pattern == "payee":
        t["new_payee"] = 0
    elif pattern == "amount":
        t["amount_gbp"] = 40
        t["amount_vs_typical"] = 40 / t["typical_monthly_spend_gbp"]
    return t


def _present(pattern, row):
    """Is this risk pattern actually present in the transaction?"""
    if pattern == "login":
        return row["unknown_device"] == 1 or row["failed_logins"] >= 3
    if pattern == "abroad":
        return row["foreign_transaction"] == 1 and row["travel_notice"] == 0
    if pattern == "night":
        return row["hour_of_day"] in NIGHT_HOURS
    if pattern == "payee":
        return row["new_payee"] == 1 and row["transaction_type"] == "BANK_TRANSFER"
    if pattern == "amount":
        return row["amount_gbp"] >= 650 or row["amount_vs_typical"] >= 0.5
    return False


def _describe(pattern, row):
    if pattern == "login":
        parts = []
        if row["unknown_device"] == 1:
            parts.append("unknown device")
        if row["failed_logins"] > 0:
            n = row["failed_logins"]
            parts.append(f"{n} failed login{'s' if n != 1 else ''}")
        text = " + ".join(parts)
        if row["unknown_device"] == 1 and row["failed_logins"] >= 3:
            return "Account takeover pattern: " + text
        return text[0].upper() + text[1:]
    if pattern == "abroad":
        return f"Abroad ({row['merchant_country']}) with no travel notice"
    if pattern == "night":
        return f"Night-time ({row['hour_of_day']:02d}:00)"
    if pattern == "payee":
        return "Transfer to a new payee"
    if pattern == "amount":
        return f"£{row['amount_gbp']:,.0f} ({row['amount_vs_typical']:.1f}x typical monthly spend)"
    return pattern


def explain_flags(model, flagged, risk):
    """List the risk patterns present in each flagged transaction, most important first.

    Importance = how far the model's score falls when that pattern is switched
    off (e.g. same transaction but at 14:00 instead of 03:00).
    """
    patterns = ["login", "abroad", "night", "payee", "amount"]
    drops = np.column_stack([
        risk - predict_risk(model, _switch_off(flagged, p)) for p in patterns
    ])
    reasons = []
    for row_number, (_, row) in enumerate(flagged.iterrows()):
        order = np.argsort(-drops[row_number], kind="stable")
        details = [_describe(patterns[i], row) for i in order if _present(patterns[i], row)][:3]
        channel = row["transaction_type"].replace("_", " ").lower()
        reasons.append(f"Risk {risk[row_number]:.0%} ({channel}): "
                       + (" | ".join(details) or "unusual combination of signals"))
    return reasons


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
def make_submission(train, test):
    model = fit_model(train)
    risk = predict_risk(model, test)
    flags = risk >= THRESHOLD
    output = test[["transaction_id"]].assign(reasons="", is_flagged=flags.astype(int))
    if flags.any():
        output.loc[flags, "reasons"] = explain_flags(model, test.loc[flags], risk[flags])
    return output


def main():
    parser = argparse.ArgumentParser(description="Validate the fraud model and export predictions.")
    parser.add_argument("--output", type=Path, default=ROOT / "test_predictions.csv")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    train = load_train()
    test = load_test()
    with threadpool_limits(limits=2):
        if not args.skip_validation:
            validate_models(train)
            cutoff_table(train)
        output = make_submission(train, test)
    output.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}: {len(output):,} rows, {output['is_flagged'].sum()} flagged "
          f"(cutoff {THRESHOLD:.0%})")


if __name__ == "__main__":
    main()
