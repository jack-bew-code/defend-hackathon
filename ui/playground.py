"""Model engine for the "Try settings" tab.

A copy of the fraud/boosted-submission model (HistGradientBoosting on the
transaction signals) that can be re-run with different settings. The model
branches are not touched. All tunable values live in PARAMS / RECOMMENDED below,
so a team config file can replace them later.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data import load_test, load_train

TARGET = "is_fraud"
CATCH, FALSE_ALARM, MISS = 10, -5, -10  # competition rules, not tunable

FEATURE_GROUPS = {
    "amount": ("Amount", ["amount_gbp", "amount_vs_typical"]),
    "time": ("Time of day", ["hour_of_day"]),
    "device": ("Device & logins", ["unknown_device", "failed_logins"]),
    "payee_travel": ("Payee, foreign & travel notice", ["new_payee", "foreign_transaction", "travel_notice"]),
    "type_country": ("Transaction type & country", ["transaction_type", "merchant_country"]),
    "account": ("Account type & usual spend", ["account_type", "typical_monthly_spend_gbp"]),
}
ALL_GROUPS = tuple(FEATURE_GROUPS)


@dataclass(frozen=True)
class Settings:
    cutoff: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    holdout_size: int
    features: tuple = ALL_GROUPS


# Slider ranges and help text. (label, min, max, step, help)
PARAMS = {
    "cutoff": ("Fraud cutoff", 0.05, 0.60, 0.01,
               "Flag a transaction when the model's fraud probability is at least this. "
               "0.20 is the break-even point for the +10 / −5 / −10 scoring."),
    "learning_rate": ("Learning rate", 0.01, 0.30, 0.01,
                      "How big a step each new tree takes. Smaller = slower, steadier learning (needs more trees)."),
    "max_iter": ("Number of trees", 50, 1000, 10,
                 "How many trees are added in turn, each fixing the last ones' mistakes."),
    "max_leaf_nodes": ("Leaves per tree", 3, 31, 1,
                       "How detailed each tree can be. Fewer leaves = simpler rules, less overfitting."),
    "min_samples_leaf": ("Min transactions per leaf", 5, 100, 5,
                         "A rule must cover at least this many training transactions. Higher = ignores rare flukes."),
    "l2_regularization": ("L2 regularisation", 0.0, 5.0, 0.1,
                          "Penalty that keeps trees from leaning too hard on any one pattern."),
    "holdout_size": ("Test on most recent N transactions", 1000, 3000, 250,
                     "How many of the latest training transactions are hidden and used to score the settings."),
}

# The fraud/boosted-submission settings, for reference
FRAUD_BRANCH = Settings(cutoff=0.20, learning_rate=0.1, max_iter=170, max_leaf_nodes=15,
                        min_samples_leaf=20, l2_regularization=0.0, holdout_size=2000)
# Best of a 162-combination grid search (learning rate x trees x leaves x min-leaf x L2),
# scored at the 0.20 cutoff on 4 time-based folds of 1,500 (late May - Aug, max 4,650 pts):
#   recommended 3,970 pts [1130, 975, 995, 870]  vs  fraud branch 3,795 [1095, 965, 885, 850]
# It beats the fraud-branch settings in every fold. The top ~10 combinations are within
# ~25 pts of each other, so treat small differences as noise.
RECOMMENDED = Settings(cutoff=0.20, learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                       min_samples_leaf=40, l2_regularization=0.0, holdout_size=2000)


def columns(features: tuple) -> list[str]:
    return [c for g in features for c in FEATURE_GROUPS[g][1]]


def encode(df: pd.DataFrame, cols: list[str], like: pd.Index | None = None) -> pd.DataFrame:
    out = pd.get_dummies(df[cols])
    return out if like is None else out.reindex(columns=like, fill_value=0)


def fit(df: pd.DataFrame, s: Settings):
    X = encode(df, columns(s.features))
    model = HistGradientBoostingClassifier(
        learning_rate=s.learning_rate, max_iter=s.max_iter, max_leaf_nodes=s.max_leaf_nodes,
        min_samples_leaf=s.min_samples_leaf, l2_regularization=s.l2_regularization, random_state=0,
    ).fit(X, df[TARGET])
    return model, X.columns


def predict(model, like, df: pd.DataFrame, s: Settings) -> np.ndarray:
    return model.predict_proba(encode(df, columns(s.features), like))[:, 1]


def score(actual: np.ndarray, flags: np.ndarray) -> dict:
    fraud = np.asarray(actual) == 1
    caught = int((fraud & flags).sum())
    false_alarms = int((~fraud & flags).sum())
    missed = int((fraud & ~flags).sum())
    return {"points": CATCH * caught + FALSE_ALARM * false_alarms + MISS * missed,
            "max_points": CATCH * int(fraud.sum()),
            "caught": caught, "false_alarms": false_alarms, "missed": missed}


def load_data():
    train, test = load_train(), load_test()
    return train.sort_values("timestamp", kind="stable").reset_index(drop=True), test


def run(s: Settings, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Score the settings on the most recent holdout, then predict September."""
    learn, hold = train.iloc[:-s.holdout_size], train.iloc[-s.holdout_size:]
    model, like = fit(learn, s)
    risk = predict(model, like, hold, s)
    result = score(hold[TARGET], risk >= s.cutoff)

    full_model, full_like = fit(train, s)
    test_risk = predict(full_model, full_like, test, s)
    predictions = pd.DataFrame({"transaction_id": test["transaction_id"],
                                "risk": test_risk.round(4),
                                "is_flagged": (test_risk >= s.cutoff).astype(int)})
    return {
        **result,
        "september_flags": int(predictions.is_flagged.sum()),
        "holdout_risk": risk, "holdout_actual": hold[TARGET].to_numpy(),
        "holdout_period": f"{hold.timestamp.iloc[0][:10]} to {hold.timestamp.iloc[-1][:10]}",
        "predictions": predictions,
    }


def cutoff_curve(risk: np.ndarray, actual: np.ndarray, cutoffs=None) -> pd.DataFrame:
    cutoffs = np.round(np.arange(0.05, 0.61, 0.01), 2) if cutoffs is None else cutoffs
    return pd.DataFrame([{"cutoff": c, **score(actual, risk >= c)} for c in cutoffs])


def settings_dict(s: Settings) -> dict:
    return asdict(s)


def replace_holdout(s: Settings, holdout_size: int) -> Settings:
    """Recommended settings re-scored on the same holdout size, for a fair comparison."""
    from dataclasses import replace
    return replace(s, holdout_size=holdout_size)
