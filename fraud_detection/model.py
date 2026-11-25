"""Train and score fraud detectors using the hackathon transaction data.

Run from the repository root with ``python -m fraud_detection.model``.
The public functions are also usable from a Streamlit app.
"""
# Standard-library modules handle command-line options, JSON config files,
# comparison operators, and paths that work regardless of the current folder.
import argparse
import json
import operator
from pathlib import Path
from typing import Any

# These libraries load and prepare tables, train the decision tree, and print
# the tree's learned questions in a readable form.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder

# Reuse the repository's data-loading functions so the CSV paths are reliable.
from src.data import load_test, load_train

# Keep related files together in this folder; the defaults can be overridden
# from the command line when another config or output location is needed.
FOLDER = Path(__file__).resolve().parent
DEFAULT_CONFIG = FOLDER / "config.json"
DEFAULT_OUTPUT = FOLDER
LABEL = "is_fraud"

# These are the CSV columns and time-based details that become numeric inputs
# to the decision tree.
NUMERIC_FEATURES = [
    "typical_monthly_spend_gbp",
    "amount_gbp",
    "amount_vs_typical",
    "hour_of_day",
    "foreign_transaction",
    "unknown_device",
    "failed_logins",
    "new_payee",
    "travel_notice",
    "timestamp_month",
    "timestamp_day_of_week",
    "timestamp_hour",
    "timestamp_minute",
    "timestamp_is_weekend",
    "timestamp_is_late_night",
]

# These CSV values are labels rather than measurements, so the model learns a
# separate category for each value instead of treating its text as a number.
CATEGORICAL_FEATURES = [
    "customer_id",
    "account_type",
    "transaction_type",
    "merchant_country",
]

# The target label and unique transaction reference are deliberately excluded:
# using the target would reveal the answer, and the reference is not behavior.
SOURCE_FEATURES = [
    "timestamp",
    "customer_id",
    "account_type",
    "typical_monthly_spend_gbp",
    "transaction_type",
    "amount_gbp",
    "amount_vs_typical",
    "hour_of_day",
    "merchant_country",
    "foreign_transaction",
    "unknown_device",
    "failed_logins",
    "new_payee",
    "travel_notice",
]

# Translate the short names in config.json into Python comparisons:
# for example, "gte" means "greater than or equal to".
RULE_OPERATORS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and check editable settings from a JSON configuration file."""
    # Read the external settings file so thresholds and model options can be
    # changed without editing this Python program.
    with Path(path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    # Stop with a useful message if a setting would make scoring ambiguous.
    if config["prediction_method"] not in {"weighted_rules", "decision_tree"}:
        raise ValueError("prediction_method must be 'weighted_rules' or 'decision_tree'")
    if not isinstance(config["practice_rows"], int) or config["practice_rows"] < 1:
        raise ValueError("practice_rows must be a positive integer")
    if not isinstance(config["random_state"], int):
        raise ValueError("random_state must be an integer")

    tree = config["decision_tree"]
    if not isinstance(tree["max_depth"], int) or tree["max_depth"] < 1:
        raise ValueError("decision_tree.max_depth must be a positive integer")
    threshold = tree["probability_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("decision_tree.probability_threshold must be between 0 and 1")

    rules = config["rules"]
    if not isinstance(rules["threshold"], (int, float)) or rules["threshold"] < 0:
        raise ValueError("rules.threshold must be a non-negative number")
    for rule in rules["items"]:
        if rule["operator"] not in RULE_OPERATORS:
            raise ValueError(f"Unsupported operator in rule {rule['name']!r}")
        if not isinstance(rule["weight"], (int, float)) or rule["weight"] < 0:
            raise ValueError(f"Rule {rule['name']!r} weight must be non-negative")
    # Give the validated settings back to the caller for training and scoring.
    return config


def split_data(
    train: pd.DataFrame,
    training_rows: int,
    practice_rows: int,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible learn and held-out practice sets of exact sizes."""
    # Check that the input has answers and that the requested row counts fit.
    if LABEL not in train.columns:
        raise ValueError(f"Training data must include the {LABEL!r} column")
    if training_rows < 1 or practice_rows < 1:
        raise ValueError("training_rows and practice_rows must both be positive")
    if training_rows + practice_rows > len(train):
        raise ValueError(
            f"Requested {training_rows + practice_rows} rows, but train.csv has "
            f"only {len(train)}"
        )
    if train[LABEL].nunique() != 2 or training_rows < 2 or practice_rows < 2:
        raise ValueError(
            "Stratified training and practice sets each need both fraud classes"
        )

    # Randomly select the requested number of rows. Stratification keeps the
    # fraud/genuine proportions similar in both sets, and the fixed seed makes
    # the same split repeatable for comparisons.
    learn, practice = train_test_split(
        train,
        train_size=training_rows,
        test_size=practice_rows,
        random_state=random_state,
        stratify=train[LABEL],
    )
    # Only `learn` is used to fit the model; `practice` stays aside for checking.
    return learn, practice


def make_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Use every meaningful CSV input and derive useful timestamp features."""
    # Verify all expected columns are present before trying to build features.
    required = SOURCE_FEATURES
    missing = sorted(set(required) - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions are missing feature columns: {', '.join(missing)}")

    # Convert the timestamp text to a date/time value, then break it into parts
    # the model can use, such as month, weekday, and late-night activity.
    timestamp = pd.to_datetime(transactions["timestamp"], errors="raise")
    time_features = pd.DataFrame(
        {
            "timestamp_month": timestamp.dt.month,
            "timestamp_day_of_week": timestamp.dt.day_of_week,
            "timestamp_hour": timestamp.dt.hour,
            "timestamp_minute": timestamp.dt.minute,
            "timestamp_is_weekend": timestamp.dt.day_of_week.ge(5).astype(int),
            "timestamp_is_late_night": timestamp.dt.hour.between(0, 5).astype(int),
        },
        index=transactions.index,
    )

    # Keep every meaningful predictor from the CSV except transaction_id and
    # is_fraud. The former is only a unique reference; the latter is the answer.
    features = transactions[CATEGORICAL_FEATURES + [
        "typical_monthly_spend_gbp",
        "amount_gbp",
        "amount_vs_typical",
        "hour_of_day",
        "foreign_transaction",
        "unknown_device",
        "failed_logins",
        "new_payee",
        "travel_notice",
    ]].copy()
    features = pd.concat([features, time_features], axis=1)
    # Return a consistent column order for model training and later predictions.
    features = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return features


def train_decision_tree(
    learn: pd.DataFrame,
    settings: dict[str, Any],
    random_state: int = 0,
) -> Pipeline:
    """Fit a decision tree using only the supplied learn rows."""
    # Both outcomes must be represented so the model can learn to distinguish
    # genuine transactions from fraud.
    if learn[LABEL].nunique() != 2:
        raise ValueError("The learn set must contain both genuine and fraud transactions")
    # Convert the text categories to numeric 0/1 columns, while passing the
    # already-numeric amounts and flags through unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", "passthrough", NUMERIC_FEATURES),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    # A decision tree learns a set of feature questions. max_depth in config
    # limits how many questions it can chain together.
    tree = DecisionTreeClassifier(
        max_depth=settings["max_depth"], random_state=random_state
    )
    # Bundle encoding and the tree so practice/test rows get exactly the same
    # transformations as the learn rows.
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", tree)]
    )
    # Give the pipeline transaction features and their known 0/1 answers.
    model.fit(make_features(learn), learn[LABEL])
    # Return the fitted preprocessing + model pipeline for scoring.
    return model


def score_rules(transactions: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Return weighted-rule scores and flags for each transaction."""
    # Build the same original and timestamp-derived fields used by the tree.
    features = make_features(transactions)
    # Every transaction starts with zero risk points and no matching reasons.
    scores = pd.Series(0.0, index=transactions.index)
    reasons: list[list[str]] = [[] for _ in range(len(transactions))]

    # Check each configured rule against every transaction.
    for rule in settings["items"]:
        weight = rule["weight"]
        # A zero-weight rule is intentionally disabled and has no effect.
        if weight == 0:
            continue
        feature = rule["feature"]
        if feature not in features:
            raise ValueError(f"Rule {rule['name']!r} uses missing feature {feature!r}")
        matched = RULE_OPERATORS[rule["operator"]](
            features[feature], rule["value"]
        ).fillna(False)
        # Add this rule's weight to matching transactions and remember its name
        # so a person can see why a transaction received its score.
        scores.loc[matched] += weight
        for position, is_match in enumerate(matched.to_numpy()):
            if is_match:
                reasons[position].append(rule["name"])

    # A transaction is flagged when its total points meet the configured
    # threshold; finish a readable explanation for both yes and no decisions.
    flagged = scores >= settings["threshold"]
    threshold = settings["threshold"]
    for position, (score, is_flagged) in enumerate(zip(scores, flagged)):
        matched_rules = reasons[position]
        if matched_rules:
            explanation = f"Matched rules: {', '.join(matched_rules)}"
        else:
            explanation = "No weighted rules matched"
        if is_flagged:
            decision = f"Flagged: score {score:g} meets threshold {threshold:g}"
        else:
            decision = f"Not flagged: score {score:g} is below threshold {threshold:g}"
        reasons[position] = [explanation, decision]

    return pd.DataFrame(
        {
            "transaction_id": transactions["transaction_id"],
            "risk_score": scores,
            "reasons": [" | ".join(items) for items in reasons],
            "is_flagged": flagged.astype(int),
        },
        index=transactions.index,
    )


def score_decision_tree(
    model: Pipeline,
    transactions: pd.DataFrame,
    probability_threshold: float,
) -> pd.DataFrame:
    """Return decision-tree fraud probabilities, explanations, and flags."""
    # Find which probability column belongs to the fraud label (1).
    # The classifier is the final step in the pipeline; inspect its known labels.
    classifier = model.named_steps["classifier"]
    fraud_class = np.flatnonzero(classifier.classes_ == 1)
    if len(fraud_class) != 1:
        raise ValueError("The trained decision tree does not contain a fraud class")
    # Ask the fitted tree for a fraud probability for each transaction.
    probabilities = model.predict_proba(make_features(transactions))[:, fraud_class[0]]
    # Compare each probability with the configured cutoff to make a 1/0 flag.
    flagged = probabilities >= probability_threshold
    # Explain the probability comparison for every row, including those that
    # are not flagged, so no output decision has a blank reason.
    reasons = [
        (
            f"Flagged: estimated fraud probability {probability:.1%} meets or "
            f"exceeds cutoff {probability_threshold:.1%}"
        )
        if is_flagged
        else (
            f"Not flagged: estimated fraud probability {probability:.1%} is "
            f"below cutoff {probability_threshold:.1%}"
        )
        for probability, is_flagged in zip(probabilities, flagged)
    ]
    # Return the transaction ID, percentage risk score, explanation, and flag.
    return pd.DataFrame(
        {
            "transaction_id": transactions["transaction_id"],
            "risk_score": (probabilities * 100).round(1),
            "reasons": reasons,
            "is_flagged": flagged.astype(int),
        },
        index=transactions.index,
    )


def score_transactions(
    model: Pipeline,
    transactions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Score transactions with the method selected in the configuration."""
    # Use either the learned tree or the editable weighted rules, as selected
    # in config.json.
    if config["prediction_method"] == "decision_tree":
        return score_decision_tree(
            model, transactions, config["decision_tree"]["probability_threshold"]
        )
    return score_rules(transactions, config["rules"])


def genuine_averages(learn: pd.DataFrame) -> dict[str, float]:
    """Calculate numeric feature averages from genuine learn rows only."""
    # Keep only known genuine transactions from the learn set. The practice
    # set is excluded so its answers do not influence training-time summaries.
    genuine = learn.loc[learn[LABEL] == 0]
    # Average numeric columns only; remove the answer column itself from the
    # summary, then round the results for a readable JSON file.
    numeric = genuine.select_dtypes(include="number").drop(columns=[LABEL], errors="ignore")
    return numeric.mean().round(3).to_dict()


def evaluate_practice(
    predictions: pd.DataFrame, practice: pd.DataFrame
) -> dict[str, int]:
    """Calculate the hackathon point score and practice-set outcomes."""
    # Compare predicted flags with the practice set's known answers.
    flags = predictions["is_flagged"].astype(bool)
    fraud = practice[LABEL] == 1
    # Count the three outcomes separately: caught fraud, false alarms, and
    # fraud that was missed.
    caught = int((flags & fraud).sum())
    false_alarms = int((flags & ~fraud).sum())
    missed = int((~flags & fraud).sum())
    practice_rounds = len(practice)
    fraud_transactions = int(fraud.sum())
    not_flagged = int((~flags).sum())
    # Use the hackathon scoring rules to calculate the overall practice score.
    return {
        "points": 10 * caught - 5 * false_alarms - 10 * missed,
        "maximum_points": 10 * fraud_transactions,
        "caught": caught,
        "false_alarms": false_alarms,
        "missed": missed,
        "not_flagged": not_flagged,
        "flagged": practice_rounds - not_flagged,
        "fraud_transactions": fraud_transactions,
        "practice_rounds": practice_rounds,
    }


def run(
    training_rows: int,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Load data, train/evaluate the model, and write practice and test outputs."""
    # Step 1: load the same labeled and unlabeled CSV files as notebook Step 1.
    train = load_train()
    test = load_test()
    # Read the editable practice size, tree settings, and scoring options.
    config = load_config(config_path)
    # Step 2: separate the requested learning rows from held-out practice rows.
    learn, practice = split_data(
        train,
        training_rows,
        config["practice_rows"],
        config["random_state"],
    )
    # Step 3: fit the model using only learning rows and their known answers.
    model = train_decision_tree(
        learn,
        config["decision_tree"],
        config["random_state"],
    )
    # Step 4: score practice rows (to check quality) and test rows (to submit).
    practice_scores = score_transactions(model, practice, config)
    test_scores = score_transactions(model, test, config)
    # Practice answers are compared only after the model has made predictions.
    results = evaluate_practice(practice_scores, practice)

    # Step 5: create the output folder if needed and save prediction tables.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    practice_output = practice_scores.copy()
    practice_output[LABEL] = practice[LABEL]
    practice_output.to_csv(output_path / "practice_predictions.csv", index=False)
    test_scores.to_csv(output_path / "test_predictions.csv", index=False)
    # Save genuine-only averages as a separate, human-readable JSON file.
    with (output_path / "genuine_averages.json").open("w", encoding="utf-8") as output_file:
        json.dump(genuine_averages(learn), output_file, indent=2, allow_nan=False)

    # Return the useful results as well as the model, allowing a Streamlit app
    # to display metrics or reuse the fitted model without reading output files.
    return {
        "training_rows": len(learn),
        "learning_rows": len(learn),
        "learn_rows": len(learn),
        "practice_rows": len(practice),
        "test_rows": len(test),
        "flagged_test_rows": int(test_scores["is_flagged"].sum()),
        "practice_results": results,
        "model": model,
    }


def main() -> None:
    # Set up options users can provide when starting this program.
    parser = argparse.ArgumentParser(description="Train and score transaction fraud data.")
    parser.add_argument(
        "--training-rows",
        type=int,
        help="number of train.csv rows to use for learning (prompted if omitted)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Ask for the learn-set size if it was not supplied on the command line.
    training_rows = args.training_rows
    if training_rows is None:
        training_rows = int(input("How many train.csv rows should be used for training? "))

    # Run the complete workflow, then show row counts and practice performance.
    result = run(training_rows, args.config, args.output_dir)
    print(
        f"Rows: {result['training_rows']:,} training/learning, "
        f"{result['practice_rows']:,} practice, "
        f"{result['test_rows']:,} testing"
    )
    practice = result["practice_results"]
    print(
        f"Practice rounds: {practice['practice_rounds']:,} transactions"
    )
    print(
        f"Practice points: {practice['points']} / "
        f"{practice['maximum_points']} maximum possible"
    )
    print(
        f"Correctly caught fraud: {practice['caught']}; "
        f"false alarms: {practice['false_alarms']}; "
        f"missed fraud: {practice['missed']}"
    )
    print(
        f"Practice transactions flagged: {practice['flagged']}; "
        f"not flagged: {practice['not_flagged']}"
    )
    # Test answers are intentionally unknown; report only how many were flagged.
    print(f"Flagged {result['flagged_test_rows']:,} test transactions.")
    # Show the learned questions so people can explain the decision tree.
    print(
        "Decision tree:\n"
        f"{export_text(result['model'].named_steps['classifier'], feature_names=list(result['model'].named_steps['preprocessor'].get_feature_names_out()))}"
    )
    # Tell the user where the prediction files and averages were written.
    print(f"Saved results and genuine averages in {args.output_dir}")


if __name__ == "__main__":
    main()
