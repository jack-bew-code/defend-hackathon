# unittest runs these focused checks without adding another test dependency.
import unittest

import pandas as pd

# Test the public scoring and splitting functions used by the workflow.
from experiments.ij_decision_tree.model import (
    make_features,
    score_decision_tree,
    score_rules,
    split_data,
    train_decision_tree,
)


class FraudModelTests(unittest.TestCase):
    def test_features_use_source_columns_without_target_or_transaction_id(self) -> None:
        # Include every train.csv predictor and the label, as in the real data.
        transactions = pd.DataFrame(
            {
                "transaction_id": ["tx-1"],
                "timestamp": ["2026-04-04 02:15:00"],
                "customer_id": ["C001"],
                "account_type": ["STUDENT"],
                "typical_monthly_spend_gbp": [500],
                "transaction_type": ["BANK_TRANSFER"],
                "amount_gbp": [400],
                "amount_vs_typical": [0.8],
                "hour_of_day": [2],
                "merchant_country": ["US"],
                "foreign_transaction": [1],
                "unknown_device": [1],
                "failed_logins": [3],
                "new_payee": [1],
                "travel_notice": [0],
                "is_fraud": [1],
            }
        )

        # Feature preparation should keep meaningful inputs, derive time
        # signals, and never pass the answer or transaction reference to the tree.
        features = make_features(transactions)

        self.assertIn("customer_id", features)
        self.assertIn("account_type", features)
        self.assertIn("merchant_country", features)
        self.assertEqual(features.loc[0, "timestamp_is_late_night"], 1)
        self.assertEqual(features.loc[0, "timestamp_is_weekend"], 1)
        self.assertNotIn("is_fraud", features.columns)
        self.assertNotIn("transaction_id", features.columns)

        # Fit a tiny tree with both labels and check that even unflagged output
        # rows explain their probability-versus-cutoff decision.
        genuine = transactions.copy()
        genuine["transaction_id"] = "tx-genuine"
        genuine["is_fraud"] = 0
        learn = pd.concat([genuine, transactions], ignore_index=True)
        model = train_decision_tree(learn, {"max_depth": 2})
        predictions = score_decision_tree(model, learn, probability_threshold=0.5)
        self.assertTrue(predictions["reasons"].str.strip().ne("").all())

    def test_rules_apply_comparators_weights_threshold_and_zero_weight(self) -> None:
        # Create a few transactions covering both the >= and == conditions.
        transactions = pd.DataFrame(
            {
                "transaction_id": ["a", "b", "c"],
                "timestamp": [
                    "2026-04-04 02:15:00",
                    "2026-04-04 14:15:00",
                    "2026-04-05 10:15:00",
                ],
                "customer_id": ["C001", "C002", "C003"],
                "account_type": ["STUDENT", "CURRENT", "PREMIUM"],
                "typical_monthly_spend_gbp": [500, 1000, 2000],
                "transaction_type": ["BANK_TRANSFER", "CARD_ONLINE", "CARD_IN_STORE"],
                "amount_gbp": [100, 200, 300],
                "amount_vs_typical": [0.2, 0.2, 0.15],
                "hour_of_day": [2, 14, 10],
                "merchant_country": ["GB", "US", "GB"],
                "foreign_transaction": [0, 1, 0],
                "unknown_device": [1, 0, 0],
                "failed_logins": [3, 4, 5],
                "new_payee": [0, 1, 0],
                "travel_notice": [0, 0, 0],
            }
        )
        # Give the first rule two points, the exact-match rule one point, and
        # disable the device rule with a zero weight.
        settings = {
            "threshold": 3,
            "items": [
                {
                    "name": "Three or more failed logins",
                    "feature": "failed_logins",
                    "operator": "gte",
                    "value": 3,
                    "weight": 2,
                },
                {
                    "name": "Exactly four logins",
                    "feature": "failed_logins",
                    "operator": "eq",
                    "value": 4,
                    "weight": 1,
                },
                {
                    "name": "Disabled rule",
                    "feature": "unknown_device",
                    "operator": "eq",
                    "value": 1,
                    "weight": 0,
                },
            ],
        }

        # Score the rows, then verify the points, cutoff, explanations, and
        # disabled-rule behavior.
        result = score_rules(transactions, settings)

        self.assertEqual(result["risk_score"].tolist(), [2, 3, 2])
        self.assertEqual(result["is_flagged"].tolist(), [0, 1, 0])
        self.assertIn("Three or more failed logins", result.loc[0, "reasons"])
        self.assertNotIn("Disabled rule", " ".join(result["reasons"]))
        self.assertTrue(result["reasons"].str.strip().ne("").all())

        # Even when no rule matches, every transaction still explains why it
        # was not flagged.
        no_match = score_rules(transactions, {"threshold": 1, "items": []})
        self.assertTrue(no_match["reasons"].str.contains("No weighted rules matched").all())
        self.assertTrue(no_match["reasons"].str.strip().ne("").all())

    def test_split_uses_configured_row_counts(self) -> None:
        # Build a small labeled dataset with equal numbers of both classes.
        train = pd.DataFrame(
            {
                "transaction_id": [f"tx-{i}" for i in range(20)],
                "is_fraud": [0, 1] * 10,
            }
        )

        # Ask for fixed learn and practice sizes from this toy dataset.
        learn, practice = split_data(train, training_rows=12, practice_rows=5)

        # Confirm both sizes are exact and no transaction appears in both sets.
        self.assertEqual(len(learn), 12)
        self.assertEqual(len(practice), 5)
        self.assertTrue(set(learn["transaction_id"]).isdisjoint(practice["transaction_id"]))


if __name__ == "__main__":
    unittest.main()
