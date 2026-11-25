# Configurable fraud detection

From the repository root, create an environment and install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the model and its focused tests:

```bash
python -m fraud_detection.model --training-rows 8000
python -m unittest fraud_detection.test_model
```

The program loads `train.csv` and `test.csv` using the repository's shared data
loader. If `--training-rows` is omitted, it prompts for the number of training
rows. The held-out practice-set size and scoring options are read from
[`config.json`](./config.json).
With the current 2,000-row practice setting, use 8,000 training rows so the
requested sets fit within the 10,000 rows in `train.csv`.

The practice size must fit alongside the requested training size. Any remaining
training rows are not used. Practice labels are checked only after training;
the model is fitted on the learn set alone.

Edit `config.json` to adjust the rule score threshold, rule conditions, rule
weights (set a weight to `0` to disable it), tree depth, or prediction method.
Supported rule operators are `eq`, `ne`, `gt`, `gte`, `lt`, and `lte`.
`prediction_method` can be `weighted_rules` or `decision_tree`.

The decision tree uses the transaction details, customer and account
categories, transaction type, merchant country, and time-derived values
(month, weekday, hour, minute, weekend, and late-night). `transaction_id` is
kept only to identify output rows, and `is_fraud` is the answer the model must
learn to predict, so neither is used as a model input. The weighted-rule
configuration includes example weights for late-night activity, transaction
types, merchant country, and unusually large amounts; these rules are used
when `prediction_method` is set to `weighted_rules`.

The program writes `practice_predictions.csv`, `test_predictions.csv`, and
`genuine_averages.json` to this folder by default. Genuine averages are
calculated using only genuine rows in the learn set.
Every prediction includes a reason, including transactions that are not
flagged. The command-line summary reports training/learning, practice, and
testing row counts; points out of the practice maximum; caught fraud, false
alarms, missed fraud; and the total flagged and unflagged practice transactions.

The functions in [`model.py`](./model.py) can be imported by a Streamlit UI,
without requiring Streamlit as a dependency:

```python
from fraud_detection.model import (
    load_config,
    load_train,
    make_features,
    score_transactions,
    split_data,
    train_decision_tree,
)

config = load_config()
train = load_train()
learn, practice = split_data(train, 8000, config["practice_rows"])
model = train_decision_tree(learn, config["decision_tree"], config["random_state"])
practice_scores = score_transactions(model, practice, config)
```
