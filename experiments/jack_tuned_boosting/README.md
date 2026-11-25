# Tuned boosting + red-flag features

**Source:** `jack/dev` (`predict_fraud.py` + `src/features.py`). Only change: `features.py` is imported from this folder.

Starts from the team's gradient-boosting model and asks two questions with **4-fold time-based validation** (train on earlier months, score the next block, late May – August):

1. **Do smaller, slower-learning trees help?** Yes: learning rate 0.03, 500 trees, 7 leaves, L2 1.0 scored **3,925** vs **3,795** for the original settings, winning in 3 of 4 blocks.
2. **Do engineered features help?** Red-flag combinations (unknown device × failed logins, abroad without travel notice, transfer to new payee, round £100 amounts) didn't change the score, because the model already learns them, but they make the reasons easier to read. Per-customer history (velocity, usual amount/hour) made it slightly *worse* (3,875): fraud in this data is about the transaction, not changes in a customer's behaviour.

Finding worth noting: **72% of fraud are exact £100 multiples vs 9% of genuine transactions.**

```bash
python -m experiments.jack_tuned_boosting.model --output outputs/jack_tuned_boosting.csv
```
