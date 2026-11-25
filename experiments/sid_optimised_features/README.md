# Optimised fraud-pattern features

**Source:** `sid/branch` (`Test_pyfile optimised.py`), copied unchanged.

Gradient boosting (300 trees, learning rate 0.05, 15 leaves, L2 1.0) on hand-built fraud patterns instead of raw columns:
night-time (00:00–05:59), account takeover (unknown device + 3 or more failed logins), clean login, abroad without a travel notice, transfer to a new payee, amount and amount vs usual spend.
Columns with no fraud signal (exact hour, account type, usual spend on its own, merchant country) are left out.
Flags transactions at a 20% fraud probability, the break-even point for the scoring.

**Results:** 1,235 / 1,500 on the last 2,000 training transactions; Jun–Aug monthly backtest 3,905 points (441 caught, 61 false alarms, 20 missed). Reasons are written in plain English, e.g. *"Risk 100% (bank transfer): £1,200 (2.4x typical monthly spend) | Transfer to a new payee | Night-time (01:00)"*.

```bash
python -m experiments.sid_optimised_features.model --output outputs/sid_optimised.csv
```
