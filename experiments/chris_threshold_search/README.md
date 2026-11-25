# Ensemble with cutoff search

**Source:** `jack/notebook-submission` (`fraud_model.py`), copied unchanged.

Soft-voting ensemble of gradient boosting, random forest and extra trees, with time-of-day and calendar features (day of week, weekend, overnight). Instead of the 20% break-even cutoff, it tries every cutoff from 0.05 to 0.80 on the last 2,000 training transactions and keeps the best (0.62).

**Results:** 1,250 / 1,500 with very few false alarms (2; precision 98.6%) but more missed fraud (12). The cutoff is picked on the same holdout it's scored on, so the true score on new data is likely a little lower.

```bash
python -m experiments.chris_threshold_search.model --output outputs/chris_threshold_search.csv
```
