# Model outputs (read by the UI)

Every `*.csv` in this folder shows up in the app's **Model** dropdown. The UI only displays these files. It never trains or changes a model.

**To add or update your model:**
```bash
python predict_fraud.py --output outputs/<model-name>.csv
git add outputs/ && git commit -m "Update <model-name> predictions"
```

**CSV columns**

| Column | Required | Notes |
|---|---|---|
| `transaction_id` | yes | matches `test.csv` |
| `is_flagged` | yes | 1 = fraud, 0 = genuine |
| `reasons` | optional | e.g. `New payee: 1 \| Amount (GBP): 1200` |
| `risk` | optional | fraud probability 0-1, adds risk bars and sorting |
| `risk_score` | optional | same as `risk` but 0-100 (used if there is no `risk` column) |

**Optional `<model-name>.json`** next to the CSV, shown on the Overview page:
```json
{
  "name": "Display name",
  "threshold": 0.2,
  "validation": {"description": "...", "points": 1245, "max_points": 1500,
                 "caught": 140, "false_alarms": 11, "missed": 10},
  "notes": "One or two sentences about the approach."
}
```
