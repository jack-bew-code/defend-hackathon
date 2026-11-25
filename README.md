# Participant pack

## TL;DR

- `train.csv` has 10,000 transactions with the answer (`is_fraud`) included. Learn from it.
- `test.csv` has 2,000 new transactions with no answer. Decide which ones are fraud.
- Submit a file of the transactions you flag.

## Scoring

| Event | Points |
|---|---|
| You flag a transaction that is fraud | **+10** |
| You flag a genuine transaction | **-5** |
| You miss a fraud (you do not flag it) | **-10** |
| You do not flag a genuine transaction | 0 |

Flag a transaction if you think there is about a high chance that it is fraud. About 8% of transactions are fraud and some is deliberately hard to spot, so nobody catches it all. Organisers also judge your thinking, so explain what you found and why.


## Data definitions

All data is synthetic. `train.csv` has an extra `is_fraud` column; `test.csv` does not. Flags are 1 (yes) or 0 (no).

| Column | Meaning |
|---|---|
| `transaction_id` | Unique transaction reference ID |
| `timestamp` | When it happened |
| `customer_id` | Customer reference number |
| `account_type` | STUDENT, CURRENT or PREMIUM |
| `typical_monthly_spend_gbp` | The customer's normal monthly spend |
| `transaction_type` | CARD_IN_STORE, CARD_ONLINE or BANK_TRANSFER |
| `amount_gbp` | Amount in pounds |
| `amount_vs_typical` | `amount_gbp` divided by `typical_monthly_spend_gbp` |
| `hour_of_day` | Hour of the transaction, 0-23 |
| `merchant_country` | Country of the shop or recipient (GB, FR, ES, US) |
| `foreign_transaction` | 1 if the country is not GB |
| `unknown_device` | 1 if the customer had not used this device before |
| `failed_logins` | Failed login attempts just before the transaction |
| `new_payee` | 1 if a transfer went to a recently added recipient |
| `travel_notice` | 1 if the customer told the bank they were travelling to that country |
| `is_fraud` | (`train.csv` only) 1 = fraud, 0 = genuine |


---------------------------------Team based work-------------------------------

## Gradient boosting submission

Install the packages and generate the submission from the repository root.

```sh
python -m pip install -r requirements.txt
python predict_fraud.py
```

The script reports each model's score on the latest 2,000 labeled transactions,
then trains the boosted model on all labeled rows and writes
`test_predictions.csv`. The file contains one row per test transaction with
`transaction_id`, `reasons`, and `is_flagged`.

The review cutoff is 20%. Under the challenge scoring, flagging a fraud earns
20 points relative to missing it, while flagging a genuine transaction costs
5 points. The cutoff is the break-even point when the estimated probabilities
are calibrated. For flagged rows, the script shows observed feature values
whose replacement with a common genuine value lowered the model's score.
