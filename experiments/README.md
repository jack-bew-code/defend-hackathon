# Experiments

Alternative models the team built during the hackathon, kept here so the whole exploration is in one place. The submitted model is [`predict_fraud.py`](../predict_fraud.py) in the repo root.

Every experiment's September predictions are in [`outputs/`](../outputs) and can be explored and compared in the [demo app](https://defend-hackathon-winning-team.streamlit.app).

| Folder | Idea | Own test score* | Sept flags |
|---|---|---|---|
| [`sid_optimised_features/`](sid_optimised_features) | Hand-built fraud patterns (night, account takeover, abroad without travel notice), noisy columns dropped | 1,235 / 1,500 | 170 |
| [`jack_tuned_boosting/`](jack_tuned_boosting) | Tuned gradient boosting + red-flag combinations, 4-fold time-based validation | 1,215 / 1,500 | 171 |
| [`chris_threshold_search/`](chris_threshold_search) | Voting ensemble (boosting, random forest, extra trees) with a searched cutoff | 1,250 / 1,500 † | 156 |
| [`ij_decision_tree/`](ij_decision_tree) | Single decision tree or weighted rules, fully driven by `config.json` | 1,260 / 1,600 ‡ | 171 |

\* Points on held-out training data under the competition scoring (+10 caught, −5 false alarm, −10 missed). The submitted model scores **1,245 / 1,500** on the same test.
† Cutoff chosen on the same holdout it is scored on, so optimistic.
‡ Scored on 2,000 *random* training rows (160 frauds) rather than the most recent 2,000 (150 frauds), so not directly comparable.

Run any of them from the **repository root**, e.g. `python -m experiments.sid_optimised_features.model`.
