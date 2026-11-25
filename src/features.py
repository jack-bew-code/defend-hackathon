"""Shared feature engineering. Add functions here rather than copying code between notebooks.

Customer history features only look *backwards* in time: each transaction is
compared with that customer's earlier transactions, never later ones, and never
with fraud labels. So it is safe to run on train and test together, which lets
September (test) rows use each customer's April-August history.

    from src.features import add_features
    both = add_features(pd.concat([train, test]))
"""
import numpy as np
import pandas as pd

HISTORY_FEATURES = {
    "cust_prior_txns": "Customer's previous transactions",
    "hours_since_prev": "Hours since customer's last transaction",
    "txns_last_1h": "Customer transactions in previous hour",
    "txns_last_24h": "Customer transactions in previous 24h",
    "amount_vs_cust_avg": "Amount / customer's usual amount",
    "amount_cust_zscore": "Amount vs customer's usual (std devs)",
    "hours_from_usual_time": "Hours from customer's usual time",
    "first_time_country": "First transaction in this country",
    "first_time_type": "First transaction of this type",
    "foreign_no_notice": "Foreign with no travel notice",
    "unknown_device_failed_logins": "Unknown device x failed logins",
    "transfer_to_new_payee": "Bank transfer to new payee",
    "is_night": "Night-time (00:00-05:59)",
}


def _count_recent(times: pd.Series, window: np.timedelta64) -> np.ndarray:
    """For time-sorted transactions, count earlier ones within `window`."""
    t = times.to_numpy()
    return np.arange(len(t)) - np.searchsorted(t, t - window, side="left")


def _prior_stat(group, column, stat):
    """Expanding statistic over each customer's *previous* rows only."""
    return group[column].transform(lambda s: getattr(s.shift().expanding(), stat)())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_ts"] = pd.to_datetime(out["timestamp"])
    out["_order"] = np.arange(len(out))
    out = out.sort_values(["customer_id", "_ts", "_order"], kind="stable")
    g = out.groupby("customer_id", sort=False)

    # How much history we have, and how fast the customer is transacting
    out["cust_prior_txns"] = g.cumcount()
    out["hours_since_prev"] = g["_ts"].diff().dt.total_seconds() / 3600
    out["txns_last_1h"] = g["_ts"].transform(lambda s: _count_recent(s, np.timedelta64(1, "h")))
    out["txns_last_24h"] = g["_ts"].transform(lambda s: _count_recent(s, np.timedelta64(24, "h")))

    # Is this amount unusual *for this customer*?
    prior_mean = _prior_stat(g, "amount_gbp", "mean")
    prior_std = _prior_stat(g, "amount_gbp", "std")
    out["amount_vs_cust_avg"] = out["amount_gbp"] / prior_mean
    out["amount_cust_zscore"] = (out["amount_gbp"] - prior_mean) / prior_std.replace(0, np.nan)

    # Is this an unusual time of day for this customer? (circular: 23h and 1h are 2h apart)
    usual_hour = _prior_stat(g, "hour_of_day", "median")
    gap = (out["hour_of_day"] - usual_hour).abs()
    out["hours_from_usual_time"] = np.minimum(gap, 24 - gap)

    # Something this customer has never done before
    out["first_time_country"] = out.groupby(["customer_id", "merchant_country"]).cumcount().eq(0)
    out["first_time_type"] = out.groupby(["customer_id", "transaction_type"]).cumcount().eq(0)
    # The very first transaction is "first" for everything, which says nothing
    no_history = out["cust_prior_txns"].eq(0)
    out.loc[no_history, ["first_time_country", "first_time_type"]] = False

    # Combinations of red flags that are stronger together than apart
    out["foreign_no_notice"] = out["foreign_transaction"].eq(1) & out["travel_notice"].eq(0)
    out["unknown_device_failed_logins"] = out["unknown_device"] * out["failed_logins"]
    out["transfer_to_new_payee"] = out["transaction_type"].eq("BANK_TRANSFER") & out["new_payee"].eq(1)
    out["is_night"] = out["hour_of_day"].between(0, 5)

    bools = ["first_time_country", "first_time_type", "foreign_no_notice",
             "transfer_to_new_payee", "is_night"]
    out[bools] = out[bools].astype(int)

    out = out.sort_values("_order").drop(columns=["_ts", "_order"])
    out.index = df.index
    return out
