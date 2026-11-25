"""Read model prediction files from outputs/ and join them to test.csv for display.

Display only: nothing here trains, scores or changes a model.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import ROOT, load_test

OUTPUTS = ROOT / "outputs"
REQUIRED = {"transaction_id", "is_flagged"}
OPTIONAL = ["reasons", "risk"]

AMOUNT_BANDS = [0, 50, 200, 500, 1000, 2000, float("inf")]
AMOUNT_LABELS = ["< £50", "£50-200", "£200-500", "£500-1k", "£1k-2k", "£2k+"]


def list_models() -> list[str]:
    return sorted(p.stem for p in OUTPUTS.glob("*.csv"))


def _mtime(name: str) -> float:
    """Part of the cache key, so re-committed outputs show up without a restart."""
    files = [OUTPUTS / f"{name}.csv", OUTPUTS / f"{name}.json"]
    return max((f.stat().st_mtime for f in files if f.exists()), default=0)


@st.cache_data
def transactions() -> pd.DataFrame:
    df = load_test()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount_band"] = pd.cut(df["amount_gbp"], AMOUNT_BANDS, labels=AMOUNT_LABELS, right=False)
    df["foreign_no_notice"] = (df["foreign_transaction"].eq(1) & df["travel_notice"].eq(0)).astype(int)
    df["any_failed_logins"] = df["failed_logins"].gt(0).astype(int)
    return df


@st.cache_data
def _load(name: str, _mtime_key: float):
    path = OUTPUTS / f"{name}.csv"
    preds = pd.read_csv(path)
    missing = REQUIRED - set(preds.columns)
    if missing:
        return None, {}, f"`{path.name}` is missing column(s): {', '.join(sorted(missing))}"

    cols = ["transaction_id", "is_flagged"] + [c for c in OPTIONAL if c in preds.columns]
    preds = preds[cols].copy()
    preds["is_flagged"] = preds["is_flagged"].fillna(0).astype(int)
    if "reasons" in preds:
        preds["reasons"] = preds["reasons"].fillna("").astype(str)

    df = transactions().merge(preds, on="transaction_id", how="inner")
    if df.empty:
        return None, {}, f"No `transaction_id` in `{path.name}` matches `test.csv`."

    info_path = OUTPUTS / f"{name}.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            info = {"notes": f"(Could not read {info_path.name})"}
    return df, info, None


def load_model(name: str):
    """Returns (dataframe, info dict, error message or None)."""
    return _load(name, _mtime(name))


def split_reasons(text: str) -> list[tuple[str, str]]:
    """'New payee: 1 | Amount (GBP): 1200' -> [('New payee', '1'), ('Amount (GBP)', '1200')]."""
    parts = []
    for chunk in str(text).split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, _, value = chunk.partition(":")
        parts.append((label.strip(), value.strip()))
    return parts


def reason_counts(df: pd.DataFrame) -> pd.Series:
    if "reasons" not in df:
        return pd.Series(dtype=int)
    labels = [label for text in df.loc[df.is_flagged.eq(1), "reasons"] for label, _ in split_reasons(text)]
    return pd.Series(labels, dtype=str).value_counts()
