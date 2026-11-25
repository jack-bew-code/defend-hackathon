"""Read model prediction files from outputs/ and join them to test.csv for display.

Display only: nothing here trains, scores or changes a model.
"""
import json
import re

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


def model_label(name: str) -> str:
    """Display name from outputs/<name>.json, else the file name."""
    path = OUTPUTS / f"{name}.json"
    try:
        return json.loads(path.read_text()).get("name", name) if path.exists() else name
    except (json.JSONDecodeError, OSError):
        return name


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

    if "risk" not in preds.columns and "risk_score" in preds.columns:  # 0-100 scale (IJ's model)
        preds["risk"] = pd.to_numeric(preds["risk_score"], errors="coerce") / 100
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


TYPE_LABELS = {"CARD_IN_STORE": "Card in store", "CARD_ONLINE": "Card online", "BANK_TRANSFER": "Bank transfer"}
ACCOUNT_LABELS = {"STUDENT": "Student", "CURRENT": "Current", "PREMIUM": "Premium"}
COUNTRY_LABELS = {"GB": "UK", "FR": "France", "ES": "Spain", "US": "USA"}
# Labels that only restate the score or channel: shown as chips, left out of "most common reasons"
SCORE_LABELS = {"Risk", "Fraud probability", "Channel"}

_IJ_STYLE = re.compile(r"^(?:Not flagged|Flagged): estimated fraud probability ([\d.]+)%")
_SID_STYLE = re.compile(r"^Risk (\d+%) \(([^)]*)\):\s*(.*)$")


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


def _tidy(label: str, value: str) -> tuple[str, str]:
    """Codes -> words, 1/0 -> yes/no, amounts in £ (for 'Label: value' reasons)."""
    value = TYPE_LABELS.get(value, ACCOUNT_LABELS.get(value, COUNTRY_LABELS.get(value, value)))
    if label.lower() in {"new payee", "unknown device", "foreign transaction", "travel notice"}:
        value = {"1": "yes", "0": "no"}.get(value, value)
    elif label.startswith("Amount (GBP)"):
        label, value = "Amount", f"£{float(value):,.0f}" if value.replace(".", "", 1).isdigit() else value
    elif label == "Transaction hour" and value.isdigit():
        value = f"{int(value):02d}:00"
    return label, value


def pretty_reasons(text: str) -> list[tuple[str, str]]:
    """Readable (label, value) chips for any model's reason text. value '' = plain chip.

    Handles three styles:
      'Transaction type: BANK_TRANSFER | New payee: 1'                    (boosted)
      'Risk 87% (bank transfer): Transfer to a new payee | Night-time ...' (Sid)
      'Flagged: estimated fraud probability 100.0% meets or exceeds ...'    (IJ)
    """
    text = str(text).strip()
    if not text:
        return []
    if m := _IJ_STYLE.match(text):
        return [("Fraud probability", f"{float(m.group(1)):.0f}%")]
    if m := _SID_STYLE.match(text):
        out = [("Risk", m.group(1)), ("Channel", m.group(2).capitalize())]
        out += [(chunk.strip(), "") for chunk in m.group(3).split("|") if chunk.strip()]
        return out
    return [_tidy(label, value) for label, value in split_reasons(text)]


def reason_category(label: str, value: str) -> str | None:
    """Group similar reasons for the 'most common reasons' chart (None = skip)."""
    if label in SCORE_LABELS:
        return None
    if value:  # 'Label: value' style -> the label is the reason
        return label
    low = label.lower()
    for key, name in [("takeover", "Account takeover pattern"), ("night", "Night-time"),
                      ("new payee", "Transfer to a new payee"), ("travel notice", "Abroad, no travel notice"),
                      ("unknown device", "Unknown device / failed logins"),
                      ("failed login", "Unknown device / failed logins"), ("£", "Large amount")]:
        if key in low:
            return name
    return label


def reason_counts(df: pd.DataFrame) -> pd.Series:
    if "reasons" not in df:
        return pd.Series(dtype=int)
    cats = []
    for text in df.loc[df.is_flagged.eq(1), "reasons"]:
        seen = {reason_category(label, value) for label, value in pretty_reasons(text)}
        cats += [c for c in seen if c]
    return pd.Series(cats, dtype=str).value_counts()
