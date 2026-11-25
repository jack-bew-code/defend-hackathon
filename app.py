"""Fraud detection demo for the judges.

Display only: shows the prediction files the model team saves in outputs/.
Run locally with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from ui import charts
from ui.load import list_models, load_model, reason_counts, split_reasons
from ui.style import FRAUD, GENUINE, MUTED, cards, chips, inject, key_values

st.set_page_config(page_title="Fraud Radar", page_icon="🛡️", layout="wide")
inject()

TYPE_LABELS = {"CARD_IN_STORE": "Card in store", "CARD_ONLINE": "Card online", "BANK_TRANSFER": "Bank transfer"}
ACCOUNT_LABELS = {"STUDENT": "Student", "CURRENT": "Current", "PREMIUM": "Premium"}
COUNTRY_LABELS = {"GB": "UK", "FR": "France", "ES": "Spain", "US": "USA"}


def pretty_reasons(text: str) -> list[tuple[str, str]]:
    """Make the model's reason text readable: codes -> words, 1/0 -> yes/no."""
    out = []
    for label, value in split_reasons(text):
        value = TYPE_LABELS.get(value, ACCOUNT_LABELS.get(value, COUNTRY_LABELS.get(value, value)))
        if label.lower() in {"new payee", "unknown device", "foreign transaction", "travel notice"}:
            value = {"1": "yes", "0": "no"}.get(value, value)
        elif label.startswith("Amount (GBP)"):
            label, value = "Amount", f"£{float(value):,.0f}" if value.replace(".", "", 1).isdigit() else value
        elif label == "Transaction hour" and value.isdigit():
            value = f"{int(value):02d}:00"
        out.append((label, value))
    return out


SIGNALS = {
    "unknown_device": "Unknown device",
    "any_failed_logins": "Failed logins before payment",
    "new_payee": "Transfer to a new payee",
    "foreign_no_notice": "Abroad with no travel notice",
    "foreign_transaction": "Foreign merchant",
}

# ---------- sidebar: pick a model ----------
models = list_models()
with st.sidebar:
    st.markdown("## 🛡️ Fraud Radar")
    st.markdown("Reviewing **2,000 September transactions** and flagging the ones our model thinks are fraud.")
    if not models:
        st.error("No prediction files in `outputs/` yet. See `outputs/README.md`.")
        st.stop()
    model = st.selectbox("Model", models, help="Every CSV in outputs/ appears here.")

df, info, error = load_model(model)
if error:
    st.error(error)
    st.stop()

has_risk, has_reasons = "risk" in df, "reasons" in df
flagged = df[df.is_flagged.eq(1)]
if has_risk:
    flagged = flagged.sort_values("risk", ascending=False)

with st.sidebar:
    st.markdown(f"**{info.get('name', model)}**")
    if info.get("notes"):
        st.markdown(info["notes"])
    st.caption(f"Source: outputs/{model}.csv")

# ---------- header ----------
st.markdown(
    f'<div class="hero"><h1>Catching fraud before it costs customers</h1>'
    f'<p>{info.get("name", model)} · {len(df):,} transactions · '
    f'{df.timestamp.min():%d %b} to {df.timestamp.max():%d %b %Y}</p></div>',
    unsafe_allow_html=True,
)

tab_overview, tab_flags, tab_patterns, tab_compare = st.tabs(
    ["Overview", "Flagged transactions", "What the model catches", "Compare models"]
)

# ---------- 1. overview ----------
with tab_overview:
    cards([
        ("Transactions reviewed", f"{len(df):,}", "September 2026", MUTED),
        ("Flagged as fraud", f"{len(flagged):,}", f"{len(flagged) / len(df):.1%} of all transactions", FRAUD),
        ("Value flagged", f"£{flagged.amount_gbp.sum():,.0f}",
         f"avg £{flagged.amount_gbp.mean():,.0f} per flag" if len(flagged) else "", FRAUD),
        ("Left alone", f"{len(df) - len(flagged):,}", "treated as genuine", GENUINE),
    ])

    val = info.get("validation")
    if val:
        st.write("")
        st.markdown("#### How the model scored on data it hadn't seen")
        if val.get("description"):
            st.caption(f"Reported by the model team · {val['description']}")
        pct = f"{val['points'] / val['max_points']:.0%} of the maximum" if val.get("max_points") else ""
        cards([
            ("Points", f"{val.get('points', 0):,}" + (f" / {val['max_points']:,}" if val.get("max_points") else ""), pct, FRAUD),
            ("Frauds caught", f"{val.get('caught', 0):,}", "+10 points each", GENUINE),
            ("False alarms", f"{val.get('false_alarms', 0):,}", "−5 points each", MUTED),
            ("Frauds missed", f"{val.get('missed', 0):,}", "−10 points each", MUTED),
        ])

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(charts.donut(len(flagged), len(df)), width="stretch")
    with right:
        threshold = info.get("threshold")
        st.markdown("#### How the competition is scored")
        st.markdown(
            '<table class="rules"><tr><th>Event</th><th>Points</th></tr>'
            '<tr><td>Flag a transaction that is fraud</td><td><b>+10</b></td></tr>'
            '<tr><td>Flag a genuine transaction</td><td><b>−5</b></td></tr>'
            '<tr><td>Miss a fraud</td><td><b>−10</b></td></tr>'
            '<tr><td>Leave a genuine transaction alone</td><td>0</td></tr></table>',
            unsafe_allow_html=True,
        )
        if threshold is not None:
            st.markdown(f"The model flags a transaction when it estimates at least a "
                        f"**{threshold:.0%} chance** it is fraud.")
        st.caption("About 8% of transactions are fraud, so a well-tuned model should flag roughly 160 of 2,000.")

# ---------- 2. flagged transactions ----------
with tab_flags:
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect("Transaction type", sorted(flagged.transaction_type.unique()),
                           format_func=lambda t: TYPE_LABELS.get(t, t))
    countries = f2.multiselect("Merchant country", sorted(flagged.merchant_country.unique()))
    accounts = f3.multiselect("Account type", sorted(flagged.account_type.unique()),
                              format_func=lambda a: ACCOUNT_LABELS.get(a, a))
    view = flagged
    if types:
        view = view[view.transaction_type.isin(types)]
    if countries:
        view = view[view.merchant_country.isin(countries)]
    if accounts:
        view = view[view.account_type.isin(accounts)]

    st.caption(f"Showing {len(view):,} of {len(flagged):,} flagged transactions"
               + (" · highest risk first" if has_risk else "") + " · click a row for details")

    table = pd.DataFrame({
        "ID": view.transaction_id,
        "When": view.timestamp,
        "Amount": view.amount_gbp,
        "Type": view.transaction_type.map(TYPE_LABELS),
        "Country": view.merchant_country,
        "Unknown device": view.unknown_device.eq(1),
        "Failed logins": view.failed_logins,
    })
    config = {
        "When": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
        "Amount": st.column_config.NumberColumn(format="£%.2f"),
    }
    if has_risk:
        table.insert(1, "Risk", view.risk)
        config["Risk"] = st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)
    if has_reasons:
        table["Why flagged"] = view.reasons.map(
            lambda t: " · ".join(f"{k}: {v}" if v else k for k, v in pretty_reasons(t)))
        config["Why flagged"] = st.column_config.TextColumn(width="large")

    event = st.dataframe(table, column_config=config, hide_index=True, width="stretch",
                         height=420, on_select="rerun", selection_mode="single-row")

    rows = event.selection.rows if event else []
    if rows:
        tx = view.iloc[rows[0]]
        st.markdown(f"#### {tx.transaction_id}")
        if has_reasons and tx.reasons:
            st.markdown("**Why it was flagged**<br>" + chips(pretty_reasons(tx.reasons)), unsafe_allow_html=True)
            st.write("")
        st.markdown(key_values([
            ("Amount", f"£{tx.amount_gbp:,.2f}"),
            ("vs. typical monthly spend", f"{tx.amount_vs_typical:.1f}×"),
            ("When", f"{tx.timestamp:%a %d %b, %H:%M}"),
            ("Type", TYPE_LABELS.get(tx.transaction_type, tx.transaction_type)),
            ("Merchant country", COUNTRY_LABELS.get(tx.merchant_country, tx.merchant_country)),
            ("Account", ACCOUNT_LABELS.get(tx.account_type, tx.account_type)),
            ("Unknown device", "Yes" if tx.unknown_device else "No"),
            ("Failed logins", int(tx.failed_logins)),
            ("New payee", "Yes" if tx.new_payee else "No"),
            ("Travel notice", "Yes" if tx.travel_notice else "No"),
            ("Customer", tx.customer_id),
        ] + ([("Model risk", f"{tx.risk:.0%}")] if has_risk else [])), unsafe_allow_html=True)

# ---------- 3. patterns ----------
with tab_patterns:
    st.markdown("What the flagged transactions have in common, compared with the ones the model left alone.")
    st.plotly_chart(charts.signal_share(df, SIGNALS), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(charts.flag_rate(df, "transaction_type", "Share flagged, by transaction type", TYPE_LABELS),
                   width="stretch")
    b.plotly_chart(charts.flag_rate(df, "amount_band", "Share flagged, by amount"), width="stretch")
    c, d = st.columns(2)
    c.plotly_chart(charts.flagged_by_hour(df), width="stretch")
    d.plotly_chart(charts.flag_rate(df, "merchant_country", "Share flagged, by merchant country", COUNTRY_LABELS),
                   width="stretch")
    counts = reason_counts(df)
    if len(counts):
        st.plotly_chart(charts.top_reasons(counts), width="stretch")

# ---------- 4. compare ----------
with tab_compare:
    others = [m for m in models if m != model]
    if not others:
        st.info("Only one model so far. When teammates save another predictions CSV in `outputs/`, "
                "you can compare which transactions each model flags here.")
    else:
        other = st.selectbox("Compare with", others)
        df_b, info_b, err_b = load_model(other)
        if err_b:
            st.error(err_b)
        else:
            merged = df[["transaction_id", "is_flagged"]].merge(
                df_b[["transaction_id", "is_flagged"]], on="transaction_id", suffixes=("_a", "_b"))
            a_, b_ = merged.is_flagged_a.eq(1), merged.is_flagged_b.eq(1)
            both, only_a, only_b = int((a_ & b_).sum()), int((a_ & ~b_).sum()), int((~a_ & b_).sum())
            agree = (a_ == b_).mean()
            cards([
                (info.get("name", model), f"{int(a_.sum()):,}", "flagged", MUTED),
                (info_b.get("name", other), f"{int(b_.sum()):,}", "flagged", MUTED),
                ("Agree on", f"{agree:.1%}", "of all transactions", GENUINE),
            ])
            st.plotly_chart(charts.overlap(both, only_a, only_b, model, other), width="stretch")
            diff = merged[a_ != b_].transaction_id
            if len(diff):
                st.markdown("#### Where they disagree")
                show = df[df.transaction_id.isin(diff)][["transaction_id", "amount_gbp", "transaction_type",
                                                         "merchant_country"]].copy()
                show["Flagged by"] = show.transaction_id.map(
                    dict(zip(merged.transaction_id, merged.is_flagged_a.map({1: model, 0: other}))))
                st.dataframe(show.rename(columns={"transaction_id": "ID", "amount_gbp": "Amount",
                                                  "transaction_type": "Type", "merchant_country": "Country"}),
                             hide_index=True, width="stretch",
                             column_config={"Amount": st.column_config.NumberColumn(format="£%.2f")})
