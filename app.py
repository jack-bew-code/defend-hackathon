"""Fraud detection demo for the judges.

Display only: shows the prediction files the model team saves in outputs/.
Run locally with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from ui import charts
from ui import playground as pg
from ui.load import (ACCOUNT_LABELS, COUNTRY_LABELS, TYPE_LABELS, list_models, load_model,
                     model_label, pretty_reasons, reason_counts)
from ui.style import FRAUD, GENUINE, MUTED, cards, chips, inject, key_values

st.set_page_config(page_title="Fraud Radar", page_icon="🛡️", layout="wide")
inject()


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
    model = st.selectbox("Model", models, format_func=model_label, help="Every CSV in outputs/ appears here.")

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

tab_overview, tab_flags, tab_patterns, tab_compare, tab_settings = st.tabs(
    ["Overview", "Flagged transactions", "What the model catches", "Compare models", "Try settings"]
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
        other = st.selectbox("Compare with", others, format_func=model_label)
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
                show["transaction_type"] = show.transaction_type.map(TYPE_LABELS).fillna(show.transaction_type)
                show["merchant_country"] = show.merchant_country.map(COUNTRY_LABELS).fillna(show.merchant_country)
                show["Flagged by"] = show.transaction_id.map(dict(zip(
                    merged.transaction_id, merged.is_flagged_a.map({1: "Only " + model, 0: "Only " + other}))))
                st.dataframe(show.rename(columns={"transaction_id": "ID", "amount_gbp": "Amount",
                                                  "transaction_type": "Type", "merchant_country": "Country"}),
                             hide_index=True, width="stretch",
                             column_config={"Amount": st.column_config.NumberColumn(format="£%.2f")})

# ---------- 5. try settings ----------
@st.cache_data(show_spinner=False)
def _pg_data():
    return pg.load_data()


@st.cache_data(show_spinner=False, max_entries=64)
def _pg_run(settings: tuple):
    train, test = _pg_data()
    return pg.run(pg.Settings(**dict(settings)), train, test)


def _run(s: pg.Settings) -> dict:
    return _pg_run(tuple(pg.settings_dict(s).items()))


def _pg_reset():
    for k, v in pg.settings_dict(pg.RECOMMENDED).items():
        if k == "features":
            for g in pg.ALL_GROUPS:
                st.session_state[f"pg_feat_{g}"] = g in v
        else:
            st.session_state[f"pg_{k}"] = v


def _changes(s: pg.Settings) -> str:
    rec, cur = pg.settings_dict(pg.RECOMMENDED), pg.settings_dict(s)
    out = []
    for k, v in cur.items():
        if k == "features":
            dropped = [pg.FEATURE_GROUPS[g][0] for g in pg.ALL_GROUPS if g not in v]
            if dropped:
                out.append("without " + ", ".join(dropped))
        elif v != rec[k]:
            out.append(f"{pg.PARAMS[k][0]} {rec[k]} → {v}")
    return "; ".join(out) or "recommended settings"


if "pg_cutoff" not in st.session_state:
    _pg_reset()
st.session_state.setdefault("pg_history", [])

with tab_settings:
    st.markdown("Change the model's settings, run it, and see how the score moves. Each run trains on the "
                "earlier training transactions and is scored on the most recent ones it hasn't seen "
                "(+10 caught · −5 false alarm · −10 missed). The model branches are not changed.")
    left, right = st.columns([1, 2.6], gap="medium")

    with left:
        with st.form("pg_form", border=True):
            st.markdown("**Decision**")
            label, lo, hi, step, help_ = pg.PARAMS["cutoff"]
            st.slider(label, min_value=lo, max_value=hi, step=step, key="pg_cutoff", help=help_, format="%.2f")
            st.markdown("**Model**")
            for k in ["learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]:
                label, lo, hi, step, help_ = pg.PARAMS[k]
                st.slider(label, min_value=lo, max_value=hi, step=step, key=f"pg_{k}", help=help_,
                          format="%.2f" if isinstance(step, float) else "%d")
            st.markdown("**Signals the model can use**")
            for g, (label, cols) in pg.FEATURE_GROUPS.items():
                st.checkbox(label, key=f"pg_feat_{g}", help="Columns: " + ", ".join(cols))
            st.markdown("**Test**")
            label, lo, hi, step, help_ = pg.PARAMS["holdout_size"]
            st.slider(label, min_value=lo, max_value=hi, step=step, key="pg_holdout_size", help=help_)
            submitted = st.form_submit_button("Run model", type="primary", width="stretch")
        st.button("Reset to recommended", on_click=_pg_reset, width="stretch")

    current = pg.Settings(
        **{k: st.session_state[f"pg_{k}"] for k in pg.PARAMS},
        features=tuple(g for g in pg.ALL_GROUPS if st.session_state[f"pg_feat_{g}"]),
    )

    with right:
        if not current.features:
            st.warning("Pick at least one group of signals for the model to use.")
        else:
            if submitted or not st.session_state.pg_history:
                with st.spinner("Training and scoring…"):
                    res = _run(current)
                st.session_state.pg_last = current
                st.session_state.pg_history.append({
                    "run": f"Run {len(st.session_state.pg_history) + 1}",
                    "recommended": current == pg.RECOMMENDED,
                    "points": res["points"], "caught": res["caught"],
                    "false_alarms": res["false_alarms"], "missed": res["missed"],
                    "september_flags": res["september_flags"], "changes": _changes(current),
                })

        shown = st.session_state.get("pg_last", pg.RECOMMENDED)
        res, base = _run(shown), _run(pg.RECOMMENDED if shown.holdout_size == pg.RECOMMENDED.holdout_size
                                      else pg.replace_holdout(pg.RECOMMENDED, shown.holdout_size))
        st.markdown(f"#### Result: {_changes(shown)}")
        st.caption(f"Scored on {shown.holdout_size:,} unseen transactions ({res['holdout_period']}) · "
                   f"maximum possible {res['max_points']:,} points · deltas compare with the recommended settings")
        m = st.columns(5)
        m[0].metric("Points", f"{res['points']:,}", delta=res["points"] - base["points"] or None)
        m[1].metric("Caught", res["caught"], delta=res["caught"] - base["caught"] or None)
        m[2].metric("False alarms", res["false_alarms"], delta=res["false_alarms"] - base["false_alarms"] or None,
                    delta_color="inverse")
        m[3].metric("Missed", res["missed"], delta=res["missed"] - base["missed"] or None,
                    delta_color="inverse")
        m[4].metric("Sept flags", res["september_flags"],
                    delta=res["september_flags"] - base["september_flags"] or None, delta_color="off")

        curve = pg.cutoff_curve(res["holdout_risk"], res["holdout_actual"])
        st.plotly_chart(charts.cutoff_curve(curve, shown.cutoff, pg.RECOMMENDED.cutoff), width="stretch")

        hist = pd.DataFrame(st.session_state.pg_history)
        if len(hist):
            st.plotly_chart(charts.runs_bar(hist), width="stretch")
            st.dataframe(
                hist.rename(columns={"run": "Run", "points": "Points", "caught": "Caught",
                                     "false_alarms": "False alarms", "missed": "Missed",
                                     "september_flags": "Sept flags", "changes": "Settings changed"})
                    .drop(columns="recommended"),
                hide_index=True, width="stretch",
                column_config={"Settings changed": st.column_config.TextColumn(width="large")})
        b1, b2 = st.columns(2)
        b1.download_button("Download September predictions (this run)",
                           res["predictions"].to_csv(index=False).encode(),
                           file_name="playground_predictions.csv", mime="text/csv", width="stretch",
                           help="Same format as outputs/. Commit it there to show it in the other tabs.")
        if b2.button("Clear run history", width="stretch"):
            st.session_state.pg_history = []
            st.session_state.pop("pg_last", None)
            st.rerun()
