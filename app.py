# app.py — Repayment Outcomes Lab (uploads + AI via Groq + .env support)
# Streamlit dashboard for repayment KPIs, A/B testing, cohorts, and lifecycle planning.
# Widgets are kept OUTSIDE cached functions to avoid CachedWidgetWarning.

from pathlib import Path
from math import erf, sqrt as msqrt
import io, zipfile, os

from dotenv import load_dotenv  # ← .env support

import streamlit as st
import pandas as pd
import numpy as np

# Load environment variables from .env (local) before anything else
load_dotenv(override=True)

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="Repayment Outcomes Lab", page_icon="💳", layout="wide")
st.title("💳 Repayment Outcomes Lab")
st.caption(
    "Analyze repayment outcomes with KPIs, A/B tests, cohorts, and lifecycle planning. "
    "Use **Demo (synthetic)** data, **Upload bundle (.zip)**, **Upload individual CSVs**, or **Local folder (./data)**."
)

DATA_DIR = Path(__file__).parent / "data"

REQUIRED = {
    "users":      ["user_id","signup_date","channel","risk_segment","autopay"],
    "loans":      ["loan_id","user_id","orig_date","due_date","amount_due","apr","term_months"],
    "repay":      ["loan_id","user_id","due_date","paid_date","amount_due","dpd","charged_off","recovered_amount"],
    "assign":     ["user_id","experiment_name","variant"],
    "messages":   ["message_id","user_id","loan_id","campaign","send_time","delivered","clicked","opted_out"],
}
PARSE_DATES = {
    "users": ["signup_date","first_orig"],
    "loans": ["orig_date","due_date"],
    "repay": ["due_date","paid_date"],
    "messages": ["send_time"],
}

# =========================================
# Helpers (no widgets in cached functions)
# =========================================
def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _validate(df: pd.DataFrame, name: str) -> bool:
    need = REQUIRED[name]
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"`{name}` is missing columns: {missing}")
        return False
    return True

def _ensure_user_cohort(users_df: pd.DataFrame, loans_df: pd.DataFrame) -> pd.DataFrame:
    if "first_orig" not in users_df.columns or users_df["first_orig"].isna().all():
        fo = loans_df.groupby("user_id")["orig_date"].min().rename("first_orig")
        users_df = users_df.merge(fo, on="user_id", how="left")
    users_df["cohort_month"] = users_df["first_orig"].dt.to_period("M").astype("string")
    return users_df

def _read_csv_maybe_gz(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return pd.read_csv(gz)
    return None

def _read_local_folder() -> tuple[pd.DataFrame, ...] | None:
    if not DATA_DIR.exists():
        return None
    users = _read_csv_maybe_gz(DATA_DIR / "users.csv")
    loans = _read_csv_maybe_gz(DATA_DIR / "loans.csv")
    repay = _read_csv_maybe_gz(DATA_DIR / "repayments.csv")
    assign = _read_csv_maybe_gz(DATA_DIR / "assignments.csv")
    msgs  = _read_csv_maybe_gz(DATA_DIR / "messages.csv")
    if any(x is None for x in [users, loans, repay, assign, msgs]):
        return None
    return users, loans, repay, assign, msgs

# ---------- Cached data builders/parsers (pure, no widgets) ----------
@st.cache_data
def _make_synthetic(seed: int = 42, n_users: int = 20000):
    rng = np.random.default_rng(seed)
    CHANNELS = ["Paid Social", "Search", "Organic", "Referral", "Affiliate"]
    RISK = ["Low", "Medium", "High"]

    # Users
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-06-30")
    days = (end - start).days
    user_ids = np.arange(1, n_users + 1)
    signup = start + pd.to_timedelta(rng.integers(0, days, size=n_users), unit="D")
    users = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup,
        "channel": rng.choice(CHANNELS, size=n_users, p=[0.28, 0.22, 0.32, 0.12, 0.06]),
        "risk_segment": rng.choice(RISK, size=n_users, p=[0.45, 0.40, 0.15]),
        "autopay": rng.choice([True, False], size=n_users, p=[0.55, 0.45]),
    })

    # Loans (0–3 per user)
    loan_counts = rng.choice([0, 1, 2, 3], size=n_users, p=[0.15, 0.55, 0.25, 0.05])
    rows, loan_id = [], 1
    for i in range(n_users):
        n = int(loan_counts[i])
        last = users.loc[i, "signup_date"]
        for _ in range(n):
            orig = last + pd.to_timedelta(int(rng.integers(15, 121)), unit="D")
            term = int(rng.integers(1, 4))
            due = orig + pd.to_timedelta(30 * term, unit="D")
            amt = float(np.round(max(50, rng.normal(250, 110)), 2))
            apr = float(np.round(rng.normal(0.22, 0.06), 3))
            rows.append([loan_id, users.loc[i, "user_id"], orig, due, amt, apr, term])
            loan_id += 1
            last = orig
    loans = pd.DataFrame(rows, columns=["loan_id", "user_id", "orig_date", "due_date", "amount_due", "apr", "term_months"])

    # Experiment assignment
    assign = pd.DataFrame({
        "user_id": users["user_id"],
        "experiment_name": "pre_due_nudge_v1",
        "variant": rng.choice(["A", "B"], size=n_users, p=[0.5, 0.5])
    })

    # Repayments (with ~2% absolute uplift for B)
    ch = {"Paid Social": -0.03, "Search": 0.01, "Organic": 0.02, "Referral": 0.035, "Affiliate": -0.015}
    rk = {"Low": 0.05, "Medium": 0.0, "High": -0.08}
    auto = 0.06
    uplift = {"A": 0.0, "B": 0.02}

    rep_rows = []
    for _, l in loans.iterrows():
        u = users.loc[users.user_id == l.user_id].iloc[0]
        v = assign.loc[assign.user_id == l.user_id, "variant"].iloc[0]
        p_on_time = 0.76 + ch[u.channel] + rk[u.risk_segment] + (auto if u.autopay else 0.0) + uplift[v]
        p_on_time = max(0.05, min(0.98, p_on_time + float(rng.normal(0, 0.03))))
        on_time = rng.random() < p_on_time
        if on_time:
            dpd = 0
            paid_date = l.due_date - pd.to_timedelta(int(rng.integers(0, 3)), unit="D")
            charged_off = False
            recovered = 0.0
        else:
            bucket = rng.choice([15, 45, 75, 105], p=[0.6, 0.2, 0.12, 0.08] if u.risk_segment != "High" else [0.4, 0.25, 0.2, 0.15])
            dpd = int(max(1, rng.integers(bucket - 10, bucket + 10)))
            charged_off = dpd >= 120 and (rng.random() < (0.35 if u.risk_segment == "High" else 0.18))
            if charged_off:
                paid_date = pd.NaT
                recovered = float(np.round(max(0.0, rng.normal(0.15, 0.07)) * l.amount_due, 2))
            else:
                paid_date = l.due_date + pd.to_timedelta(dpd, unit="D")
                recovered = 0.0
        rep_rows.append([l.loan_id, l.user_id, l.due_date, paid_date, float(np.round(l.amount_due, 2)), int(dpd), bool(charged_off), float(np.round(recovered, 2))])

    repay = pd.DataFrame(rep_rows, columns=["loan_id", "user_id", "due_date", "paid_date", "amount_due", "dpd", "charged_off", "recovered_amount"])
    msgs = pd.DataFrame(columns=["message_id", "user_id", "loan_id", "campaign", "send_time", "delivered", "clicked", "opted_out"])

    first_orig = loans.groupby("user_id")["orig_date"].min().rename("first_orig")
    users = users.merge(first_orig, on="user_id", how="left")
    users["cohort_month"] = users["first_orig"].dt.to_period("M").astype("string")
    return users, loans, repay, assign, msgs

@st.cache_data
def _read_zip_bundle_bytes(zip_bytes: bytes):
    if zip_bytes is None:
        return None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        def get(name, parse_key=None):
            candidates = [name, name.replace(".csv", "_test.csv")]
            for cand in candidates:
                if cand in z.namelist():
                    df = pd.read_csv(z.open(cand))
                    if parse_key and parse_key in PARSE_DATES:
                        df = _parse_dates(df, PARSE_DATES[parse_key])
                    return df
            return None
        users = get("users.csv","users")
        loans = get("loans.csv","loans")
        repay = get("repayments.csv","repay")
        assign = get("assignments.csv",None)
        msgs  = get("messages.csv","messages")
        if any(x is None for x in [users, loans, repay, assign, msgs]):
            return "MISSING"
        return users, loans, repay, assign, msgs

@st.cache_data
def _read_csv_bytes(b: bytes):
    if b is None:
        return None
    return pd.read_csv(io.BytesIO(b))

# =========================================
# Sidebar widgets (UI lives OUTSIDE cached funcs)
# =========================================
st.sidebar.subheader("Data source")
mode = st.sidebar.radio(
    "Choose your source",
    ["Demo (synthetic)", "Upload bundle (.zip)", "Upload individual CSVs", "Local folder (./data)"],
    index=0
)

zip_bytes = None
users_b = loans_b = repay_b = assign_b = msgs_b = None

if mode == "Upload bundle (.zip)":
    up_zip = st.sidebar.file_uploader("Upload a .zip with users.csv, loans.csv, repayments.csv, assignments.csv, messages.csv", type=["zip"])
    zip_bytes = up_zip.getvalue() if up_zip else None

elif mode == "Upload individual CSVs":
    up_users = st.sidebar.file_uploader("users.csv (or users_test.csv)", type=["csv","gz"])
    up_loans = st.sidebar.file_uploader("loans.csv (or loans_test.csv)", type=["csv","gz"])
    up_repay = st.sidebar.file_uploader("repayments.csv (or repayments_test.csv)", type=["csv","gz"])
    up_assign= st.sidebar.file_uploader("assignments.csv (or assignments_test.csv)", type=["csv","gz"])
    up_msgs  = st.sidebar.file_uploader("messages.csv (or messages_test.csv)", type=["csv","gz"])
    users_b  = up_users.getvalue() if up_users else None
    loans_b  = up_loans.getvalue() if up_loans else None
    repay_b  = up_repay.getvalue() if up_repay else None
    assign_b = up_assign.getvalue() if up_assign else None
    msgs_b   = up_msgs.getvalue() if up_msgs else None

# =========================================
# Load data based on UI choices
# =========================================
if mode == "Demo (synthetic)":
    users, loans, repay, assign, msgs = _make_synthetic()

elif mode == "Local folder (./data)":
    res = _read_local_folder()
    if res is None:
        st.info("No complete dataset found in ./data — falling back to synthetic demo.")
        users, loans, repay, assign, msgs = _make_synthetic()
    else:
        users, loans, repay, assign, msgs = res

elif mode == "Upload bundle (.zip)":
    res = _read_zip_bundle_bytes(zip_bytes)
    if res is None:
        st.info("Upload a .zip to proceed, or switch to Demo.")
        users, loans, repay, assign, msgs = _make_synthetic()
    elif res == "MISSING":
        st.error("Zip is missing one or more required files: users, loans, repayments, assignments, messages. Using demo data instead.")
        users, loans, repay, assign, msgs = _make_synthetic()
    else:
        users, loans, repay, assign, msgs = res

elif mode == "Upload individual CSVs":
    users = _read_csv_bytes(users_b)
    loans = _read_csv_bytes(loans_b)
    repay = _read_csv_bytes(repay_b)
    assign= _read_csv_bytes(assign_b)
    msgs  = _read_csv_bytes(msgs_b)
    if any(x is None for x in [users, loans, repay, assign, msgs]):
        st.info("Waiting for all CSVs — using demo data until uploads are complete.")
        users, loans, repay, assign, msgs = _make_synthetic()

# Parse dates & validate
users = _parse_dates(users, PARSE_DATES["users"])
loans = _parse_dates(loans, PARSE_DATES["loans"])
repay = _parse_dates(repay, PARSE_DATES["repay"])
msgs  = _parse_dates(msgs,  PARSE_DATES["messages"])

ok = True
ok &= _validate(users, "users")
ok &= _validate(loans, "loans")
ok &= _validate(repay, "repay")
ok &= _validate(assign, "assign")
ok &= _validate(msgs, "messages")
if not ok:
    st.stop()

users = _ensure_user_cohort(users, loans)

with st.sidebar.expander("Preview data"):
    st.write("users:", users.head(3))
    st.write("loans:", loans.head(3))
    st.write("repayments:", repay.head(3))

# =========================================
# Build base table for analyses
# =========================================
base = (
    loans.merge(repay, on=["loan_id", "user_id", "due_date"], how="left")
         .merge(assign[["user_id", "experiment_name", "variant"]], on="user_id", how="left")
         .merge(users[["user_id", "channel", "risk_segment", "autopay", "cohort_month"]], on="user_id", how="left")
)
base["on_time"] = (base["dpd"] == 0).astype(int)

# =========================================
# Sidebar filters
# =========================================
st.sidebar.header("Filters")
chan = st.sidebar.multiselect("Channel", sorted([c for c in users["channel"].dropna().unique()]))
risk = st.sidebar.multiselect("Risk Segment", sorted([r for r in users["risk_segment"].dropna().unique()]))
var  = st.sidebar.multiselect("Experiment Variant", ["A", "B"])

# ---- Groq key indicator (shows whether AI can run) ----
key_present = bool(os.environ.get("GROQ_API_KEY")) or ("GROQ_API_KEY" in st.secrets)
st.sidebar.caption("Groq key: ✅ detected" if key_present else "Groq key: ❌ not set")

df = base.copy()
if chan:
    df = df[df["channel"].isin(chan)]
if risk:
    df = df[df["risk_segment"].isin(risk)]
if var:
    df = df[df["variant"].isin(var)]

# =========================================
# Tabs (includes 🤖 Ask AI)
# =========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 KPI Dashboard", "🧪 A/B Testing", "📊 Cohorts", "🧭 Lifecycle Planner", "🤖 Ask AI"])

# ---------- KPI Dashboard ----------
with tab1:
    st.subheader("Core Repayment KPIs")
    col1, col2, col3, col4 = st.columns(4)
    on_time_rate = df["on_time"].mean() if len(df) else 0.0
    dpd30        = (df["dpd"] >= 30).mean() if len(df) else 0.0
    chargeoff    = df["charged_off"].mean() if len(df) else 0.0
    autopay_rate = users["autopay"].mean() if len(users) else 0.0
    col1.metric("On-time repayment rate", f"{on_time_rate:.1%}")
    col2.metric("30+ DPD rate",           f"{dpd30:.1%}")
    col3.metric("Charge-off rate",        f"{chargeoff:.1%}")
    col4.metric("Autopay adoption",       f"{autopay_rate:.1%}")

    st.markdown("**Breakouts**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("On-time by Channel")
        if len(df):
            st.bar_chart(df.groupby("channel")["on_time"].mean().sort_values())
        else:
            st.info("No rows after filters.")
    with c2:
        st.write("30+ DPD by Risk Segment")
        if len(df):
            st.bar_chart(
                (df.assign(dpd30=(df["dpd"] >= 30).astype(int))
                   .groupby("risk_segment")["dpd30"]
                   .mean()
                   .sort_values())
            )
        else:
            st.info("No rows after filters.")

# ---------- A/B Testing ----------
with tab2:
    st.subheader("A/B Test Readout — pre_due_nudge_v1")
    ab = df.dropna(subset=["variant"])
    if len(ab) == 0:
        st.info("No experiment rows after filters.")
    else:
        summary = (
            ab.groupby("variant")
              .agg(users=("user_id", "nunique"),
                   loans=("loan_id", "count"),
                   on_time=("on_time", "mean"))
              .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        if set(summary["variant"]) == {"A", "B"}:
            a = summary[summary["variant"] == "A"].iloc[0]
            b = summary[summary["variant"] == "B"].iloc[0]
            p1, n1 = float(a["on_time"]), int(a["loans"])
            p2, n2 = float(b["on_time"]), int(b["loans"])
            p_pool = (p1 * n1 + p2 * n2) / max(1, (n1 + n2))
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / max(1, n1) + 1 / max(1, n2))) if (n1 > 0 and n2 > 0) else np.nan
            z = (p2 - p1) / se if (se and se > 0) else np.nan
            p_val = 2 * (1 - 0.5 * (1 + erf(abs(z) / msqrt(2)))) if not np.isnan(z) else np.nan

            st.markdown(f"**Lift (B − A):** {(p2 - p1):.2%}  |  **z:** {z:.2f}  |  **p-value:** {p_val:.4f}")
            st.write("On-time by Variant")
            st.bar_chart(ab.groupby("variant")["on_time"].mean())
        else:
            st.info("Need both A and B variants to compute lift.")

    st.divider()
    st.caption("Guardrails to monitor: opt-outs, CS tickets (proxy), repayment volatility.")

# ---------- Cohorts ----------
with tab3:
    st.subheader("Cohort Analysis (by first loan month)")

    first_loan = loans.groupby("user_id")["orig_date"].min().rename("first_orig")
    tmp = base.merge(first_loan, on="user_id", how="left", suffixes=("", "_first"))
    tmp["cohort_month"] = pd.to_datetime(tmp["first_orig"]).dt.to_period("M").astype("string")
    tmp["event_month"]  = pd.to_datetime(tmp["orig_date"]).dt.to_period("M").astype("string")

    def month_diff(a: str, b: str) -> int:
        try:
            ay, am = map(int, a.split("-"))
            by, bm = map(int, b.split("-"))
            return (by - ay) * 12 + (bm - am)
        except Exception:
            return 0

    tmp["month_n"] = [month_diff(c, e) for c, e in zip(tmp["cohort_month"], tmp["event_month"])]
    tmp["on_time"] = (tmp["dpd"] == 0).astype(int)

    grp = (tmp.groupby(["cohort_month","month_n"])
              .agg(on_time_rate=("on_time","mean"), n=("on_time","size"))
              .reset_index())

    table_rates  = (grp.pivot(index="cohort_month", columns="month_n", values="on_time_rate")*100).round(1)
    table_counts =  grp.pivot(index="cohort_month", columns="month_n", values="n")

    # Chronological row order
    table_rates = table_rates.reindex(sorted(table_rates.index, key=lambda s: pd.to_datetime(s, format="%Y-%m")))
    table_counts = table_counts.reindex(table_rates.index)

    st.write("On-time % by Cohort × Month (blank cells can mean tiny n or no loans due)")
    st.dataframe(table_rates, use_container_width=True)

    with st.expander("Show counts (n) per cell"):
        st.dataframe(table_counts, use_container_width=True)

    st.write("Average on-time by month since start")
    pivot_for_line = grp.groupby("month_n")["on_time_rate"].mean()
    if len(pivot_for_line):
        st.line_chart(pivot_for_line)
    else:
        st.info("No cohort data to chart.")

# ---------- Lifecycle Planner ----------
with tab4:
    st.subheader("Lifecycle Messaging Planner (Preview)")
    col1, col2, col3 = st.columns(3)
    t3 = col1.checkbox("T-3 Pre-due (treatment-only)", True)
    t1 = col2.checkbox("T-1 Pre-due", True)
    p2 = col3.checkbox("+2d Grace/Extension", True)

    # Choose an anchor date so preview has data
    if "due_date" in df.columns and pd.notnull(df["due_date"]).any():
        anchor = pd.to_datetime(df["due_date"]).min()
    else:
        anchor = pd.Timestamp.today().normalize()

    horizon = st.slider("Preview horizon (days)", 3, 14, 7)
    due_window = (pd.to_datetime(df["due_date"]) >= anchor) & \
                 (pd.to_datetime(df["due_date"]) <= anchor + pd.Timedelta(days=horizon))
    preview = df.loc[due_window, ["user_id", "loan_id", "due_date", "variant", "autopay"]].copy() if len(df) else pd.DataFrame()

    if not preview.empty:
        preview["T-3"] = (preview["variant"] == "B") & t3
        preview["T-1"] = t1
        preview["+2d"] = p2
        st.write("Upcoming sends (preview):")
        st.dataframe(preview.head(200), use_container_width=True)
    else:
        st.info("No loans in the preview window. Try widening the horizon or clearing filters.")

    st.caption("Use a 10% global holdout, quiet hours, and frequency caps before production.")

# ---------- 🤖 Ask AI (Groq) ----------
with tab5:
    st.subheader("Ask AI about your dashboard")

    # Sidebar AI settings (model picker)
    st.sidebar.subheader("AI settings")
    model_choice = st.sidebar.selectbox(
        "Groq model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0,
        help="Pick faster vs. bigger models."
    )

    # Lazy import so the app still runs without groq installed
    groq_available = True
    try:
        from groq import Groq
    except Exception:
        groq_available = False

    # Build small, privacy-safe context (aggregates only)
    kpis = {
        "on_time_rate": float(df["on_time"].mean()) if len(df) else None,
        "dpd30_rate": float((df["dpd"] >= 30).mean()) if len(df) else None,
        "chargeoff_rate": float(df["charged_off"].mean()) if len(df) else None,
    }
    # A/B summary (if available)
    ab = df.dropna(subset=["variant"])
    ab_summary = None
    if len(ab):
        ab_summary = (
            ab.groupby("variant")
              .agg(loans=("loan_id","count"), on_time=("on_time","mean"))
              .reset_index()
              .to_dict(orient="records")
        )

    question = st.text_input("Ask a question (e.g., 'What is on-time? Did B beat A? Why is month 2 low?')")

    def _ai_answer(q: str) -> str:
        if not groq_available:
            return "Install the Groq SDK first: `pip install groq`."
        # Read the key from environment or Streamlit secrets (works local + cloud)
        api_key = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "Set `GROQ_API_KEY` in .env (local) or .streamlit/secrets.toml (cloud) to enable the AI helper."
        client = Groq(api_key=api_key)
        system = (
            "You are a fintech product analyst. Answer briefly in plain English. "
            "Use only the provided aggregates (no PII). If asked about A/B, report lift = on_time_B - on_time_A, "
            "and what it means. Explain numbers simply."
        )
        user_payload = {
            "question": q,
            "kpis": kpis,
            "ab": ab_summary,
            "notes": "on_time = (dpd==0). Cohort line is months-since-start (0..N)."
        }
        try:
            resp = client.chat.completions.create(
                model=model_choice,
                temperature=0.2,
                max_tokens=500,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":str(user_payload)}
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"AI error: {e}"

    if st.button("Answer") and question:
        with st.spinner("Thinking..."):
            st.write(_ai_answer(question))

    st.caption("Powered by Groq. Only aggregated metrics from this page are sent—no raw user data.")
