
# Repayment Outcomes Lab

An end-to-end **portfolio project** showing how a consumer fintech PM/Ops generalist analyzes and improves **repayment outcomes** with:
- **KPIs** (on-time rate, 30+ DPD, charge-offs)
- **Cohort analysis** (cohort-month × month-since-start)
- **A/B testing** of a **pre-due nudge** (T-3 push with payoff planner)
- **Lifecycle messaging** preview (T-3, T-1, +2d)

**Stack:** Python, Streamlit, Pandas, NumPy. Data is **synthetic** and safe to share.

---

## 📦 Project structure

```
repayment_outcomes_lab/
├─ app.py                 # Streamlit dashboard (KPI, A/B, cohorts, lifecycle)
├─ requirements.txt
├─ data/
│  ├─ users.csv
│  ├─ loans.csv
│  ├─ repayments.csv
│  ├─ assignments.csv
│  └─ messages.csv
└─ sql/
   └─ queries.sql         # Example SQL readouts
```

## 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

If Streamlit opens on a different port, copy the local URL from the terminal into your browser.

## 🧭 What to look at

1) **📈 KPI Dashboard** — On-time rate, 30+ DPD, charge-off, autopay adoption. Breakouts by channel/risk.
2) **🧪 A/B Testing** — Readout for `pre_due_nudge_v1` (Variant **B** gets a T-3 push). Shows lift, z-score, p-value.
3) **📊 Cohorts** — Heat-table of on-time by `cohort_month × month_since_start`, and average curve.
4) **🧭 Lifecycle Planner** — Preview which users would get **T-3 / T-1 / +2d** campaigns, with treatment-only logic for T-3.

## 📐 Metric definitions

- **On-time repayment rate** = share of loans with `dpd = 0`.
- **30+ DPD rate** = share with `dpd ≥ 30`.
- **Charge-off rate** = share with `charged_off = true`.
- **Cohort month** = first-loan month per user.
- **Lift** = on_time(B) − on_time(A).

## 🔬 Experiment: Pre-due nudge

**Hypothesis:** A T-3 pre-due push with a simple payoff planner increases on-time repayment by **≥2%** without raising opt-outs.

- **Primary metric:** On-time repayment rate
- **Guardrails:** Opt-out %, proxy CS volume
- **Randomization unit:** User
- **Notes:** This dataset bakes in ~2% absolute uplift for Variant B to emulate a “real” win.

## 🧰 SQL snippets

See `sql/queries.sql` for: KPI readouts, A/B test, cohorts, and 30→60 roll-rate.

## 📝 Resume bullets (paste-ready)

- Built an end-to-end **Repayment Outcomes Lab**: KPIs (on-time, 30+ DPD), cohort analysis, and an A/B framework on synthetic fintech loans; documented lift and guardrails.
- Prototyped a **lifecycle messaging** planner (T-3/T-1/+2d) with treatment-only logic and preview windows; aligned to incremental measurement via holdouts.
- Implemented a **Streamlit** dashboard and **SQL** queries for experiment readouts and cohort curves.

## 🧭 Next steps (extensions)

- Add **power analysis** & sample size calculator
- Simulate **autopay enrollment experiment**
- Include **collections strategy** variants (promise-to-pay scripts)
- Track **LTV** and **loss rates** by cohort/channel
- Add **Braze/Customer.io**-style export stubs
