
# Repayment Outcomes Lab (Streamlit + AI) #
A simple dashboard to measure and improve on-time repayment in consumer fintech.
It works even if you don’t upload data (it has demo/synthetic data), and it can also read your CSV files.
There’s an optional AI tab (Groq) so you can ask questions in plain English.

## Why I built this ##
Lots of teams send reminders, but fewer can show what actually works.
This project is a small, clear measurement lab: it helps you see if a reminder (like a T-3 “3 days before due” nudge) really increases on-time payments—and by how much.

I’m an Operations/Product person. I used AI + low-code tools to build a working demo that shows the full PM/Ops loop:

Define KPIs → Run an A/B test → Read cohorts → Plan lifecycle messages.

## Problem statement ##
- We want more people to pay on time (good for users and the business).
- We send reminders, but we don’t know which reminder causes the improvement.
- We need a small tool that shows A vs B, in a way anyone can understand.

Goal: Measure on-time rate, check guardrails (30+ DPD, charge-offs, opt-outs), and decide which nudges to keep.

## What this app does (purpose) ##
- Shows the on-time repayment rate and other key metrics.
- Compares Variant A vs Variant B (A/B test) and gives lift and a z-value/p-value.
- Shows cohorts (by first loan month) so you see if results hold over time.
- Has a Lifecycle Planner to preview who would get reminders in the next few days.
- Optional AI tab (Groq) to ask questions like “Did B beat A?” or “Why is month 2 low?”

## Features (at a glance) ##
+ 📈 KPI Dashboard
   - On-time %, 30+ DPD %, Charge-off %.
   - Breakouts by Channel and Risk segment.

+ 🧪 A/B Testing
   - Rates for A and B, Lift (B − A), z-value and p-value.

+ 📊 Cohorts
   - Rows = cohort month (when users got their first loan).
   - Columns = month since start (0,1,2…).
   - Each cell is on-time % for that group in that month.

+ 🧭 Lifecycle Planner
   - Anchor = start date of the preview.
   - Horizon = how many days ahead to look.
   - See who would get T-3, T-1, +2d messages.

+ 🤖 Ask AI (Groq) (optional)
   - Ask plain-English questions; it uses only page aggregates (no PII).

## Who this is for ##
- Ops / Product people who want to prove what works.
- Beginner-friendly: you can run it with no coding—just install and click.

## How to load data ##
In the sidebar, pick one:
- Demo (synthetic) → built-in fake data (works instantly).
- Upload bundle (.zip) → a zip with those 5 CSVs at the root.
- Upload individual CSVs → upload all five one by one.
- Local folder (./data) → put the five CSVs into a data/ folder next to app.py.
If uploads are missing, the app falls back to demo data so it still works.

## How to understand the charts ##
- On-time % (higher is better) = how many loans were paid on/before due date.
- 30+ DPD % (lower is better) = share of loans 30 days or more late.
- A/B Lift = on_time_B − on_time_A (how much better B is than A).
- z-value ≈ “how strong is the difference?”
- About 2 or more means it’s likely real (not luck).

### Cohort table ###
- Row (cohort month) = when users started (e.g., 2025-06).
- Column (month since start) = 0,1,2… (their 1st, 2nd, 3rd month).
- Cell value = on-time % for that group at that stage.
   100 = all on time that month (but check the count n; small n can be noisy).
   can also mean no data in that cell (the app notes this).

### Lifecycle Planner ###
- Anchor = start date of preview.
- Horizon = how many days ahead to look.
- Shows who would get T-3, T-1, and +2d.

## Business impact (math) ##
- **N** = due loans this month
- **L** = lift in on-time (e.g., 2% = 0.02)
- **C** = cost avoided per on-time (collections, support, risk)
- **F** = late fee you’d lose if they pay on time
- **m** = cost per message (e.g., $0.01/SMS)
- **S** = messages sent (e.g., T-3 and T-1 → about 2×N)
**Net ≈ (N × L × (C − F)) − (S × m)**
**If C − F > 0**, even a tiny lift can pay for itself.

## Limitations ##
- The demo treats “one outcome per loan.” Real systems often have installments.
- The cohort table may have small samples in later months—check counts.
- The A/B math is simple (two-proportion z). In production you may want a full experimentation platform.

## Roadmap (future ideas) ##
- Add installment-level modeling and roll-rate trees.
- Power analysis (how big a test you need).
- Holdouts, quiet hours, frequency caps in the planner.
- Message channel breakdown (SMS vs push vs email).
- Cost & ROI panel with sliders (L, C, F, m).
- Export winning recipe to Braze/Iterable.

## Conclusion ##
This app is a small, clear lab for repayment outcomes.
It helps you:
   - measure what matters (on-time payments),
   - prove what works (A/B lift),
   - check safety (guardrails),
   - and plan next steps (lifecycle preview).

It’s beginner-friendly, runs locally, and is easy to deploy.
Use it to make evidence-based lifecycle decisions — then scale only the nudges that help.


