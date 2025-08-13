
-- On-time repayment rate (overall)
SELECT AVG(CASE WHEN dpd = 0 THEN 1 ELSE 0 END) AS on_time_rate
FROM repayments;

-- 30+ DPD rate
SELECT AVG(CASE WHEN dpd >= 30 THEN 1 ELSE 0 END) AS dpd30_rate
FROM repayments;

-- On-time by acquisition channel
SELECT u.channel,
       AVG(CASE WHEN r.dpd = 0 THEN 1 ELSE 0 END) AS on_time_rate
FROM loans l
JOIN repayments r ON l.loan_id = r.loan_id
JOIN users u ON u.user_id = l.user_id
GROUP BY 1
ORDER BY 2 DESC;

-- A/B test readout (pre_due_nudge_v1)
SELECT a.variant,
       COUNT(*) AS loans,
       AVG(CASE WHEN r.dpd = 0 THEN 1 ELSE 0 END) AS on_time_rate
FROM assignments a
JOIN loans l ON l.user_id = a.user_id
JOIN repayments r ON r.loan_id = l.loan_id
WHERE a.experiment_name = 'pre_due_nudge_v1'
GROUP BY 1;

-- Cohort analysis: cohort_month × month_since_start
WITH first_loan AS (
  SELECT user_id, DATE_TRUNC('month', MIN(orig_date)) AS cohort_month
  FROM loans GROUP BY 1
),
events AS (
  SELECT l.user_id,
         DATE_TRUNC('month', l.orig_date) AS event_month,
         CASE WHEN r.dpd = 0 THEN 1 ELSE 0 END AS on_time
  FROM loans l
  JOIN repayments r ON r.loan_id = l.loan_id
)
SELECT TO_CHAR(f.cohort_month, 'YYYY-MM') AS cohort_month,
       (DATE_PART('year', e.event_month) - DATE_PART('year', f.cohort_month)) * 12
       + (DATE_PART('month', e.event_month) - DATE_PART('month', f.cohort_month)) AS month_n,
       AVG(e.on_time) AS on_time_rate
FROM first_loan f
JOIN events e ON e.user_id = f.user_id
GROUP BY 1,2
ORDER BY 1,2;

-- Roll rate: 30→60
WITH snap AS (
  SELECT loan_id,
         CASE WHEN dpd BETWEEN 30 AND 59 THEN 1 ELSE 0 END AS at_30,
         CASE WHEN dpd BETWEEN 60 AND 89 THEN 1 ELSE 0 END AS at_60
  FROM repayments
)
SELECT SUM(CASE WHEN at_30=1 AND at_60=1 THEN 1 ELSE 0 END)::float
       / NULLIF(SUM(at_30),0) AS roll_30_to_60
FROM snap;
