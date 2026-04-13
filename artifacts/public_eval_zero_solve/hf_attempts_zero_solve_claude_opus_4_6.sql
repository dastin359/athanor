WITH all_attempts AS (
  SELECT 'attempt_1' AS which, attempt_1 AS attempt FROM attempts WHERE attempt_1 IS NOT NULL
  UNION ALL
  SELECT 'attempt_2' AS which, attempt_2 AS attempt FROM attempts WHERE attempt_2 IS NOT NULL
),
parsed AS (
  SELECT
    regexp_extract(attempt, '"task_id":"([0-9a-f]+)"', 1) AS task_id,
    regexp_extract(attempt, '"model":"([^"]+)"', 1) AS model,
    CASE WHEN attempt LIKE '%"correct":true%' THEN 1 ELSE 0 END AS solved
  FROM all_attempts
)
SELECT
  task_id,
  SUM(solved) AS solves,
  COUNT(*) AS attempts,
  ROUND(100.0 * SUM(solved) / COUNT(*), 2) AS solve_rate_pct
FROM parsed
WHERE task_id <> ''
  AND model = 'claude-opus-4-6'
GROUP BY task_id
HAVING SUM(solved) = 0
ORDER BY attempts DESC, task_id;
