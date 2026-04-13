WITH all_attempts AS (
  SELECT attempt_1 AS attempt FROM attempts WHERE attempt_1 IS NOT NULL
  UNION ALL
  SELECT attempt_2 AS attempt FROM attempts WHERE attempt_2 IS NOT NULL
),
parsed AS (
  SELECT
    regexp_extract(attempt, '"task_id":"([0-9a-f]+)"', 1) AS task_id,
    CAST(regexp_extract(attempt, '"pair_index":([0-9]+)', 1) AS INTEGER) AS pair_index,
    CASE WHEN attempt LIKE '%"correct":true%' THEN 1 ELSE 0 END AS solved
  FROM all_attempts
),
per_pair AS (
  SELECT
    task_id,
    pair_index,
    MAX(solved) AS ever_solved,
    COUNT(*) AS num_attempt_logs
  FROM parsed
  WHERE task_id <> ''
  GROUP BY task_id, pair_index
)
SELECT
  task_id,
  pair_index,
  num_attempt_logs
FROM per_pair
WHERE ever_solved = 0
  AND num_attempt_logs >= 80
ORDER BY task_id, pair_index;
