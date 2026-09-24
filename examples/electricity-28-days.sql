-- Run from the repository root after electricity.sql in the same session.
-- Model startup is already warm.
.bail on
.timer off

CREATE TEMP TABLE all_prices AS
SELECT id, timestamp AS date, target AS price,
       "Ampirion Load Forecast" AS load_forecast,
       "PV+Wind Forecast" AS renewables_forecast
FROM read_parquet(['outputs/electricity/prices-history.parquet', 'outputs/electricity/prices-next-day.parquet'], union_by_name=true);

-- Each (id, window_id) is an independent series. Day 0 is the supplied test day.
CREATE TEMP TABLE windows AS
SELECT i::INTEGER AS window_id,
       TIMESTAMP '2017-12-12' - i * INTERVAL '1 day' AS forecast_start
FROM range(28) AS days(i);

CREATE TEMP TABLE daily_history AS
SELECT p.*, w.window_id
FROM all_prices p JOIN windows w
  ON p.date >= w.forecast_start - INTERVAL '512 hours'
 AND p.date < w.forecast_start;

CREATE TEMP TABLE daily_actuals AS
SELECT p.id, w.window_id, p.date, p.price
FROM all_prices p JOIN windows w
  ON p.date >= w.forecast_start
 AND p.date < w.forecast_start + INTERVAL '24 hours';

-- Never include held-out prices in future model inputs.
CREATE TEMP TABLE daily_future AS
SELECT p.id, w.window_id, p.date, p.load_forecast, p.renewables_forecast
FROM all_prices p JOIN windows w
  ON p.date >= w.forecast_start
 AND p.date < w.forecast_start + INTERVAL '24 hours';

SELECT CASE WHEN
    (SELECT count(*) FROM daily_history) = 28 * 512
    AND (SELECT count(*) FROM daily_actuals) = 28 * 24
    AND (SELECT count(*) FROM daily_future) = 28 * 24
    THEN true ELSE error('Incomplete daily windows') END AS windows_verified;

-- These two timers include forecast-query preparation and materialization,
-- excluding downloads, model startup, and source-table preparation above.
.timer on
CREATE TEMP TABLE daily_baseline AS
SELECT * FROM tfc_forecast(
    'daily_history', 24, target_col := 'price', timestamp_col := 'date',
    id_cols := ['id', 'window_id'], frequency := '1 hour', context := 512,
    quantiles := [0.10, 0.25, 0.50, 0.75, 0.90], batch_size := 8);

CREATE TEMP TABLE daily_covariates AS
SELECT * FROM tfc_forecast(
    'daily_history', 24, target_col := 'price', timestamp_col := 'date',
    id_cols := ['id', 'window_id'], frequency := '1 hour', context := 512,
    quantiles := [0.10, 0.25, 0.50, 0.75, 0.90], batch_size := 8,
    covariate_cols := ['load_forecast', 'renewables_forecast'],
    future_table := 'daily_future');
.timer off

-- Score the saved predictions with DOUBLE arithmetic, as tfc_evaluate does.
CREATE TEMP TABLE daily_scores AS
SELECT b.window_id, min(b.date) AS forecast_start,
       avg(abs(b.prediction::DOUBLE - a.price::DOUBLE)) AS history_mae,
       avg(abs(c.prediction::DOUBLE - a.price::DOUBLE)) AS covariate_mae
FROM daily_baseline b
JOIN daily_covariates c USING (id, window_id, date)
JOIN daily_actuals a USING (id, window_id, date)
GROUP BY b.window_id;

-- Independently check the native evaluator against saved prediction errors.
CREATE TEMP TABLE daily_baseline_evaluation AS
SELECT * FROM tfc_evaluate(
    'daily_history', 'daily_actuals', horizon := 24,
    target_col := 'price', timestamp_col := 'date', id_cols := ['id', 'window_id'],
    frequency := '1 hour', context := 512, seasonal_period := 24, batch_size := 8);
CREATE TEMP TABLE daily_covariate_evaluation AS
SELECT * FROM tfc_evaluate(
    'daily_history', 'daily_actuals', horizon := 24,
    target_col := 'price', timestamp_col := 'date', id_cols := ['id', 'window_id'],
    frequency := '1 hour', context := 512, seasonal_period := 24, batch_size := 8,
    covariate_cols := ['load_forecast', 'renewables_forecast'],
    future_table := 'daily_future');

SELECT CASE WHEN
    (SELECT count(*) FROM daily_baseline) = 672
    AND (SELECT count(*) FROM daily_covariates) = 672
    AND (SELECT count(*) FROM daily_scores) = 28
    AND (SELECT count(*) FROM daily_baseline WHERE isfinite(prediction)) = 672
    AND (SELECT count(*) FROM daily_covariates WHERE isfinite(prediction)) = 672
    AND (SELECT count(*) FROM daily_scores s
         JOIN daily_baseline_evaluation b USING (window_id)
         JOIN daily_covariate_evaluation c USING (window_id)
         WHERE b.n = 24 AND c.n = 24
           AND abs(s.history_mae - b.mae) < 0.001
           AND abs(s.covariate_mae - c.mae) < 0.001) = 28
    AND ((SELECT model FROM tfc_models()) <> 't0-beta-onnx-fp16' OR (SELECT abs(avg(history_mae) - 9.5223376354) < 0.001
         AND abs(avg(covariate_mae) - 4.1338665060) < 0.001
         AND count(*) FILTER (WHERE covariate_mae < history_mae) = 27
         FROM daily_scores))
    THEN true ELSE error('28-day electricity verification failed') END AS daily_verified;

SELECT avg(history_mae) AS history_mae, avg(covariate_mae) AS covariate_mae,
       100 * (1 - avg(covariate_mae) / avg(history_mae)) AS reduction_percent,
       count(*) FILTER (WHERE covariate_mae < history_mae) AS improved_days
FROM daily_scores;

COPY (SELECT * FROM daily_scores ORDER BY forecast_start)
TO 'outputs/electricity/electricity-daily-scores.csv' (HEADER);
