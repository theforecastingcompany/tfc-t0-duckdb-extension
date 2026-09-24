-- Run after electricity-forecast.sql in the same DuckDB session.
.bail on
.timer on

CREATE TEMP VIEW actuals AS
SELECT id, timestamp AS date, target AS price
FROM read_parquet('outputs/electricity/prices-next-day.parquet');

CREATE TEMP TABLE baseline_evaluation AS
SELECT * FROM tfc_evaluate(
    'history', 'actuals',
    horizon := 24,
    target_col := 'price',
    timestamp_col := 'date',
    id_cols := ['id'],
    frequency := '1 hour',
    context := 512,
    seasonal_period := 24
);

SELECT id, n, mae, rmse, mase FROM baseline_evaluation;
