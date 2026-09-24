-- Run after electricity-evaluate.sql in the same DuckDB session.
.bail on
.timer on

CREATE TEMP VIEW future_inputs AS
SELECT id, timestamp AS date,
       "Ampirion Load Forecast" AS load_forecast,
       "PV+Wind Forecast" AS renewables_forecast
FROM read_parquet('outputs/electricity/prices-next-day.parquet');

CREATE TEMP TABLE forecast_with_covariates AS
SELECT * FROM tfc_forecast(
    'history', 24,
    target_col := 'price',
    timestamp_col := 'date',
    id_cols := ['id'],
    frequency := '1 hour',
    context := 512,
    covariate_cols := ['load_forecast', 'renewables_forecast'],
    future_table := 'future_inputs',
    quantiles := [0.10, 0.25, 0.50, 0.75, 0.90]
);

SELECT * FROM forecast_with_covariates ORDER BY id, date;

CREATE TEMP TABLE evaluation_result AS
SELECT * FROM tfc_evaluate(
    'history', 'actuals',
    horizon := 24,
    target_col := 'price',
    timestamp_col := 'date',
    id_cols := ['id'],
    frequency := '1 hour',
    context := 512,
    covariate_cols := ['load_forecast', 'renewables_forecast'],
    future_table := 'future_inputs',
    seasonal_period := 24
);

SELECT 'History only' AS inputs, mae, rmse, mase FROM baseline_evaluation
UNION ALL
SELECT 'Load + renewables forecasts', mae, rmse, mase FROM evaluation_result;
