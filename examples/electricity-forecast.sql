-- Run from the repository root after loading the extension.
.bail on
.timer on
SET threads=1;
SET preserve_insertion_order=false;

CREATE TEMP VIEW history AS
SELECT id, timestamp AS date, target AS price,
       "Ampirion Load Forecast" AS load_forecast,
       "PV+Wind Forecast" AS renewables_forecast
FROM read_parquet('outputs/electricity/prices-history.parquet')
WHERE timestamp >= TIMESTAMP '2017-11-20 16:00:00';

CREATE TEMP TABLE forecast_result AS
SELECT * FROM tfc_forecast(
    'history', 24,
    target_col := 'price',
    timestamp_col := 'date',
    id_cols := ['id'],
    frequency := '1 hour',
    context := 512,
    quantiles := [0.10, 0.25, 0.50, 0.75, 0.90]
);

SELECT * FROM forecast_result ORDER BY id, date;
