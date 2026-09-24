SET threads=1;
CREATE TEMP MACRO assert_true(ok, message) AS CASE WHEN coalesce(ok, false) THEN true ELSE error(message) END;
CREATE TEMP TABLE sales AS
SELECT store_id, DATE '2024-01-01'+i::INT AS day, (10+coalesce(store_id,3)+i%7)::FLOAT AS sales
FROM (VALUES (1::SMALLINT),(2::SMALLINT),(NULL::SMALLINT))s(store_id),range(14)t(i)
ORDER BY day DESC;
CREATE TEMP TABLE forecasts AS SELECT * FROM tfc_forecast('sales',3,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id']);
SELECT assert_true(count(*)=9 AND count(DISTINCT date)=3 AND min(date)=DATE '2024-01-15'
    AND bool_and(isfinite(prediction) AND prediction=p50 AND p10<=p50 AND p50<=p90),
    'default native forecast shape, dates and quantiles') FROM forecasts;
SELECT assert_true(list(column_name ORDER BY column_index)=
    ['store_id','date','prediction','p10','p50','p90','model','revision'], 'default schema')
FROM duckdb_columns() WHERE table_name='forecasts';
SELECT assert_true(bool_and(typeof(store_id)='SMALLINT'), 'ID type preserved') FROM forecasts;
CREATE TEMP VIEW isolated AS SELECT day AS date,sales AS target FROM sales WHERE store_id IS NULL;
SELECT assert_true(NOT EXISTS (
    (SELECT * EXCLUDE(store_id) FROM forecasts WHERE store_id IS NULL)
    EXCEPT ALL (SELECT * FROM tfc_forecast('isolated',3))), 'NULL group matches isolated inference');

PREPARE selected AS SELECT * FROM tfc_forecast('isolated',3,quantiles:=$1);
EXECUTE selected([0.75,0.25]);
CREATE TEMP TABLE subset AS SELECT * FROM tfc_forecast('isolated',3,quantiles:=[0.75,0.25]);
SELECT assert_true(list(column_name ORDER BY column_index)=
    ['date','prediction','p25','p75','model','revision'], 'selected quantile order')
FROM duckdb_columns() WHERE table_name='subset';
CREATE TEMP TABLE points AS SELECT * FROM tfc_forecast('isolated',3,quantiles:=[]);
SELECT assert_true(list(column_name ORDER BY column_index)=
    ['date','prediction','model','revision'], 'point-only schema')
FROM duckdb_columns() WHERE table_name='points';
SELECT assert_true(len(tfc_forecast_quantiles([1,2,3]::FLOAT[],50))=50*len(quantile_levels), 'rounded horizon truncation') FROM tfc_models();
CREATE TEMP VIEW tail AS SELECT * FROM isolated WHERE date>=DATE '2024-01-07';
SELECT assert_true(NOT EXISTS (
    (SELECT * FROM tfc_forecast('isolated',2,context:=8))
    EXCEPT ALL (SELECT * FROM tfc_forecast('tail',2))), 'context equals explicit tail');

CREATE TEMP TABLE heldout AS SELECT store_id,date AS day,prediction::DOUBLE+2 AS sales FROM forecasts;
SELECT assert_true(count(*)=3 AND bool_and(n=3 AND mae=2 AND mse=4 AND rmse=2),
    'grouped evaluation with mapped columns and NULL IDs')
FROM tfc_evaluate('sales','heldout',target_col:='sales',timestamp_col:='day',id_cols:=['store_id']);
UPDATE heldout SET sales=sales+1;
SELECT assert_true(count(*)=3 AND bool_and(mae=3), 'actuals do not affect forecasts')
FROM tfc_evaluate('sales','heldout',target_col:='sales',timestamp_col:='day',id_cols:=['store_id']);
UPDATE sales SET sales=sales+1000 WHERE store_id=1;
SELECT assert_true(NOT EXISTS (
    (SELECT * FROM tfc_forecast('sales',3,target_col:='sales',timestamp_col:='day',id_cols:=['store_id'])
        WHERE store_id IS DISTINCT FROM 1)
    EXCEPT ALL (SELECT * FROM forecasts WHERE store_id IS DISTINCT FROM 1)), 'independent series');
SELECT assert_true(count(*)=0, 'rolling backtests remain deferred')
FROM duckdb_functions() WHERE function_name LIKE 'tfc_backtest%';
