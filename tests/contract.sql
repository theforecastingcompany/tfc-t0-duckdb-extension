-- Constant forecasts of 1 make metric expectations independent of the model.
CREATE MACRO assert_true(ok, message) AS CASE WHEN coalesce(ok, false) THEN true ELSE error(message) END;
CREATE TABLE history AS
SELECT DATE '2024-01-01'+i::INT AS date, (1+2*i)::DOUBLE AS target FROM range(3)t(i);
CREATE TABLE actuals AS
SELECT DATE '2024-01-04'+i::INT AS date, (2*i)::DOUBLE AS target FROM range(3)t(i);

SELECT assert_true(n=3 AND abs(mae-5.0/3)<1e-12 AND abs(mse-11.0/3)<1e-12
    AND abs(rmse-sqrt(11.0/3))<1e-12 AND abs(mape-62.5)<1e-12
    AND abs(smape-100*(2+2.0/3+6.0/5)/3)<1e-12 AND abs(mase-5.0/6)<1e-12,
    'independent evaluation arithmetic') FROM tfc_evaluate('history','actuals');
SELECT assert_true(abs(mase-5.0/12)<1e-12, 'seasonal MASE')
FROM tfc_evaluate('history','actuals',seasonal_period:=2);
SELECT assert_true(mase IS NULL, 'insufficient MASE history')
FROM tfc_evaluate('history','actuals',seasonal_period:=3);
SELECT assert_true(n=2, 'evaluation horizon') FROM tfc_evaluate('history','actuals',horizon:=2);

CREATE TABLE long_history AS
SELECT DATE '2000-01-01'+i::INT AS date,i::FLOAT AS target FROM range(8200)t(i);
SELECT assert_true(len(history_values)=8192 AND history_values[1]=8 AND history_values[-1]=8199,
    'latest 8192 observations') FROM tfc_daily_history('long_history');

CREATE TABLE months AS SELECT (DATE '2024-01-31'+i*INTERVAL '1 month')::DATE AS date,
    i::FLOAT AS target FROM range(3)t(i);
SELECT assert_true(date=DATE '2024-04-30' AND typeof(date)='DATE', 'month-end anchor')
FROM tfc_forecast('months',1,frequency:='1 month');
CREATE TABLE hours AS SELECT TIMESTAMP '2024-01-01 10:20:30.123456'+i*INTERVAL '1 hour' AS date,
    i::FLOAT AS target FROM range(3)t(i);
SELECT assert_true(date=TIMESTAMP '2024-01-01 13:20:30.123456' AND typeof(date)='TIMESTAMP', 'timestamp grid')
FROM tfc_forecast('hours',1,frequency:='1 hour');

CREATE OR REPLACE MACRO tfc_forecast_batch(requests,horizon) AS
    list_transform(requests, lambda r: list_transform(range(horizon*21), lambda i: r.history[1]::FLOAT));
CREATE TABLE precise_history AS SELECT DATE '2024-01-01'+i::INT AS date,
    (100000000+i)::DOUBLE AS target FROM range(3)t(i);
CREATE TABLE precise_actuals AS SELECT DATE '2024-01-04' AS date,100000003::DOUBLE AS target;
SELECT assert_true(mae=3 AND mase=3, 'DOUBLE precision retained in metrics')
FROM tfc_evaluate('precise_history','precise_actuals');
CREATE TABLE zeros AS SELECT date,0::FLOAT AS target FROM history;
CREATE TABLE zero_actuals AS SELECT date,0::FLOAT AS target FROM actuals;
SELECT assert_true(mae=0 AND mse=0 AND rmse=0 AND mape IS NULL AND smape=0 AND mase IS NULL,
    'zero denominator handling') FROM tfc_evaluate('zeros','zero_actuals');

CREATE TABLE cov_history AS
SELECT date,target,(row_number() OVER (ORDER BY date)*10)::DOUBLE AS price FROM history ORDER BY date DESC;
CREATE TABLE cov_future AS
SELECT date,40+10*(row_number() OVER (ORDER BY date)-1) AS price FROM actuals ORDER BY date DESC;
CREATE OR REPLACE MACRO tfc_forecast_batch(requests,horizon) AS
    list_transform(requests, lambda r: list_transform(range(horizon*21), lambda i:
        (CASE WHEN len(r.covariate_history)>0 THEN r.covariate_history[1]+r.covariate_future[1] ELSE r.history[1] END)::FLOAT));
SELECT assert_true(count(*)=2 AND bool_and(prediction=60), 'covariates sorted, context trimmed, future selected')
FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_future',context:=2);
SELECT assert_true(count(*)=2 AND bool_and(prediction=60), 'typed interval with covariates')
FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_future',context:=2,frequency:=INTERVAL '1 day');
SELECT assert_true(count(*)=1, 'typed interval without covariates') FROM tfc_forecast('history',1,frequency:=INTERVAL '1 day');
SELECT assert_true(n=2 AND mae=59, 'covariate evaluation uses requested actual slice')
FROM tfc_evaluate('cov_history','actuals',horizon:=2,context:=2,covariate_cols:=['price'],future_table:='cov_future');
CREATE VIEW cov_short AS SELECT * FROM cov_future WHERE date=DATE '2024-01-04';
CREATE VIEW cov_gap AS SELECT date+1 AS date,price FROM cov_future;
CREATE VIEW cov_null AS SELECT date,NULL::DOUBLE AS price FROM cov_future;
CREATE VIEW cov_duplicate AS SELECT * FROM cov_future UNION ALL SELECT * FROM cov_future;
CREATE TABLE cov_group_history AS SELECT 1::SMALLINT AS store,* FROM cov_history;
CREATE TABLE cov_group_future AS SELECT 1::SMALLINT AS store,* FROM cov_future;
CREATE VIEW cov_extra AS SELECT * FROM cov_group_future UNION ALL SELECT 2::SMALLINT AS store,date,price FROM cov_future;
CREATE VIEW cov_wrong_id_type AS SELECT store::VARCHAR AS store,date,price FROM cov_group_future;
CREATE VIEW cov_wrong_date_type AS SELECT date::TIMESTAMP AS date,price FROM cov_future;
