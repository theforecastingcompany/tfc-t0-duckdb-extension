-- The native bridge must preserve row order across mixed request shapes.
CREATE TEMP TABLE requests AS
SELECT i, CASE WHEN i%2=0 THEN [1.,2.,3.] ELSE [10.,20.] END::FLOAT[] AS history,
       (i%3+1)::BIGINT AS horizon
FROM range(9) t(i);
CREATE TEMP TABLE native_results AS
SELECT i, history, horizon, tfc_forecast_quantiles(history,horizon) AS quantiles FROM requests;
SELECT assert_true(count(*)=9 AND bool_and(len(quantiles)=(SELECT len(quantile_levels) FROM tfc_models())*horizon),
    'native primitive preserves each requested horizon') FROM native_results;
SELECT assert_true(bool_and(quantiles=tfc_forecast_batch([struct_pack(history:=history,covariate_history:=[]::FLOAT[],covariate_future:=[]::FLOAT[])],horizon)[1]),
    'mixed native request batches preserve row order') FROM native_results;

-- Multiple target columns are reshaped in SQL, not parsed by Rust.
CREATE TEMP TABLE wide_targets AS
SELECT i::INTEGER AS series_id, DATE '2024-01-01'+d::INTEGER AS date,
       (i+d)::FLOAT AS first_target, (100+i+d)::FLOAT AS second_target
FROM range(2) s(i),range(20) t(d);
CREATE TEMP VIEW long_targets AS
UNPIVOT wide_targets ON first_target, second_target INTO NAME target_name VALUE target;
CREATE TEMP TABLE target_forecasts AS
SELECT * FROM tfc_forecast('long_targets',3,id_cols:=['series_id','target_name']);
SELECT assert_true(count(*)=12 AND count(DISTINCT (series_id,target_name))=4,
    'each reshaped target receives its own forecast') FROM target_forecasts;
CREATE TEMP VIEW first_target_history AS SELECT series_id,date,first_target AS target FROM wide_targets;
SELECT assert_true(NOT EXISTS (
    (SELECT * EXCLUDE(target_name) FROM target_forecasts WHERE target_name='first_target')
    EXCEPT ALL (SELECT * FROM tfc_forecast('first_target_history',3,id_cols:=['series_id']))),
    'multi-target SQL reshape matches separate target calls');
SELECT assert_true(bool_and(f.model=m.model AND f.revision=m.revision),
    'forecast metadata matches selected model configuration') FROM target_forecasts f, tfc_models() m;

-- A native input row is one complete batch; the old 32-series cap is gone.
CREATE TEMP TABLE packed_requests AS
SELECT list(struct_pack(history:=[i+1.,i+2.,i+3.]::FLOAT[],
    covariate_history:=[]::FLOAT[],covariate_future:=[]::FLOAT[]) ORDER BY i) AS requests
FROM range(33)t(i);
CREATE TEMP TABLE packed_results AS SELECT tfc_forecast_batch(requests,2) AS results FROM packed_requests;
SELECT assert_true(len(results)=33 AND list_min(list_transform(results,lambda r:len(r)))=
    2*(SELECT len(quantile_levels) FROM tfc_models()), 'explicit native batch above 32 series') FROM packed_results;
SELECT assert_true(list_max(list_transform(list_zip(results[17],tfc_forecast_quantiles([17,18,19]::FLOAT[],2)),
    lambda pair:abs(pair[1]-pair[2])/(1+abs(pair[2]))))<1e-4,
    'explicit batch retains series positions and isolated prediction parity') FROM packed_results;

-- Context lengths and per-series evaluation horizons can differ between batches.
CREATE TEMP VIEW cov_variable_actuals AS SELECT * FROM cov_actuals WHERE store_id IS DISTINCT FROM 2 OR day<DATE '2024-04-10';
CREATE TEMP TABLE variable_batched AS SELECT * FROM tfc_evaluate('cov_history','cov_variable_actuals',
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=64);
CREATE TEMP TABLE variable_serial AS SELECT * FROM tfc_evaluate('cov_history','cov_variable_actuals',
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=1);
SELECT assert_true(count(*)=3 AND bool_and(b.n=s.n AND abs(b.mae-s.mae)<1e-4*(1+abs(s.mae))
    AND abs(b.mse-s.mse)<1e-4*(1+abs(s.mse))), 'batched evaluation matches serial for mixed horizons')
FROM variable_batched b JOIN variable_serial s ON b.store_id IS NOT DISTINCT FROM s.store_id;

-- One callback's nested children can exceed 2048 even with small model batches.
CREATE TEMP TABLE many_native_batches AS
SELECT i, tfc_forecast_batch(list_transform(range(33),lambda j:struct_pack(
    history:=[1+(i*33+j)%7]::FLOAT[],covariate_history:=[]::FLOAT[],covariate_future:=[]::FLOAT[])),1) AS results
FROM range(65)t(i);
CREATE TEMP TABLE isolated_values AS
SELECT i,tfc_forecast_quantiles([i]::FLOAT[],1) AS quantiles FROM range(1,8)t(i);
SELECT assert_true(count(*)=2145 AND bool_and(list_max(list_transform(list_zip(m.results[j],s.quantiles),
    lambda pair:abs(pair[1]-pair[2])/(1+abs(pair[2]))))<1e-4),
    'nested input and output offsets beyond 2048 preserve every series')
FROM many_native_batches m, range(1,34)t(j), isolated_values s WHERE s.i=1+(m.i*33+j-1)%7;

-- DuckDB's scalar NULL propagation applies to the top-level arguments.
SELECT assert_true(tfc_forecast_batch(NULL::STRUCT(history FLOAT[],covariate_history FLOAT[],covariate_future FLOAT[])[],1) IS NULL,
    'NULL native batch propagates');
SELECT assert_true(tfc_forecast_quantiles([1]::FLOAT[],NULL::BIGINT) IS NULL,
    'NULL native horizon propagates');
