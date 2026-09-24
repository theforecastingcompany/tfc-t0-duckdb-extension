-- Observe the batches selected by SQL without requiring an ONNX model.
-- Each synthetic prediction carries its request identity and actual batch size.
CREATE OR REPLACE MACRO tfc_forecast_batch(requests,horizon) AS
    CASE WHEN len(list_distinct(list_transform(requests,lambda r:len(r.history))))<>1
        THEN error('SQL combined incompatible contexts')
    ELSE list_transform(requests, lambda r:
        [r.history[1],len(requests),horizon]::FLOAT[]) END;
CREATE TABLE batch_requests AS
SELECT i::INTEGER AS token, CASE WHEN i%7=0 THEN NULL ELSE (i%3)::SMALLINT END AS store,
    ('item-'||(i%5))::VARCHAR AS label,
    CASE WHEN i%2=0 THEN [i,i+1] ELSE [i,i+1,i+2] END::DOUBLE[] AS history_values,
    (i%3+1)::BIGINT AS __tfc_requested_horizon,
    []::DOUBLE[] AS __tfc_cov_values, []::FLOAT[] AS __tfc_cov_future
FROM range(5003)t(i) ORDER BY i DESC;

SET threads=1;
CREATE TABLE batches_single_thread AS SELECT * FROM __tfc_predict('batch_requests',64);
SELECT assert_true(count(*)=5003 AND bool_and(__tfc_quantiles[1]=token
    AND __tfc_quantiles[2] BETWEEN 1 AND 64
    AND __tfc_quantiles[3]=__tfc_requested_horizon),
    'SQL batch size, mixed horizons and output identity') FROM batches_single_thread;
SELECT assert_true(abs(sum(1.0/__tfc_quantiles[2])-84)<1e-8,
    'SQL forms exactly 84 model batches across six compatible shape groups') FROM batches_single_thread;
SELECT assert_true(NOT EXISTS (
    (SELECT * EXCLUDE(__tfc_quantiles) FROM batches_single_thread EXCEPT ALL SELECT * FROM batch_requests)
    UNION ALL (SELECT * FROM batch_requests EXCEPT ALL SELECT * EXCLUDE(__tfc_quantiles) FROM batches_single_thread)
), 'typed composite and NULL metadata preserved through packing');

SET threads=4;
SELECT assert_true(count(*)=5003 AND bool_and(__tfc_quantiles[1]=token
    AND __tfc_quantiles[2] BETWEEN 1 AND 64 AND __tfc_quantiles[3]=__tfc_requested_horizon),
    'parallel execution preserves request alignment') FROM __tfc_predict('batch_requests',64);
SELECT assert_true(count(*)=5003 AND bool_and(__tfc_quantiles[1]=token AND __tfc_quantiles[2]=1),
    'more than one execution chunk of explicit batches') FROM __tfc_predict('batch_requests',1);

CREATE TABLE homogeneous_requests AS SELECT * REPLACE (
    [token]::DOUBLE[] AS history_values, 1::BIGINT AS __tfc_requested_horizon) FROM batch_requests;
SELECT assert_true(count(*)=5003 AND bool_and(__tfc_quantiles[1]=token AND __tfc_quantiles[2]=5003),
    'one explicit batch can exceed DuckDB chunk size') FROM __tfc_predict('homogeneous_requests',10000);
CREATE TABLE empty_requests AS SELECT * FROM batch_requests WHERE false;
SELECT assert_true(count(*)=0,'empty prepared input produces no batches') FROM __tfc_predict('empty_requests',64);

-- Both public wrappers must actually route through SQL-built batches.
CREATE TABLE batching_history AS SELECT s::SMALLINT AS store,
    DATE '2024-01-01'+t::INTEGER AS date, s::DOUBLE AS target
FROM range(70)series(s), range(3)steps(t);
CREATE TABLE batching_actuals AS SELECT s::SMALLINT AS store,
    DATE '2024-01-04'+t::INTEGER AS date, s::DOUBLE AS target
FROM range(70)series(s), range(3)steps(t) WHERE t<=s%3;
CREATE OR REPLACE MACRO tfc_forecast_batch(requests,horizon) AS
    CASE WHEN len(requests)>64 OR len(list_distinct(list_transform(requests,lambda r:len(r.history))))<>1
        THEN error('invalid SQL batch')
    ELSE list_transform(requests, lambda r:
        list_transform(range(horizon*21), lambda q:r.history[1]::FLOAT)) END;
SELECT assert_true(count(*)=140 AND bool_and(prediction=store), 'forecast wrapper batch alignment')
FROM tfc_forecast('batching_history',2,id_cols:=['store'],batch_size:=64);
SELECT assert_true(count(*)=70 AND bool_and(n=store%3+1 AND mae=0 AND mse=0),
    'evaluation batches differing actual horizons and restores series')
FROM tfc_evaluate('batching_history','batching_actuals',id_cols:=['store'],batch_size:=64);
