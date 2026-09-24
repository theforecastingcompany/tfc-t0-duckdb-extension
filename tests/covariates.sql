-- Different context lengths and NULL IDs exercise batching and alignment together.
CREATE TEMP TABLE cov_history AS
SELECT store_id, DATE '2024-01-01'+i::INT AS day,
    (20+10*(i%7=0)::INT+coalesce(store_id,3)+sin(i/5.0))::DOUBLE AS sales,
    (i%7=0)::INT AS promotion, (2+cos(i/5.0))::DOUBLE AS price
FROM (VALUES (1::SMALLINT),(2::SMALLINT),(NULL::SMALLINT))s(store_id),range(97)t(i)
WHERE store_id IS DISTINCT FROM 2 OR i>=20 ORDER BY day DESC;
CREATE TEMP TABLE cov_future AS
SELECT store_id,DATE '2024-04-07'+i::INT AS day,
    ((97+i)%7=0)::INT AS promotion,(2+cos((97+i)/5.0))::DOUBLE AS price
FROM (VALUES (1::SMALLINT),(2::SMALLINT),(NULL::SMALLINT))s(store_id),range(5)t(i)
ORDER BY day DESC;
CREATE TEMP TABLE cov_batch AS SELECT * FROM tfc_forecast('cov_history',5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=8);
CREATE TEMP TABLE cov_single AS SELECT * FROM tfc_forecast('cov_history',5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=1);
SELECT assert_true(count(*)=15 AND bool_and(isfinite(prediction) AND p10<=p50 AND p50<=p90),
    'covariate forecast coverage') FROM cov_batch;
SELECT assert_true(count(*)=15 AND bool_and(abs(b.prediction-s.prediction)<1e-4*(1+abs(s.prediction))
    AND abs(b.p10-s.p10)<1e-4*(1+abs(s.p10)) AND abs(b.p90-s.p90)<1e-4*(1+abs(s.p90))),
    'batched and serial covariate forecasts agree')
FROM cov_batch b JOIN cov_single s ON b.store_id IS NOT DISTINCT FROM s.store_id AND b.date=s.date;
CREATE TEMP VIEW cov_tail AS SELECT * FROM cov_history WHERE day>=DATE '2024-03-31';
CREATE TEMP TABLE cov_trimmed AS SELECT * FROM tfc_forecast('cov_history',5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],context:=7,
    covariate_cols:=['promotion','price'],future_table:='cov_future');
SELECT assert_true(bool_and(abs(t.prediction-f.prediction)<1e-4*(1+abs(t.prediction))),
    'target and covariate context trim together')
FROM cov_trimmed t JOIN tfc_forecast('cov_tail',5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future') f
ON t.store_id IS NOT DISTINCT FROM f.store_id AND t.date=f.date;

CREATE TEMP TABLE cov_actuals AS SELECT store_id,date AS day,prediction::DOUBLE+2 AS sales FROM cov_single;
CREATE TEMP TABLE cov_evaluation AS SELECT * FROM tfc_evaluate('cov_history','cov_actuals',horizon:=5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=1);
SELECT assert_true(count(*)=3 AND bool_and(n=5 AND abs(mae-2)<1e-10 AND abs(mse-4)<1e-10),
    'evaluation uses covariates') FROM cov_evaluation;
UPDATE cov_actuals SET sales=sales+10;
SELECT assert_true(bool_and(abs(mae-12)<1e-10), 'actual targets never enter covariate inference')
FROM tfc_evaluate('cov_history','cov_actuals',horizon:=5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future',batch_size:=1);

UPDATE cov_future SET promotion=100 WHERE store_id=1;
CREATE TEMP TABLE cov_changed AS SELECT * FROM tfc_forecast('cov_history',5,
    target_col:='sales',timestamp_col:='day',id_cols:=['store_id'],
    covariate_cols:=['promotion','price'],future_table:='cov_future');
SELECT assert_true(max(abs(b.prediction-c.prediction))>0.001,
    'future covariates actually affect predictions')
FROM cov_batch b JOIN cov_changed c USING(store_id,date) WHERE b.store_id=1;
SELECT assert_true(bool_and(abs(b.prediction-c.prediction)<1e-4*(1+abs(b.prediction))),
    'another series covariates cannot leak across batch groups')
FROM cov_batch b JOIN cov_changed c ON b.store_id IS NOT DISTINCT FROM c.store_id AND b.date=c.date
WHERE b.store_id IS DISTINCT FROM 1;
