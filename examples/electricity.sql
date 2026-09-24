-- Run from the repository root after loading the extension; see README.md.
.bail on

.read examples/electricity-forecast.sql
.read examples/electricity-evaluate.sql
.read examples/electricity-covariates.sql

-- The published blog scores belong to beta FP16; other models have different accuracy.
SELECT CASE WHEN (SELECT count(*) FROM history)=512
 AND (SELECT count(*) FROM actuals)=24
 AND (SELECT count(*) FROM future_inputs)=24
 AND (SELECT count(*) FROM forecast_result)=24
 AND (SELECT count(*) FROM forecast_with_covariates)=24
 AND (SELECT model <> 't0-beta-onnx-fp16' OR abs(mae-6.527077754338582)<0.001 FROM baseline_evaluation)
 AND (SELECT model <> 't0-beta-onnx-fp16' OR abs(mae-2.289589246114095)<0.001 FROM evaluation_result)
 THEN true ELSE error('Local ONNX demo verification failed') END AS verified;

COPY (SELECT * FROM history ORDER BY id, date)
TO 'outputs/electricity/example-history.csv' (HEADER);
COPY (
    SELECT f.id, f.date, a.price AS actual,
           f.prediction, f.p10, f.p25, f.p50, f.p75, f.p90,
           c.prediction AS covariate_prediction,
           c.p10 AS covariate_p10, c.p25 AS covariate_p25,
           c.p50 AS covariate_p50, c.p75 AS covariate_p75, c.p90 AS covariate_p90
    FROM forecast_result f
    JOIN actuals a USING (id, date)
    JOIN forecast_with_covariates c USING (id, date)
    ORDER BY f.id, f.date
) TO 'outputs/electricity/example-forecast.csv' (HEADER);
COPY (
    SELECT 'History only' AS inputs, * FROM baseline_evaluation
    UNION ALL
    SELECT 'Load + renewables forecasts', * FROM evaluation_result
) TO 'outputs/electricity/covariate-comparison.csv' (HEADER);
