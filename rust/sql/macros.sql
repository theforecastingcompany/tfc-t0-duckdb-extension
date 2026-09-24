CREATE OR REPLACE MACRO __tfc_levels() AS __TFC_LEVELS__;

-- Only identifier/literal-quoted names enter generated SQL. The source is still
-- resolved by query_table; column mappings are names, never SQL expressions.
-- The Rust entrypoint registers through a separate connection. Macros must
-- survive that connection and be replaceable on the next database session's LOAD.
CREATE OR REPLACE MACRO __tfc_quote_identifier(name) AS
    CASE WHEN name IS NULL OR name = '' THEN error('column names in id_cols must be nonempty')
         ELSE '"' || replace(name, '"', '""') || '"' END;
-- Quantiles change the output schema, so requests are validated at bind time.
CREATE OR REPLACE MACRO __tfc_quantile_columns(requested) AS
    CASE
        WHEN requested IS NULL THEN error('quantiles must not be NULL; use [] for point-only output')
        WHEN len(list_filter(requested::DOUBLE[], lambda q: q IS NULL OR NOT list_contains(__tfc_levels(), q))) > 0
            THEN error('quantiles must contain supported non-NULL probabilities; see tfc_models().quantile_levels')
        WHEN len(list_distinct(requested::DOUBLE[])) <> len(requested)
            THEN error('quantiles must not contain duplicate probabilities')
        ELSE list_transform(list_sort(requested::DOUBLE[]),
            lambda q: 'p' || lpad(((q * 100)::INTEGER)::VARCHAR, 2, '0'))
    END;

-- Select only requested columns from the enclosing macro's output CTE.
-- Quote every ID; all remaining column names come from fixed internal lists.
CREATE OR REPLACE MACRO __tfc_output_sql(id_cols, fields, requested) AS
    'SELECT ' || array_to_string(list_transform(
        list_concat(id_cols::VARCHAR[], fields, __tfc_quantile_columns(requested), ['model', 'revision']),
        lambda c: __tfc_quote_identifier(c)), ', ') || ' FROM __tfc_output';

CREATE OR REPLACE MACRO __tfc_input_sql(source_table, target_col, timestamp_col, id_cols, covariate_cols := []) AS
    CASE
        WHEN source_table IS NULL OR source_table = ''
            THEN error('source_table must be a nonempty table name')
        WHEN target_col IS NULL OR target_col = '' OR timestamp_col IS NULL OR timestamp_col = ''
            THEN error('target_col and timestamp_col must be nonempty column names')
        WHEN id_cols IS NULL OR len(list_filter(id_cols::VARCHAR[], lambda c: c IS NULL OR c = '')) > 0
            THEN error('id_cols must be a list of nonempty column names')
        WHEN len(list_distinct(list_transform(id_cols::VARCHAR[], lambda c: lower(c)))) <> len(id_cols)
            THEN error('id_cols must not contain duplicate column names')
        WHEN lower(target_col) = lower(timestamp_col)
            OR list_contains(list_transform(id_cols::VARCHAR[], lambda c: lower(c)), lower(target_col))
            OR list_contains(list_transform(id_cols::VARCHAR[], lambda c: lower(c)), lower(timestamp_col))
            THEN error('target_col, timestamp_col and id_cols must name distinct columns')
        WHEN len(list_filter(id_cols::VARCHAR[], lambda c: starts_with(lower(c), '__tfc_') OR lower(c) IN
            ('date', 'cutoff', 'prediction', 'model', 'revision', 'history_values', 'origin', 'mae', 'mse', 'rmse', 'mape', 'smape', 'mase', 'n',
             'p01', 'p05', 'p10', 'p15', 'p20', 'p25', 'p30', 'p35', 'p40', 'p45', 'p50',
             'p55', 'p60', 'p65', 'p70', 'p75', 'p80', 'p85', 'p90', 'p95', 'p99'))) > 0
            THEN error('id_cols conflict with reserved forecast columns; alias them in a view')
        ELSE 'SELECT '
            || CASE WHEN len(id_cols) = 0 THEN '' ELSE
                array_to_string(list_transform(id_cols::VARCHAR[], lambda c: __tfc_quote_identifier(c)), ', ') || ', '
               END
            || __tfc_quote_identifier(timestamp_col) || ' AS __tfc_date, '
            || __tfc_quote_identifier(target_col) || ' AS __tfc_target, true AS __tfc_present, ['
            || array_to_string(list_transform(covariate_cols::VARCHAR[], lambda c: __tfc_quote_identifier(c) || '::DOUBLE'), ', ')
            || ']::DOUBLE[] AS __tfc_covariates'
            || ' FROM query_table(' || chr(39) || replace(source_table, chr(39), chr(39) || chr(39)) || chr(39) || ')'
    END;

-- Positive DuckDB intervals: minutes/hours/days/weeks/months/quarters/years.
CREATE OR REPLACE MACRO __tfc_frequency(value) AS
    CASE WHEN try_cast(value AS INTERVAL) IS NULL
           OR try_cast(value AS INTERVAL) <= INTERVAL '0 seconds'
           OR datepart('year', try_cast(value AS INTERVAL)) < 0 OR datepart('month', try_cast(value AS INTERVAL)) < 0
           OR datepart('day', try_cast(value AS INTERVAL)) < 0 OR datepart('hour', try_cast(value AS INTERVAL)) < 0
           OR datepart('minute', try_cast(value AS INTERVAL)) < 0 OR datepart('microsecond', try_cast(value AS INTERVAL)) < 0
        THEN error('frequency must be a positive interval without negative components')
        ELSE try_cast(value AS INTERVAL) END;

-- Always advance from the original anchor, avoiding repeated month-end drift.
CREATE OR REPLACE MACRO __tfc_time_at(anchor, step, frequency) AS
    cast_to_type(anchor + step::BIGINT * __tfc_frequency(frequency), anchor);

CREATE OR REPLACE MACRO __tfc_context(value) AS
    CASE WHEN value IS NULL OR value < 1 OR value > __TFC_MAX_CONTEXT__ OR value <> floor(value)
        THEN error('context must be an integer in 1..__TFC_MAX_CONTEXT__') ELSE value::BIGINT END;
CREATE OR REPLACE MACRO __tfc_horizon(value) AS
    CASE WHEN value IS NULL OR value < 1 OR value > 1024 OR value <> floor(value)
        THEN error('horizon must be an integer in 1..1024') ELSE value::BIGINT END;

CREATE OR REPLACE MACRO __tfc_series(source_table, target_col, timestamp_col, id_cols, frequency, check_grid, covariate_cols := []) AS TABLE
WITH source AS MATERIALIZED (
    SELECT * FROM query(__tfc_input_sql(source_table, target_col, timestamp_col, id_cols, covariate_cols))
)
SELECT
    source.* EXCLUDE (__tfc_date, __tfc_target, __tfc_present, __tfc_covariates),
    CASE
        WHEN count(__tfc_present) = 0 THEN error('history or actuals must not be empty')
        WHEN count(__tfc_date) <> count(*) OR count(__tfc_target) <> count(*)
            THEN error('date and target must not contain NULL')
        WHEN first(typeof(__tfc_date)) NOT IN ('DATE', 'TIMESTAMP', 'TIMESTAMP WITH TIME ZONE')
            THEN error('timestamp_col must have type DATE, TIMESTAMP or TIMESTAMPTZ')
        WHEN NOT bool_and(isfinite(__tfc_date)) OR NOT bool_and(isfinite(__tfc_target::DOUBLE))
            THEN error('date and target must be finite')
        WHEN count(DISTINCT __tfc_date) <> count(*)
            THEN error('duplicate dates are not supported')
        WHEN first(typeof(__tfc_date)) = 'DATE' AND
            ((datepart('hour', __tfc_frequency(frequency)) * 3600000000
              + datepart('minute', __tfc_frequency(frequency)) * 60000000
              + datepart('microsecond', __tfc_frequency(frequency))) % 86400000000 <> 0)
            THEN error('subdaily frequency requires TIMESTAMP or TIMESTAMPTZ')
        WHEN check_grid AND list(__tfc_date ORDER BY __tfc_date) <>
            list_transform(range(count(*)), lambda i: __tfc_time_at(min(__tfc_date), i, frequency))
            THEN error('history must have a complete regular grid at the requested frequency')
        ELSE list(__tfc_target::DOUBLE ORDER BY __tfc_date)
    END AS history_values,
    CASE WHEN list_count(flatten(list(__tfc_covariates))) <> count(*) * len(covariate_cols)
        OR len(list_filter(flatten(list(__tfc_covariates)), lambda v: NOT isfinite(v))) > 0
        THEN error('covariates must contain only finite, non-null values')
        ELSE flatten(list(__tfc_covariates ORDER BY __tfc_date)) END AS __tfc_cov_values,
    min(__tfc_date) AS origin,
    max(__tfc_date) AS cutoff,
    list(__tfc_date ORDER BY __tfc_date) AS __tfc_dates,
    count(*) AS __tfc_length
FROM (SELECT 1) AS empty_guard LEFT JOIN source ON true
GROUP BY ALL;

CREATE OR REPLACE MACRO __tfc_history(source_table, target_col, timestamp_col, id_cols, frequency, context, covariate_cols := []) AS TABLE
SELECT * REPLACE (
    list_slice(history_values, -__tfc_context(context), -1) AS history_values,
    list_slice(__tfc_cov_values, -__tfc_context(context) * len(covariate_cols), -1) AS __tfc_cov_values
)
FROM __tfc_series(source_table, target_col, timestamp_col, id_cols, frequency, true, covariate_cols);

-- Retain the original daily helper; the forecasting wrapper now uses __tfc_history.
CREATE OR REPLACE MACRO tfc_daily_history(source_table, target_col := 'target', timestamp_col := 'date', id_cols := []) AS TABLE
SELECT * EXCLUDE (__tfc_dates, __tfc_length, __tfc_cov_values) REPLACE (history_values::FLOAT[] AS history_values)
FROM __tfc_history(source_table, target_col, timestamp_col, id_cols, '1 day', __TFC_MAX_CONTEXT__);

-- Keep independent series aligned, including NULL IDs, without stringifying them.
CREATE OR REPLACE MACRO __tfc_id_join(left_alias, right_alias, id_cols) AS
    CASE WHEN len(id_cols) = 0 THEN 'true' ELSE
        array_to_string(list_transform(id_cols::VARCHAR[], lambda c:
            'CASE WHEN typeof(' || left_alias || '.' || __tfc_quote_identifier(c) || ') <> typeof('
            || right_alias || '.' || __tfc_quote_identifier(c) || ') THEN error('
            || chr(39) || 'history and future/actuals must use the same ID types' || chr(39)
            || ') ELSE ' || left_alias || '.' || __tfc_quote_identifier(c) || ' IS NOT DISTINCT FROM '
            || right_alias || '.' || __tfc_quote_identifier(c) || ' END'), ' AND ') END;

CREATE OR REPLACE MACRO __tfc_batch_size(value) AS
    CASE WHEN value IS NULL OR value < 1 OR NOT isfinite(value) OR value <> floor(value)
        OR try_cast(value AS BIGINT) IS NULL
        THEN error('batch_size must be a positive BIGINT integer') ELSE value::BIGINT END;

-- Convenience for one series; the native function always receives an explicit batch.
CREATE OR REPLACE MACRO tfc_forecast_quantiles(history, horizon, covariate_history := []::FLOAT[], covariate_future := []::FLOAT[]) AS
    tfc_forecast_batch([struct_pack(history := history::FLOAT[],
        covariate_history := covariate_history::FLOAT[],
        covariate_future := covariate_future::FLOAT[])], horizon::BIGINT)[1];

-- SQL owns batch membership. Keep opaque row IDs beside requests so typed and
-- NULL user IDs never cross the native boundary or affect result alignment.
CREATE OR REPLACE MACRO __tfc_predict(source_table, batch_size) AS TABLE
WITH source AS MATERIALIZED (
    SELECT *, row_number() OVER () AS __tfc_row_id FROM query_table(source_table)
), numbered AS (
    SELECT *, (row_number() OVER (
        PARTITION BY len(history_values), __tfc_requested_horizon ORDER BY __tfc_row_id
    ) - 1) // __tfc_batch_size(batch_size) AS __tfc_batch_id
    FROM source
), packed AS (
    SELECT len(history_values) AS __tfc_context_length, __tfc_requested_horizon, __tfc_batch_id,
        list(__tfc_row_id ORDER BY __tfc_row_id) AS __tfc_rows,
        list(struct_pack(history := history_values::FLOAT[],
            covariate_history := __tfc_cov_values::FLOAT[],
            covariate_future := __tfc_cov_future::FLOAT[])
            ORDER BY __tfc_row_id) AS __tfc_requests
    FROM numbered GROUP BY __tfc_context_length, __tfc_requested_horizon, __tfc_batch_id
), predicted AS MATERIALIZED (
    SELECT __tfc_rows, tfc_forecast_batch(__tfc_requests, __tfc_requested_horizon) AS __tfc_results
    FROM packed
), expanded AS (
    SELECT unnest(__tfc_rows) AS __tfc_row_id, unnest(__tfc_results) AS __tfc_quantiles
    FROM predicted
)
SELECT source.* EXCLUDE (__tfc_row_id), expanded.__tfc_quantiles
FROM source JOIN expanded USING (__tfc_row_id);

-- Both public wrappers supply __tfc_inference_input with a per-series horizon.
CREATE OR REPLACE MACRO __tfc_covariate_join(future_table, covariate_cols, target_col, timestamp_col, id_cols, frequency) AS
    CASE
        WHEN covariate_cols IS NULL OR len(list_filter(covariate_cols::VARCHAR[], lambda c: c IS NULL OR c = '')) > 0
            THEN error('covariate_cols must contain nonempty column names')
        WHEN len(list_distinct(list_transform(covariate_cols::VARCHAR[], lambda c: lower(c)))) <> len(covariate_cols)
            THEN error('covariate_cols must contain distinct column names')
        WHEN len(list_filter(covariate_cols::VARCHAR[], lambda c:
            list_contains(list_transform(list_concat(id_cols::VARCHAR[], [target_col, timestamp_col]), lambda n: lower(n)), lower(c)))) > 0
            THEN error('covariate_cols must be distinct from target, timestamp and ID columns')
        WHEN (len(covariate_cols) = 0) <> (future_table IS NULL)
            THEN error('future_table and nonempty covariate_cols must be supplied together')
        WHEN len(covariate_cols) = 0 THEN
            'SELECT h.*, []::FLOAT[] AS __tfc_cov_future FROM __tfc_inference_input h'
        ELSE
            'SELECT h.*, CASE '
            || 'WHEN h.history_values IS NULL OR f.history_values IS NULL THEN error(''history and future covariates must contain exactly the same series IDs'') '
            || 'WHEN typeof(h.origin) <> typeof(f.__tfc_dates[1]) THEN error(''history and future covariates must use the same timestamp type'') '
            || 'WHEN len(f.__tfc_dates) < h.__tfc_requested_horizon THEN error(''future covariates must cover every requested forecast step'') '
            || 'WHEN f.__tfc_dates <> list_transform(range(len(f.__tfc_dates)), lambda i: __tfc_time_at(h.origin, h.__tfc_length + i, '
            || chr(39) || replace(frequency::VARCHAR, chr(39), chr(39) || chr(39)) || chr(39)
            || ')) THEN error(''future covariates must start immediately after history and follow its frequency grid'') '
            || 'ELSE list_slice(f.__tfc_cov_values, 1, h.__tfc_requested_horizon * ' || len(covariate_cols)::VARCHAR
            || ')::FLOAT[] END AS __tfc_cov_future FROM __tfc_inference_input h FULL OUTER JOIN '
            || '__tfc_series(' || chr(39) || replace(future_table::VARCHAR, chr(39), chr(39) || chr(39)) || chr(39)
            || ', ' || chr(39) || replace(covariate_cols[1]::VARCHAR, chr(39), chr(39) || chr(39)) || chr(39)
            || ', ' || chr(39) || replace(timestamp_col, chr(39), chr(39) || chr(39)) || chr(39)
            || ', [' || array_to_string(list_transform(id_cols::VARCHAR[], lambda c: chr(39) || replace(c, chr(39), chr(39) || chr(39)) || chr(39)), ',') || ']'
            || ', ' || chr(39) || replace(frequency::VARCHAR, chr(39), chr(39) || chr(39)) || chr(39)
            || ', false, [' || array_to_string(list_transform(covariate_cols::VARCHAR[], lambda c: chr(39) || replace(c, chr(39), chr(39) || chr(39)) || chr(39)), ',')
            || ']) f ON ' || __tfc_id_join('h', 'f', id_cols)
    END;

CREATE OR REPLACE MACRO tfc_forecast(source_table, horizon, target_col := 'target', timestamp_col := 'date', id_cols := [],
                         quantiles := [0.1, 0.5, 0.9], frequency := '1 day', context := __TFC_MAX_CONTEXT__,
                         covariate_cols := [], future_table := NULL, batch_size := 8) AS TABLE
WITH __tfc_inference_input AS MATERIALIZED (
    SELECT *, __tfc_horizon(horizon) AS __tfc_requested_horizon
    FROM __tfc_history(source_table, target_col, timestamp_col, id_cols, frequency, context, covariate_cols)
), prepared AS MATERIALIZED (
    SELECT * FROM query(__tfc_covariate_join(future_table, covariate_cols, target_col, timestamp_col, id_cols, frequency))
), predictions AS MATERIALIZED (
    SELECT * EXCLUDE (history_values, __tfc_dates, __tfc_cov_values, __tfc_cov_future, __tfc_requested_horizon)
    FROM __tfc_predict('prepared', batch_size)
), __tfc_output AS (
SELECT
    predictions.* EXCLUDE (origin, cutoff, __tfc_length, __tfc_quantiles),
    __tfc_time_at(origin, __tfc_length + __tfc_step - 1, frequency) AS date,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.5)] AS prediction,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.01)] AS p01,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.05)] AS p05,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.1)] AS p10,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.15)] AS p15,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.2)] AS p20,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.25)] AS p25,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.3)] AS p30,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.35)] AS p35,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.4)] AS p40,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.45)] AS p45,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.5)] AS p50,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.55)] AS p55,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.6)] AS p60,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.65)] AS p65,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.7)] AS p70,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.75)] AS p75,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.8)] AS p80,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.85)] AS p85,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.9)] AS p90,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.95)] AS p95,
    __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.99)] AS p99,
    '__TFC_MODEL_ID__' AS model,
    '__TFC_MODEL_REVISION__' AS revision
FROM predictions, range(1, __tfc_horizon(horizon) + 1) AS future(__tfc_step)
)
SELECT * FROM query(__tfc_output_sql(id_cols, ['date', 'prediction'], quantiles));

CREATE OR REPLACE MACRO tfc_models() AS TABLE
SELECT '__TFC_MODEL_ID__' AS model,
       '__TFC_MODEL_REVISION__' AS revision,
       'CPU' AS device, 'median' AS point_estimate,
       __tfc_levels() AS quantile_levels, __TFC_MAX_CONTEXT__ AS max_context;

-- FULL JOIN prevents unmatched series from silently disappearing. NULL IDs match.
CREATE OR REPLACE MACRO __tfc_evaluation_join(id_cols) AS
    'SELECT h.*, a.history_values AS __tfc_actual_values, a.__tfc_dates AS __tfc_actual_dates '
    || 'FROM __tfc_history_input h FULL OUTER JOIN __tfc_actual_input a ON '
    || __tfc_id_join('h', 'a', id_cols);

CREATE OR REPLACE MACRO tfc_evaluate(source_table, actuals_table, horizon := 1024,
    target_col := 'target', timestamp_col := 'date', id_cols := [],
    frequency := '1 day', context := __TFC_MAX_CONTEXT__, seasonal_period := 1,
    covariate_cols := [], future_table := NULL, batch_size := 8) AS TABLE
WITH __tfc_history_input AS MATERIALIZED (
    SELECT * FROM __tfc_history(source_table, target_col, timestamp_col, id_cols, frequency, context, covariate_cols)
), __tfc_actual_input AS MATERIALIZED (
    SELECT * FROM __tfc_series(actuals_table, target_col, timestamp_col, id_cols, frequency, false)
), aligned AS MATERIALIZED (
    SELECT * EXCLUDE (__tfc_dates, __tfc_actual_dates, __tfc_actual_values),
        CASE
            WHEN history_values IS NULL OR __tfc_actual_values IS NULL
                THEN error('history and actuals must contain exactly the same series IDs')
            WHEN typeof(origin) <> typeof(__tfc_actual_dates[1])
                THEN error('history and actuals must use the same timestamp type')
            WHEN __tfc_actual_dates <> list_transform(range(len(__tfc_actual_dates)),
                lambda i: __tfc_time_at(origin, __tfc_length + i, frequency))
                THEN error('actuals must start immediately after history and follow its frequency grid')
            ELSE list_slice(__tfc_actual_values, 1, __tfc_horizon(horizon))
        END AS __tfc_actuals,
        CASE WHEN seasonal_period IS NULL OR seasonal_period < 1 OR seasonal_period <> floor(seasonal_period)
            THEN error('seasonal_period must be a positive integer')
            ELSE seasonal_period::BIGINT END AS __tfc_lag
    FROM query(__tfc_evaluation_join(id_cols))
) , __tfc_inference_input AS MATERIALIZED (
    SELECT *, len(__tfc_actuals) AS __tfc_requested_horizon FROM aligned
), prepared AS MATERIALIZED (
    SELECT * FROM query(__tfc_covariate_join(future_table, covariate_cols, target_col, timestamp_col, id_cols, frequency))
), predictions AS MATERIALIZED (
    SELECT * EXCLUDE (origin, cutoff, __tfc_length, history_values, __tfc_lag, __tfc_cov_values, __tfc_cov_future, __tfc_requested_horizon),
        list_avg(list_transform(range(__tfc_lag + 1, len(history_values) + 1),
            lambda i: abs(history_values[i]::DOUBLE - history_values[i - __tfc_lag]::DOUBLE))) AS __tfc_scale
    FROM __tfc_predict('prepared', batch_size)
), errors AS (
    SELECT predictions.* EXCLUDE (__tfc_actuals, __tfc_quantiles),
        __tfc_actuals[__tfc_step]::DOUBLE AS __tfc_actual,
        __tfc_quantiles[(__tfc_step - 1) * len(__tfc_levels()) + list_position(__tfc_levels(), 0.5)]::DOUBLE AS __tfc_prediction
    FROM predictions, range(1, len(__tfc_actuals) + 1) AS steps(__tfc_step)
)
SELECT * EXCLUDE (__tfc_actual, __tfc_prediction, __tfc_scale),
    count(*) AS n,
    avg(abs(__tfc_actual - __tfc_prediction)) AS mae,
    avg(pow(__tfc_actual - __tfc_prediction, 2)) AS mse,
    sqrt(avg(pow(__tfc_actual - __tfc_prediction, 2))) AS rmse,
    100 * avg(abs(__tfc_actual - __tfc_prediction) / nullif(abs(__tfc_actual), 0)) AS mape,
    200 * avg(CASE WHEN abs(__tfc_actual) + abs(__tfc_prediction) = 0 THEN 0
        ELSE abs(__tfc_actual - __tfc_prediction) / (abs(__tfc_actual) + abs(__tfc_prediction)) END) AS smape,
    avg(abs(__tfc_actual - __tfc_prediction)) / nullif(first(__tfc_scale), 0) AS mase,
    '__TFC_MODEL_ID__' AS model,
    '__TFC_MODEL_REVISION__' AS revision
FROM errors GROUP BY ALL;
