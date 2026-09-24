# T0 forecasting in DuckDB

Forecast and evaluate time series with SQL, using
[T0 ONNX models](https://huggingface.co/theforecastingcompany)
locally in the DuckDB process. Supports multiple series, known-future covariates,
selected quantiles, and batched CPU inference. No Python environment or prediction
server is needed.

This walkthrough uses **German electricity prices** throughout: 512 hours of
history and a 24-hour forecast for 12 December 2017. First forecast from prices
alone, then evaluate against held-out observations, and finally add day-ahead
load and renewables forecasts.

- [Quick start](#quick-start)
- [Forecast](#forecast-with-tfc_forecast)
- [Evaluate](#evaluate-with-tfc_evaluate)
- [Covariates](#add-known-future-covariates)
- [Batching and timing](#batch-inference-and-timing)
- [Parameters](#parameter-reference)
- [License](#license)

## Quick start

You need the **DuckDB 1.5.4 CLI**, **Rust/Cargo 1.89+** managed by `rustup`,
`just`, `curl`, a C/C++ build toolchain, and either `sha256sum` or `shasum`.
Allow disk space for the Rust build, the selected model and native runtime.
The extension is a locally built, unsigned preview, not yet in DuckDB Community.
We plan to submit it to DuckDB Community Extensions soon.

> **First download:** the default model is publicly downloadable from Hugging
> Face. No account, access approval, or token is required. The first `LOAD` needs
> an internet connection to download the model and native runtime.

Clone the repository, build the extension, download the data and open DuckDB:

```bash
git clone https://github.com/theforecastingcompany/tfc-t0-duckdb-extension.git
cd tfc-t0-duckdb-extension
rustup toolchain install 1.89.0 --profile minimal --component clippy,rustfmt
export RUSTUP_TOOLCHAIN=1.89.0
just fetch              # Fetch locked Rust dependencies on the first build
just release            # Build the optimized local extension
just electricity-data   # Download and checksum the two public Parquet files

export TFC_CACHE_DIR="$PWD/.cache/tfc-forecast"
duckdb -unsigned -cmd "LOAD '$PWD/build/tfc_forecast.duckdb_extension';"
```

The default is **T0-alpha INT8**, the smallest download. To choose another model,
set `TFC_MODEL` in the terminal **before starting DuckDB and running `LOAD`**:

```bash
export TFC_MODEL=t0-beta-onnx-fp16
duckdb -unsigned -cmd "LOAD '$PWD/build/tfc_forecast.duckdb_extension';"
```

| `TFC_MODEL` | Download | Maximum context | Native quantiles |
| --- | ---: | ---: | ---: |
| `t0-alpha-onnx-int8` (default) | 107 MB | 4,096 | 5 |
| `t0-alpha-onnx-fp16` | 208 MB | 4,096 | 5 |
| `t0-beta-onnx-int8` | 269 MB | 8,192 | 21 |
| `t0-beta-onnx-fp16` | 512 MB | 8,192 | 21 |

Only the selected model is downloaded. Restart DuckDB to switch models; use
`unset TFC_MODEL` to return to the default. `SELECT * FROM tfc_models();` shows
the loaded model, revision, quantiles and context limit. The existing
`TFC_MODEL_CONFIG` custom-manifest override takes precedence over `TFC_MODEL`.
The blog's published results use `t0-beta-onnx-fp16`; select that model to
reproduce its scores with `just electricity`.

`just release` builds the extension, and the command above loads it into DuckDB.
Repeat `LOAD` in each new session; no separate `INSTALL` is needed for this preview.
The Rust selection applies to this shell. In a new shell, repeat
`export RUSTUP_TOOLCHAIN=1.89.0` if your default Rust is older.

The first `LOAD` creates `TFC_CACHE_DIR` automatically and downloads the
[selected model](rust/core/src/models.json) (107 MB for the default) and ONNX Runtime 1.29.0.
**This can take several minutes without visible progress**; wait for DuckDB's
`D` prompt. Keep the same cache path to avoid downloading again. Both binaries
are checksum-verified on each load.

Forecasting runs locally on CPU. INT8 and FP16 describe stored weights; the
published graphs use FP32 computation. Runtime memory also includes working
buffers and may exceed the model file size.
The first forecast initializes the model session; later calls reuse it.
The blog's warm timings exclude downloads and startup.

To deliberately try a new empty cache, exit DuckDB with `.quit`, then run in the
same terminal before launching DuckDB again:

```bash
mkdir -p .cache
export TFC_CACHE_DIR="$(mktemp -d "$PWD/.cache/walkthrough.XXXXXX")"
duckdb -unsigned -cmd "LOAD '$PWD/build/tfc_forecast.duckdb_extension';"
```

This preserves your previous cache. Keep the newly selected path for offline
reuse; creating another empty directory would trigger another download.

Run the Forecast, Evaluate and Covariates sections below **in order in that same
DuckDB session**. All paths in their queries are relative to the repository root.
The data and generated results stay in the ignored `outputs/electricity/` directory.

At DuckDB's `D` prompt, run each part directly from its SQL file:

```text
.read examples/electricity-forecast.sql
.read examples/electricity-evaluate.sql
.read examples/electricity-covariates.sql
```

Run these one at a time, in order, to inspect each result. `.read` is a DuckDB CLI
command, not a shell command, and does not need a semicolon. The scripts use
temporary tables shared by that session. The SQL shown below is the equivalent
manual walkthrough; choose either the files or the inline SQL for a session.

The download recipe fetches these public AutoGluon datasets:

| File in `outputs/electricity/` | Public source |
| --- | --- |
| `prices-history.parquet` | [train.parquet](https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet) |
| `prices-next-day.parquet` | [test.parquet](https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet) |

The source columns are `id`, `timestamp`, `target`, `Ampirion Load Forecast`
(the source's spelling), and `PV+Wind Forecast`. There is one series, `DE`.
The queries below alias these to `id`, `date`, `price`, `load_forecast` and
`renewables_forecast`.

## Forecast with `tfc_forecast`

Run `.read examples/electricity-forecast.sql` at the DuckDB prompt.

Select the 512 hours from **2017-11-20 16:00 to 2017-12-11 23:00**, then forecast
the next 24 hours. Keep only observations available at the forecast cutoff in
`history`:

```sql
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
```

This returns **24 rows**, one per hour on **12 December 2017**, with columns
`id, date, prediction, p10, p25, p50, p75, p90, model, revision`.
`prediction` is the median, equal to `p50`. This first forecast uses **price
history only**: the load and renewables columns are ignored until explicitly
selected as covariates.

`id_cols` identifies independent series. Use multiple columns, such as
`['country', 'market']`, when your input has a composite ID, or omit it for one
series. IDs retain their names and types; NULL IDs form their own group.
Input rows need not be sorted. Column mappings are names, not expressions.

`context` defaults to the selected model's limit (**4,096** for alpha, **8,192** for beta): shorter histories use all available observations,
and longer histories use their most recent observations. Here we explicitly
select **512** hourly observations to match the blog.

### Choose quantiles

The walkthrough requests five quantiles in one call. You can change that
selection or return point forecasts alone:

```sql
-- Return p25, p50 and p75 for the same electricity history.
SELECT * FROM tfc_forecast('history', 24,
    target_col := 'price', timestamp_col := 'date', id_cols := ['id'],
    frequency := '1 hour', context := 512, quantiles := [0.25, 0.5, 0.75]);

-- Return point forecasts without percentile columns.
SELECT * FROM tfc_forecast('history', 24,
    target_col := 'price', timestamp_col := 'date', id_cols := ['id'],
    frequency := '1 hour', context := 512, quantiles := []);

-- Inspect the model and all supported quantile levels.
SELECT * FROM tfc_models();
```

The default is `[0.1, 0.5, 0.9]`. `prediction` is always the median. Alpha supports
`0.10`, `0.25`, `0.50`, `0.75` and `0.90`; beta supports 21 levels: `0.01`, every
0.05 from `0.05` to `0.95`, and `0.99`. Duplicate, NULL and unsupported levels
fail; no interpolation is performed. The graph computes all native levels,
and selection controls which columns are returned.

### Multiple target columns

`target_col` selects one numeric target per series. To forecast several columns,
reshape them with SQL and include the target name in the series ID. For example,
forecast both electricity price and the load-forecast series from their histories:

```sql
CREATE TEMP VIEW multiple_targets AS
UNPIVOT history ON price, load_forecast
INTO NAME target_name VALUE target;

SELECT * FROM tfc_forecast('multiple_targets', 24,
    target_col := 'target', timestamp_col := 'date',
    id_cols := ['id', 'target_name'], frequency := '1 hour', context := 512)
ORDER BY id, target_name, date;
```

This returns **48 rows**: 24 for each target. These are independent forecasts;
it does not make the two targets jointly predict one another. Use the same
reshape for actuals when evaluating multiple targets. DuckDB's `UNPIVOT` drops
NULL values by default; incomplete series are rejected by the grid checks.

## Evaluate with `tfc_evaluate`

Run `.read examples/electricity-evaluate.sql` at the DuckDB prompt.

The complete test file contains the **24 observed prices on 12 December 2017**.
Keep them separate from history and evaluate the price-only forecast:

```sql
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
```

Expected result for `DE`: **24 observations**, **MAE 6.18605**, **RMSE 6.95502**
and **MASE 0.47982** with the default alpha INT8 model. Small floating-point differences are possible.

Returns one row per series with `n, mae, mse, rmse, mape, smape, mase`, plus IDs,
model and revision. Loading `tfc_forecast` registers both forecast and evaluation
functions; there is no separate extension to load for evaluation.

**Evaluation runs inference again.** It uses history to generate predictions and
compares them with actuals; it does not read `forecast_result`. If you only need
accuracy metrics, call `tfc_evaluate` directly without forecasting first.
Held-out prices never enter inference or the MASE baseline.

| Metric | Meaning |
| --- | --- |
| MAE | Mean absolute error, in target units. |
| MSE | Mean squared error. |
| RMSE | Square root of MSE, in target units. |
| MAPE | Percentage error; rows with zero actuals are excluded. |
| sMAPE | Symmetric percentage error; a zero actual/prediction pair contributes zero. |
| MASE | MAE divided by the historical naive error at `seasonal_period`. |

Metrics use DOUBLE precision. All-zero actuals give NULL MAPE. Constant or
insufficient history gives NULL MASE. `seasonal_period := 24` uses a daily
baseline for hourly data; it changes the metric, not model inference.

History and actuals must have exactly the same series IDs and types. Actuals
must begin immediately after history, following its frequency. All actuals are
validated; the first `min(horizon, actual_count)` observations per series are scored.

## Add known-future covariates

Run `.read examples/electricity-covariates.sql` at the DuckDB prompt.

Covariates are additional model inputs. Here, use the day-ahead load and
renewables forecasts alongside price history. Their historical values are
already in `history`. For the forecast period, select only the IDs, timestamps
and these two forecasts—**no future price column**:

```sql
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
```

This returns another **24 rows** with the same five quantiles. To compare its
accuracy with the price-only baseline, pass the same covariates to evaluation:

```sql
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
```

Results below use the default **T0-alpha INT8** model; other variants differ.

| Inputs | MAE | RMSE | MASE |
| --- | ---: | ---: | ---: |
| Price history only | 6.18605 | 6.95502 | 0.47982 |
| History + load/renewables forecasts | 2.21409 | 2.96931 | 0.17173 |

Supply `covariate_cols` and `future_table` together. Covariates must be numeric,
finite and non-NULL. They cannot reuse target, timestamp or ID column names.
Each covariate needs both historical values and known future values; past-only
covariates are not supported. Historical values are trimmed with the target
context. Future rows must contain the same series IDs/types, start immediately after each history, and
cover every requested step on its frequency grid. **Short coverage raises an
error**. Extra future rows are validated but are not supplied to the model.
Evaluation requires coverage for the number of actual steps being scored.

Use covariate values that were available at the cutoff; realized future
measurements may leak information. The extension validates alignment but cannot
determine when a value became known. The files supply day-ahead forecasts, but
their release timestamps and forecast vintages have not been independently
audited. Historical-only and categorical covariates are not supported yet.

## Batch inference and timing

Both functions default to `batch_size := 8`, processing up to eight independent
series in one ONNX call. Use `batch_size := 1` for serial inference, or any
positive integer that fits in BIGINT. SQL macros collect series with matching
context lengths and horizons into explicit batches; Rust executes each supplied
batch in one ONNX call. Each target attends only to its own covariates; series
remain independent. Larger batches use more memory and are not always faster.

Before exiting the interactive session, you can time the electricity forecast
with covariates:

```sql
.timer on
CREATE OR REPLACE TEMP TABLE timed_forecast AS
SELECT * FROM tfc_forecast('history', 24,
    target_col := 'price', timestamp_col := 'date', id_cols := ['id'],
    frequency := '1 hour', context := 512,
    covariate_cols := ['load_forecast', 'renewables_forecast'],
    future_table := 'future_inputs',
    quantiles := [0.10, 0.25, 0.50, 0.75, 0.90], batch_size := 8);
.timer off
```

Time a materialized result to include the entire forecast. The first inference
loads the model into memory; repeat in the same session to measure warm calls.
Time evaluation separately because it includes another inference call. Small
floating-point differences between batch sizes are possible.

The two forecast timers in the 28-day script measure warm calls, including query
preparation and result materialization, with one DuckDB thread, four ONNX Runtime
intra-op threads and `batch_size := 8`. Source preparation, downloads and model
startup are excluded. The blog's **0.544 s / 1.124 s** observations came from an
Apple M4 with 16 GiB RAM and macOS 15.7.3. Your timings will vary; they are not
pass/fail thresholds. The walkthrough has been tested on macOS ARM64. The build
also handles Linux x86_64 with glibc; a Linux end-to-end run is not yet verified.

## Frequencies and input requirements

Set `frequency` explicitly: the electricity example uses `'1 hour'` because its
observations are hourly timestamps. The default is `'1 day'`; other supported
intervals include `'15 minutes'`, `'1 week'`, `'1 month'`, `'3 months'` and `'1 year'`.

- Timestamps must be `DATE`, `TIMESTAMP` or `TIMESTAMPTZ`; subdaily intervals
  require a timestamp. Output time is always named `date`, preserving its type.
- Every series must have a complete regular grid, with no duplicates or gaps,
  and finite, non-NULL dates and targets. Validation precedes context trimming.
- Calendar intervals advance from the original anchor, preserving month-end
  behavior. `TIMESTAMPTZ` follows the session `TimeZone`; use `'24 hours'` for
  fixed elapsed days across daylight-saving changes.
- Mapped names must be nonempty and distinct, case-insensitively. ID names must
  not conflict with forecast/metric output names, `history_values`, `origin`,
  `cutoff`, or the `__tfc_` prefix; alias those columns in a view.

## Parameter reference

| Parameter | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `source_table` | Required | Both | History table or view name. |
| `actuals_table` | Required | `tfc_evaluate` | Actual observations after history. |
| `horizon` | Required / `1024` | Forecast / eval | Forecast steps, integer 1–1024. |
| `target_col` | `'target'` | Both | Target column name. |
| `timestamp_col` | `'date'` | Both | Time column name. |
| `id_cols` | `[]` | Both | Columns identifying independent series. |
| `context` | Model limit (`4096` for alpha, `8192` for beta) | Both | Maximum history observations, from 1 to the model limit. |
| `frequency` | `'1 day'` | Both | Positive interval between observations. |
| `quantiles` | `[0.1, 0.5, 0.9]` | Forecast | Native quantile columns to return; `[]` for point only. |
| `covariate_cols` | `[]` | Both | Numeric known-future columns. |
| `future_table` | `NULL` | Both | Future covariates with matching IDs and dates. |
| `batch_size` | `8` | Both | Maximum independent series per SQL-built ONNX batch; positive BIGINT integer. |
| `seasonal_period` | `1` | `tfc_evaluate` | Positive integer lag for the MASE baseline. |

## License

Copyright 2026 The Forecasting Company, Inc.

The extension code, SQL examples, tests, and documentation in this repository
are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
Keep both `LICENSE` and `NOTICE` when redistributing this extension.

Third-party components and example data retain their own licenses:

- **Model weights:** the bundled T0 model choices are separately licensed
  under Apache-2.0. Preserve each downloaded `LICENSE` and any accompanying
  `NOTICE` when redistributing a model. All listed models are publicly
  downloadable without authentication. Other checkpoints selected with
  `TFC_MODEL_CONFIG` retain their own licenses.
- **ONNX Runtime:** the downloaded package includes its own `LICENSE` and
  `ThirdPartyNotices.txt`, which the extension retains in the runtime cache.
  Preserve these notices when redistributing the runtime.
- **Rust dependencies:** their licenses are separate from this extension's
  license. Before distributing compiled extension binaries, review the
  dependencies included in that build and ship the required license and
  attribution notices, plus any required source-availability information.
- **Electricity data:** the examples download data from the public sources
  linked above. The extension's Apache-2.0 license does not license those
  datasets; check their upstream terms before redistributing them.

## Development

Run these commands from the repository root with Rust 1.89+ selected:

```bash
just check          # Rust formatting, unit tests and Clippy
just test-contract  # SQL contracts and validation; no model needed
just test-native    # Build and test real inference; downloads uncached artifacts
```

Use `TFC_OFFLINE=1 just test-native` once the selected model and runtime are cached.
`just test` runs all three checks. Rust dependencies must be fetched first with
`just fetch`. Build output, downloaded artifacts and generated data are ignored
by Git.
