set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

default:
    @just --list

# Fetch locked Rust dependencies once; subsequent builds run offline.
fetch:
    cargo fetch --manifest-path rust/Cargo.toml --locked

# Build and package the local unsigned development extension.
build: _check-host
    cargo build --manifest-path rust/Cargo.toml --target-dir rust/target --locked --offline -p duckdb-ext
    @just package debug

# Build and package an optimized extension at the same output path.
release: _check-host
    cargo build --manifest-path rust/Cargo.toml --target-dir rust/target --locked --offline --release -p duckdb-ext
    @just package release

# Package an existing library without rebuilding (debug or release).
package profile="debug": _check-host
    #!/usr/bin/env bash
    set -euo pipefail
    case "$1" in debug|release) ;; *) echo 'Profile must be debug or release' >&2; exit 1;; esac
    platform=$(duckdb -batch -noheader -csv :memory: 'PRAGMA platform;')
    case "$platform" in
        osx_arm64) suffix=dylib ;;
        linux_amd64|linux_amd64_gcc4) suffix=so ;;
        *) echo "Unsupported packaging platform: $platform" >&2; exit 1 ;;
    esac
    library="rust/target/$1/libtfc_forecast.$suffix"
    test -f "$library" || { echo "Build the library first: $library" >&2; exit 1; }
    mkdir -p build
    staging=$(mktemp -d build/.package.XXXXXX)
    trap 'rm -rf "$staging"' EXIT
    output="$staging/tfc_forecast.duckdb_extension"
    # A new inode avoids stale macOS code-signature pages after a previous LOAD.
    cp "$library" "$output"
    {
        printf '\x00\x93\x04\x10duckdb_signature\x80\x04'
        for field in '' '' '' C_STRUCT_UNSTABLE 0.1.0 v1.5.4 "$platform" 4; do
            printf '%s' "$field"
            head -c "$((32 - ${#field}))" /dev/zero
        done
        head -c 256 /dev/zero
    } >> "$output"
    mv -f "$output" build/tfc_forecast.duckdb_extension
    printf '%s/build/tfc_forecast.duckdb_extension\n' "$PWD"

# Rust unit tests, formatting and linting; no model download.
check:
    cargo fmt --manifest-path rust/Cargo.toml --all --check
    cargo test --manifest-path rust/Cargo.toml --locked --offline --workspace
    cargo clippy --manifest-path rust/Cargo.toml --locked --offline --workspace --all-targets -- -D warnings

# Deterministic SQL contracts and validation errors; no model/runtime required.
test-contract: _check-host
    #!/usr/bin/env bash
    set -euo pipefail
    sql_contract() {
        printf '%s\n' 'CREATE MACRO tfc_forecast_batch(requests,horizon) AS list_transform(requests, lambda r: list_transform(range(horizon*21), lambda i: (r.history[1]*0+1)::FLOAT));'
        sed -e 's/__TFC_LEVELS__/[0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99]/g' -e 's/__TFC_MAX_CONTEXT__/8192/g' rust/sql/macros.sql
        cat tests/contract.sql
    }
    sql_contract | duckdb -batch -bail :memory: >/dev/null
    while IFS='|' read -r expected sql; do
        if output=$({ sql_contract; printf '%s\n' "$sql"; } | duckdb -batch -bail :memory: 2>&1); then
            echo "Expected failure: $sql" >&2; exit 1
        fi
        if [[ "$output" != *"$expected"* ]]; then
            printf 'Expected %s, got: %s\n' "$expected" "$output" >&2; exit 1
        fi
    done <<'CASES'
    quantiles|FROM tfc_forecast('history',2,quantiles:=[0.11]);
    quantiles|FROM tfc_forecast('history',2,quantiles:=[0.1,0.1]);
    context|FROM tfc_forecast('history',2,context:=8193);
    subdaily|FROM tfc_forecast('history',2,frequency:='1 hour');
    distinct|FROM tfc_forecast('history',2,target_col:='date');
    regular grid|CREATE VIEW gaps AS SELECT * FROM history WHERE target<>3; FROM tfc_forecast('gaps',2);
    duplicate|CREATE VIEW dup AS SELECT * FROM history UNION ALL SELECT * FROM history; FROM tfc_forecast('dup',2);
    NULL|UPDATE long_history SET target=NULL WHERE target=0; FROM tfc_forecast('long_history',1);
    frequency grid|UPDATE actuals SET date=date-1; FROM tfc_evaluate('history','actuals');
    cover every|FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_short');
    cover every|FROM tfc_evaluate('cov_history','actuals',horizon:=2,covariate_cols:=['price'],future_table:='cov_short');
    frequency grid|FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_gap');
    NULL|FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_null');
    duplicate dates|FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_duplicate');
    supplied together|FROM tfc_forecast('cov_history',2,covariate_cols:=['price']);
    supplied together|FROM tfc_forecast('cov_history',2,future_table:='cov_future');
    distinct|FROM tfc_forecast('cov_history',2,covariate_cols:=['target'],future_table:='actuals');
    distinct|FROM tfc_forecast('cov_history',2,covariate_cols:=['price','PRICE'],future_table:='cov_future');
    nonempty|FROM tfc_forecast('cov_history',2,covariate_cols:=NULL);
    batch_size|FROM tfc_forecast('history',2,batch_size:=0);
    batch_size|FROM tfc_forecast('history',2,batch_size:=NULL);
    batch_size|FROM tfc_forecast('history',2,batch_size:='Infinity'::DOUBLE);
    batch_size|FROM tfc_forecast('history',2,batch_size:=1e30);
    batch_size|FROM tfc_forecast('history',2,batch_size:=1.5);
    same series IDs|FROM tfc_forecast('cov_group_history',2,id_cols:=['store'],covariate_cols:=['price'],future_table:='cov_extra');
    same ID types|FROM tfc_forecast('cov_group_history',2,id_cols:=['store'],covariate_cols:=['price'],future_table:='cov_wrong_id_type');
    same timestamp type|FROM tfc_forecast('cov_history',2,covariate_cols:=['price'],future_table:='cov_wrong_date_type');
    CASES
    { sql_contract; cat tests/batching.sql; } | duckdb -batch -bail :memory: >/dev/null
    echo 'SQL contract, batching and validation checks passed'

# Real native inference; downloads missing artifacts unless TFC_OFFLINE=1.
test-native: build
    #!/usr/bin/env bash
    set -euo pipefail
    extension="$PWD/build/tfc_forecast.duckdb_extension"
    extension=${extension//\'/\'\'}
    staging=$(mktemp -d build/.test.XXXXXX)
    trap 'rm -rf "$staging"' EXIT
    # LOAD must also work when a second session reopens the same database.
    for pass in 1 2; do
        { printf "LOAD '%s';\n" "$extension"; cat tests/native.sql tests/covariates.sql tests/bridge.sql; } | duckdb -unsigned -batch -bail "$staging/reopen.duckdb" >/dev/null
    done
    # Exercise malformed native arguments through the actual DuckDB vectors.
    while IFS='|' read -r expected sql; do
        if output=$({ printf "LOAD '%s';\n" "$extension"; printf '%s\n' "$sql"; } | duckdb -unsigned -batch -bail :memory: 2>&1); then
            echo "Expected native failure: $sql" >&2; exit 1
        fi
        if [[ "$output" != *"$expected"* ]]; then
            printf 'Expected %s, got: %s\n' "$expected" "$output" >&2; exit 1
        fi
    done <<'CASES'
    context length|SELECT tfc_forecast_quantiles([]::FLOAT[],1);
    horizon|SELECT tfc_forecast_quantiles([1]::FLOAT[],0);
    NULL|SELECT tfc_forecast_quantiles([1,NULL]::FLOAT[],1);
    finite|SELECT tfc_forecast_quantiles([1,'NaN'::FLOAT]::FLOAT[],1);
    align|SELECT tfc_forecast_quantiles([1,2]::FLOAT[],1,[1]::FLOAT[],[2]::FLOAT[]);
    cover every|SELECT tfc_forecast_quantiles([1,2]::FLOAT[],2,[1,2]::FLOAT[],[3]::FLOAT[]);
    at least one|SELECT tfc_forecast_batch([]::STRUCT(history FLOAT[], covariate_history FLOAT[], covariate_future FLOAT[])[],1);
    NULL|SELECT tfc_forecast_batch([NULL]::STRUCT(history FLOAT[], covariate_history FLOAT[], covariate_future FLOAT[])[],1);
    NULL|SELECT tfc_forecast_quantiles(NULL::FLOAT[],1);
    NULL|SELECT tfc_forecast_quantiles([1]::FLOAT[],1,NULL::FLOAT[],[]::FLOAT[]);
    NULL|SELECT tfc_forecast_quantiles([1]::FLOAT[],1,[1]::FLOAT[],[NULL]::FLOAT[]);
    finite|SELECT tfc_forecast_batch(list_transform(range(2051),lambda i:struct_pack(history:=[CASE WHEN i=2050 THEN 'NaN'::FLOAT ELSE 1::FLOAT END],covariate_history:=[]::FLOAT[],covariate_future:=[]::FLOAT[])),1);
    matching context|SELECT tfc_forecast_batch([{'history':[1]::FLOAT[],'covariate_history':[]::FLOAT[],'covariate_future':[]::FLOAT[]},{'history':[1,2]::FLOAT[],'covariate_history':[]::FLOAT[],'covariate_future':[]::FLOAT[]}],1);
    context|SELECT tfc_forecast_quantiles(list_transform(range((SELECT max_context+1 FROM tfc_models())),lambda i: 1::FLOAT),1);
    context|SELECT __tfc_context((SELECT max_context+1 FROM tfc_models()));
    CASES
    echo 'Native forecast, evaluation and database reopen checks passed'

test: check test-contract test-native

# Download the two public input files and verify the exact demo snapshots.
electricity-data:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p outputs/electricity
    cd outputs/electricity
    curl -fL https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet -o prices-history.parquet
    curl -fL https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet -o prices-next-day.parquet
    if command -v sha256sum >/dev/null 2>&1; then
        checksum=(sha256sum -c -)
    elif command -v shasum >/dev/null 2>&1; then
        checksum=(shasum -a 256 -c -)
    else
        echo 'Install sha256sum or shasum to verify the downloaded data' >&2
        exit 1
    fi
    "${checksum[@]}" <<'CHECKSUMS'
    f2dc2eed69be26736c4bcb8a45703ce9c815965f72d05d555533dc829379dc05  prices-history.parquet
    b3f739cc81f281411d602456dba311d67425b8599fe7e8884b17ea8e17b6cdb8  prices-next-day.parquet
    CHECKSUMS

# Reproduce the blog's single-day example, 28-day comparison and chart CSVs.
electricity: release
    #!/usr/bin/env bash
    set -euo pipefail
    extension="$PWD/build/tfc_forecast.duckdb_extension"
    extension=${extension//\'/\'\'}
    { printf "LOAD '%s';\n" "$extension"; cat examples/electricity.sql examples/electricity-28-days.sql; } | duckdb -unsigned -batch -bail :memory:

[private]
_check-host:
    #!/usr/bin/env bash
    set -euo pipefail
    version=$(duckdb -batch -noheader -csv :memory: 'SELECT version();')
    platform=$(duckdb -batch -noheader -csv :memory: 'PRAGMA platform;')
    test "$version" = v1.5.4 || { echo 'DuckDB CLI 1.5.4 is required' >&2; exit 1; }
    case "$platform" in
        osx_arm64|linux_amd64|linux_amd64_gcc4) ;;
        *) echo "Unsupported packaging platform: $platform (expected macOS ARM64 or Linux x86_64)" >&2; exit 1 ;;
    esac
