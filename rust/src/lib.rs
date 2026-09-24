//! A small native inference bridge. Table preparation and evaluation live in SQL.
#![deny(unsafe_code)]
// DuckDB's current Arrow input adapter cannot read LIST vectors.
#[allow(unsafe_code)]
mod vectors;

use duckdb::{
    Connection,
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
};
use duckdb_ext_core::{T0Forecaster, artifact::ModelArtifact, model::ModelSpec};
use std::{
    error::Error,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct State {
    model: ModelArtifact,
    runtime: PathBuf,
    // One session per registration, serialized to bound inference concurrency.
    session: Arc<Mutex<Option<T0Forecaster>>>,
}

struct Forecast;

impl VScalar for Forecast {
    type State = State;

    fn signatures() -> Vec<ScalarFunctionSignature> {
        let floats = || LogicalTypeHandle::list(&LogicalTypeId::Float.into());
        let request = LogicalTypeHandle::struct_type(&[
            ("history", floats()),
            ("covariate_history", floats()),
            ("covariate_future", floats()),
        ]);
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::list(&request),
                LogicalTypeId::Bigint.into(),
            ],
            LogicalTypeHandle::list(&floats()),
        )]
    }

    fn invoke(
        state: &State,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let mut predictions = Vec::with_capacity(input.len());
        let mut engine = state
            .session
            .lock()
            .map_err(|_| "forecast session lock poisoned")?;
        // Each SQL row is a complete model batch. DuckDB may deliver several
        // batches in a chunk, but their membership never depends on chunking.
        for row in 0..input.len() {
            let requests = vectors::read_batch(input, row)?;
            if engine.is_none() {
                *engine = Some(T0Forecaster::load(&state.model, &state.runtime)?);
            }
            let inputs = requests.iter().collect::<Vec<_>>();
            let values = engine
                .as_mut()
                .ok_or("session unavailable")?
                .predict_batch(&inputs)?;
            predictions.push(values);
        }
        vectors::write_batches(&predictions, output)
    }
}

// The upstream attribute generates DuckDB's required unsafe C ABI entrypoint.
// Keep that generated code in one module; all handwritten inference/parsing code
// remains subject to the crate's deny(unsafe_code) lint.
#[allow(unsafe_code)]
mod entrypoint {
    use super::*;
    #[duckdb_entrypoint_c_api(ext_name = "tfc_forecast", min_duckdb_version = "v1.5.4")]
    pub fn extension_entrypoint(connection: Connection) -> Result<(), Box<dyn Error>> {
        let spec = ModelSpec::from_env()?;
        let sql = include_str!("../sql/macros.sql")
            .replace("__TFC_LEVELS__", &format!("{:?}", spec.quantile_levels()))
            .replace("__TFC_MAX_CONTEXT__", &spec.max_context().to_string())
            .replace(
                "__TFC_MODEL_REVISION__",
                &spec.revision().replace('\'', "''"),
            )
            .replace("__TFC_MODEL_ID__", &spec.id().replace('\'', "''"));
        let state = State {
            runtime: duckdb_ext_core::runtime::resolve_runtime()?,
            model: duckdb_ext_core::artifact::resolve_model(spec)?,
            session: Arc::new(Mutex::new(None)),
        };
        connection.register_scalar_function_with_state::<Forecast>("tfc_forecast_batch", &state)?;
        connection.execute_batch(&sql)?;
        Ok(())
    }
}
