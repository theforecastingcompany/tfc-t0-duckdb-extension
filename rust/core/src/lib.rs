//! Local T0 ONNX executor for the DuckDB feasibility prototype.
#![deny(unsafe_code)]
use ort::{session::Session, value::Tensor};
use std::{error::Error, path::Path};

pub mod artifact;
mod download;
pub mod model;
pub mod runtime;

pub type ForecastError = Box<dyn Error>;

pub struct T0Forecaster {
    session: Session,
    quantile_count: usize,
    max_context: usize,
}

/// A validated single-target model request. SQL owns IDs, dates and column mapping.
#[derive(Debug)]
pub struct ForecastInput {
    target_history: Vec<f32>,
    horizon: usize,
    // Time-major: all covariates for observation 0, then observation 1.
    covariate_history: Vec<f32>,
    covariate_future: Vec<f32>,
}

impl ForecastInput {
    pub fn new(
        target_history: Vec<f32>,
        horizon: usize,
        covariate_history: Vec<f32>,
        covariate_future: Vec<f32>,
    ) -> Result<Self, ForecastError> {
        validate(&target_history, horizon)?;
        if !covariate_history.len().is_multiple_of(target_history.len()) {
            return Err("covariate history must align with every history observation".into());
        }
        let columns = covariate_history.len() / target_history.len();
        if covariate_future.len() != columns * horizon {
            return Err("future covariates must cover every requested forecast step".into());
        }
        if !covariate_history
            .iter()
            .chain(&covariate_future)
            .all(|v| v.is_finite())
        {
            return Err("covariates must contain only finite, non-null values".into());
        }
        Ok(Self {
            target_history,
            horizon,
            covariate_history,
            covariate_future,
        })
    }

    pub fn context(&self) -> usize {
        self.target_history.len()
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

struct Batch {
    context: usize,
    horizon: usize,
    compute_horizon: usize,
    targets: Vec<f32>,
    target_groups: Vec<i32>,
    covariate_history: Vec<f32>,
    covariate_future: Vec<f32>,
    covariate_groups: Vec<i32>,
}

impl Batch {
    fn new(inputs: &[&ForecastInput]) -> Result<Self, ForecastError> {
        if inputs.is_empty() {
            return Err("batch must contain at least one series".into());
        }
        // Group identifiers are int32 in the ONNX input contract.
        i32::try_from(inputs.len() - 1).map_err(|_| "too many series for ONNX group IDs")?;
        let first = inputs[0];
        // Every request was validated by its constructor. Check the entire
        // batch shape before allocating tensors; no request silently sets another's shape.
        if inputs
            .iter()
            .any(|input| input.context() != first.context() || input.horizon() != first.horizon())
        {
            return Err("batched series must have matching context lengths and horizons".into());
        }
        let mut batch = Self {
            context: first.context(),
            horizon: first.horizon,
            compute_horizon: first.horizon.div_ceil(32) * 32,
            targets: Vec::new(),
            target_groups: Vec::new(),
            covariate_history: Vec::new(),
            covariate_future: Vec::new(),
            covariate_groups: Vec::new(),
        };
        for (group, input) in inputs.iter().enumerate() {
            batch.targets.extend_from_slice(&input.target_history);
            batch.target_groups.push(group as i32);
            let columns = input.covariate_history.len() / batch.context;
            for column in 0..columns {
                batch.covariate_groups.push(group as i32);
                batch.covariate_history.extend(
                    (0..batch.context).map(|t| input.covariate_history[t * columns + column]),
                );
                batch
                    .covariate_future
                    .extend((0..batch.compute_horizon).map(|t| {
                        if t < batch.horizon {
                            input.covariate_future[t * columns + column]
                        } else {
                            f32::NAN
                        }
                    }));
            }
        }
        Ok(batch)
    }
}

fn validate(history: &[f32], horizon: usize) -> Result<(), ForecastError> {
    if !(1..=8192).contains(&history.len()) {
        return Err("context length must be 1..8192".into());
    }
    if !(1..=1024).contains(&horizon) {
        return Err("horizon must be 1..1024".into());
    }
    if !history.iter().all(|x| x.is_finite()) {
        return Err("history must contain only finite, non-null values".into());
    }
    Ok(())
}

impl T0Forecaster {
    pub fn load(
        model: &artifact::ModelArtifact,
        runtime_path: &Path,
    ) -> Result<Self, ForecastError> {
        model.verify()?;
        if !runtime_path.is_file() {
            return Err("ONNX Runtime shared library does not exist".into());
        }
        ort::init_from(runtime_path.to_string_lossy())
            .with_name("tfc-forecast")
            .with_telemetry(false)
            .commit()?;
        let session = Session::builder()?
            .with_intra_threads(4)?
            .with_inter_threads(1)?
            .commit_from_file(model.path())?;
        Ok(Self {
            session,
            quantile_count: model.spec().quantile_levels().len(),
            max_context: model.spec().max_context(),
        })
    }

    /// Returns horizon-major native quantiles. Model scaling is inside the graph.
    pub fn predict(&mut self, history: &[f32], horizon: usize) -> Result<Vec<f32>, ForecastError> {
        let input = ForecastInput::new(history.to_vec(), horizon, Vec::new(), Vec::new())?;
        Ok(self.predict_batch(&[&input])?.remove(0))
    }

    pub fn predict_batch(
        &mut self,
        inputs: &[&ForecastInput],
    ) -> Result<Vec<Vec<f32>>, ForecastError> {
        if inputs
            .iter()
            .any(|input| input.context() > self.max_context)
        {
            return Err(format!(
                "selected model supports context length at most {}",
                self.max_context
            )
            .into());
        }
        let batch = Batch::new(inputs)?;
        let count = inputs.len();
        let covariates = batch.covariate_groups.len();
        let target = Tensor::from_array(([count, batch.context], batch.targets))?;
        let groups = Tensor::from_array(([count], batch.target_groups))?;
        // ort's raw Vec constructor rejects zero dimensions. Allocate these
        // zero-element tensors through ORT to preserve the graph's contract.
        let (covariate_history, covariate_future, covariate_groups) = if covariates == 0 {
            (
                Tensor::<f32>::new(self.session.allocator(), [0, batch.context])?,
                Tensor::<f32>::new(self.session.allocator(), [0, batch.compute_horizon])?,
                Tensor::<i32>::new(self.session.allocator(), [0_usize])?,
            )
        } else {
            (
                Tensor::from_array(([covariates, batch.context], batch.covariate_history))?,
                Tensor::from_array(([covariates, batch.compute_horizon], batch.covariate_future))?,
                Tensor::from_array(([covariates], batch.covariate_groups))?,
            )
        };
        let output = self.session.run(ort::inputs! {
            "target_context" => target,
            "target_group_ids" => groups,
            "future_covariate_context" => covariate_history,
            "future_covariate_future" => covariate_future,
            "future_covariate_group_ids" => covariate_groups
        })?;
        let (shape, values) = output["quantiles"].try_extract_tensor::<f32>()?;
        if **shape
            != [
                count as i64,
                batch.compute_horizon as i64,
                self.quantile_count as i64,
            ]
        {
            return Err(format!("unexpected forecast shape: {shape:?}").into());
        }
        if !values.iter().all(|x| x.is_finite())
            || values
                .chunks_exact(self.quantile_count)
                .any(|row| row.windows(2).any(|pair| pair[0] > pair[1] + 1e-6))
        {
            return Err("model produced non-finite or crossing quantiles".into());
        }
        Ok(values
            .chunks_exact(batch.compute_horizon * self.quantile_count)
            .map(|row| row[..batch.horizon * self.quantile_count].to_vec())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> ForecastInput {
        ForecastInput::new(
            vec![1., 2., 3.],
            2,
            vec![10., 100., 20., 200., 30., 300.],
            vec![40., 400., 50., 500.],
        )
        .unwrap()
    }

    #[test]
    fn batch_associates_covariates_and_pads_only_unrequested_steps() {
        let first = example();
        let second = ForecastInput::new(vec![4., 5., 6.], 2, vec![], vec![]).unwrap();
        let batch = Batch::new(&[&first, &second]).unwrap();
        assert_eq!(batch.target_groups, [0, 1]);
        assert_eq!(batch.covariate_groups, [0, 0]);
        assert_eq!(batch.targets, [1., 2., 3., 4., 5., 6.]);
        assert_eq!(batch.covariate_history, [10., 20., 30., 100., 200., 300.]);
        assert_eq!(&batch.covariate_future[..2], [40., 50.]);
        assert_eq!(&batch.covariate_future[32..34], [400., 500.]);
        assert!(batch.covariate_future[2..32].iter().all(|v| v.is_nan()));
        assert!(batch.covariate_future[34..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn invalid_inputs_cannot_be_constructed() {
        for history in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![1.; 8193]] {
            assert!(ForecastInput::new(history, 2, vec![], vec![]).is_err());
        }
        for horizon in [0, 1025] {
            assert!(ForecastInput::new(vec![1.], horizon, vec![], vec![]).is_err());
        }
        assert!(ForecastInput::new(vec![1., 2., 3.], 2, vec![1.], vec![]).is_err());
        assert!(ForecastInput::new(vec![1., 2., 3.], 2, vec![1., 2., 3.], vec![4.]).is_err());
        assert!(ForecastInput::new(vec![1.], 1, vec![f32::NAN], vec![1.]).is_err());
        assert!(ForecastInput::new(vec![1.], 1, vec![1.], vec![f32::INFINITY]).is_err());
        assert!(ForecastInput::new(vec![1.; 8192], 1024, vec![], vec![]).is_ok());
    }

    #[test]
    fn every_batch_member_must_match_the_shape() {
        let first = example();
        let longer = ForecastInput::new(vec![1.; 4], 2, vec![], vec![]).unwrap();
        let farther = ForecastInput::new(vec![1.; 3], 3, vec![], vec![]).unwrap();
        for other in [&longer, &farther] {
            assert!(Batch::new(&[&first, other]).is_err());
            assert!(Batch::new(&[other, &first]).is_err());
        }
        assert!(Batch::new(&[]).is_err());
        let large = Batch::new(&vec![&first; 65]).unwrap();
        assert_eq!(large.target_groups, (0..65).collect::<Vec<_>>());
        assert_eq!(large.targets.len(), 65 * first.context());
    }
}
