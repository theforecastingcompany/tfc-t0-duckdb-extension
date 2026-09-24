//! Checkpoint identity is configuration; tensor layout is an explicit adapter contract.
use crate::{ForecastError, download};
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    id: String,
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    bytes: u64,
    interface: String,
    #[serde(default = "beta_levels")]
    quantile_levels: Vec<f64>,
    #[serde(default = "default_max_context")]
    max_context: usize,
    #[serde(default = "default_metadata")]
    metadata_files: Vec<String>,
}

pub const DEFAULT_MODEL: &str = "t0-alpha-onnx-int8";

fn beta_levels() -> Vec<f64> {
    vec![
        0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
    ]
}
fn default_max_context() -> usize {
    8192
}
fn default_metadata() -> Vec<String> {
    ["LICENSE", "NOTICE", "config.json"]
        .map(str::to_owned)
        .to_vec()
}

impl ModelSpec {
    pub fn from_env() -> Result<Self, ForecastError> {
        match env::var_os("TFC_MODEL_CONFIG") {
            Some(path) => Self::from_json(&fs::read_to_string(path)?),
            None => Self::named(&env::var("TFC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into())),
        }
    }

    pub fn named(name: &str) -> Result<Self, ForecastError> {
        let models: Vec<Self> = serde_json::from_str(include_str!("models.json"))?;
        let names = models
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        models
            .into_iter()
            .find(|s| s.id == name)
            .ok_or_else(|| format!("unknown TFC_MODEL {name:?}; choose {names}").into())
            .and_then(Self::validated)
    }

    pub fn from_json(json: &str) -> Result<Self, ForecastError> {
        Self::validated(serde_json::from_str(json)?)
    }

    fn validated(spec: Self) -> Result<Self, ForecastError> {
        let component = |s: &str| {
            !s.is_empty()
                && s != "."
                && s != ".."
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
        };
        let parts: Vec<_> = spec.repository.split('/').collect();
        if spec.id.trim().is_empty()
            || parts.len() != 2
            || !parts.iter().all(|s| component(s))
            || !component(&spec.filename)
            || spec.bytes == 0
        {
            return Err("model config requires an ID, owner/repository, plain filename and positive byte size".into());
        }
        if spec.revision.len() != 40
            || !spec.revision.bytes().all(|c| c.is_ascii_hexdigit())
            || spec.sha256.len() != 64
            || !spec.sha256.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err(
                "model config requires a pinned 40-character revision and 64-character SHA-256"
                    .into(),
            );
        }
        if spec.interface != "t0-grouped-v1" {
            return Err("unsupported model interface: expected t0-grouped-v1".into());
        }
        if !(1..=8192).contains(&spec.max_context)
            || !spec.quantile_levels.contains(&0.5)
            || spec
                .quantile_levels
                .iter()
                .any(|q| !beta_levels().contains(q))
            || spec
                .quantile_levels
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err("model config requires a context limit in 1..8192 and sorted unique supported quantile levels including 0.5".into());
        }
        if !spec.metadata_files.iter().any(|name| name == "LICENSE")
            || !spec.metadata_files.iter().any(|name| name == "config.json")
            || spec
                .metadata_files
                .iter()
                .any(|name| !component(name) || name == &spec.filename)
        {
            return Err(
                "model metadata must include LICENSE and config.json and use plain filenames"
                    .into(),
            );
        }
        Ok(spec)
    }

    pub fn quantile_levels(&self) -> &[f64] {
        &self.quantile_levels
    }
    pub fn max_context(&self) -> usize {
        self.max_context
    }
    pub fn metadata_files(&self) -> &[String] {
        &self.metadata_files
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn repository(&self) -> &str {
        &self.repository
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn verify(&self, path: &Path) -> Result<(), ForecastError> {
        download::verify_sha256(path, &self.sha256.to_ascii_lowercase())?;
        if path.metadata()?.len() != self.bytes {
            return Err("unexpected model artifact size".into());
        }
        Ok(())
    }

    pub(crate) fn cache_directory(&self, root: &Path) -> PathBuf {
        root.join("models")
            .join(&self.repository)
            .join(&self.revision)
            .join(&self.sha256)
    }

    pub(crate) fn uses_legacy_cache(&self) -> bool {
        Self::named("t0-beta-onnx-fp16").is_ok_and(|legacy| *self == legacy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_covers_both_families_and_precisions() {
        let specs: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("models.json")).unwrap();
        assert_eq!(specs.len(), 4);
        for value in specs {
            let spec = ModelSpec::from_json(&value.to_string()).unwrap();
            assert_eq!(ModelSpec::named(spec.id()).unwrap(), spec);
            let alpha = spec.id().starts_with("t0-alpha");
            assert_eq!(spec.quantile_levels().len(), if alpha { 5 } else { 21 });
            assert_eq!(spec.max_context(), if alpha { 4096 } else { 8192 });
            assert_eq!(spec.metadata_files().iter().any(|s| s == "NOTICE"), !alpha);
        }
        assert_eq!(ModelSpec::named(DEFAULT_MODEL).unwrap().bytes, 107151882);
        assert!(ModelSpec::named("unknown").is_err());
    }

    #[test]
    fn config_rejects_unpinned_or_incompatible_models() {
        let values: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("models.json")).unwrap();
        for (key, value) in [
            ("revision", serde_json::json!("main")),
            ("interface", serde_json::json!("other-graph")),
            ("filename", serde_json::json!("../model.onnx")),
            ("bytes", serde_json::json!(0)),
            ("max_context", serde_json::json!(0)),
            ("quantile_levels", serde_json::json!([0.1, 0.1, 0.5])),
            ("quantile_levels", serde_json::json!([0.1, 0.9])),
            ("quantile_levels", serde_json::json!([0.11, 0.5])),
            (
                "metadata_files",
                serde_json::json!(["LICENSE", "config.json", "../NOTICE"]),
            ),
        ] {
            let mut invalid = values[0].clone();
            invalid[key] = value;
            assert!(ModelSpec::from_json(&invalid.to_string()).is_err(), "{key}");
        }
    }

    #[test]
    fn old_custom_configs_keep_beta_defaults_and_cache() {
        let mut spec: serde_json::Value =
            serde_json::from_str(include_str!("models.json")).unwrap();
        let value = spec[3].as_object_mut().unwrap();
        for key in ["quantile_levels", "max_context", "metadata_files"] {
            value.remove(key);
        }
        let legacy = ModelSpec::from_json(&spec[3].to_string()).unwrap();
        assert!(legacy.uses_legacy_cache());
        assert_ne!(
            legacy.cache_directory(Path::new("cache")),
            ModelSpec::named(DEFAULT_MODEL)
                .unwrap()
                .cache_directory(Path::new("cache"))
        );
    }
}
