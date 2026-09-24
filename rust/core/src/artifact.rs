//! Thin Hugging Face client wrapper with pinned-artifact verification.
use crate::{ForecastError, download, model::ModelSpec};
use hf_hub::HFClient;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct ModelArtifact {
    path: PathBuf,
    spec: ModelSpec,
}

impl ModelArtifact {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    pub fn verify(&self) -> Result<(), ForecastError> {
        self.spec.verify(&self.path)
    }
}

pub fn resolve_model(spec: ModelSpec) -> Result<ModelArtifact, ForecastError> {
    let path = if let Some(path) = env::var_os("TFC_MODEL_PATH") {
        PathBuf::from(path)
    } else {
        let root = download::cache_root()?;
        // Reuse a previously verified default-model cache without redownloading.
        let legacy = root.join(spec.revision());
        let directory = if spec.uses_legacy_cache() && legacy.join(spec.filename()).is_file() {
            legacy
        } else {
            spec.cache_directory(&root)
        };
        provision(&spec, &directory, download::offline())?
    };
    spec.verify(&path)?;
    Ok(ModelArtifact { path, spec })
}

fn verify_cached(spec: &ModelSpec, directory: &Path) -> Result<bool, ForecastError> {
    let model = directory.join(spec.filename());
    if model.is_file() {
        spec.verify(&model)?;
    }
    Ok(model.is_file()
        && spec
            .metadata_files()
            .iter()
            .all(|name| directory.join(name).is_file()))
}

pub fn provision(
    spec: &ModelSpec,
    directory: &Path,
    offline: bool,
) -> Result<PathBuf, ForecastError> {
    if verify_cached(spec, directory)? {
        return Ok(directory.join(spec.filename()));
    }
    if offline {
        return Err(format!(
            "TFC_OFFLINE: missing model package files in {}",
            directory.display()
        )
        .into());
    }
    fs::create_dir_all(directory)?;
    // hf-hub supports public downloads without a token and handles redirects
    // and transfers (plus optional credentials for custom models). Keep temporary
    // files here; only publish a model after our pinned checksum passes.
    let staging = tempfile::tempdir_in(directory)?;
    let client = HFClient::builder()
        .endpoint("https://huggingface.co")
        .client(reqwest::Client::builder().https_only(true).build()?)
        .cache_dir(staging.path().join("hub"))
        .build_sync()?;
    let (owner, name) = spec
        .repository()
        .split_once('/')
        .ok_or("invalid model repository")?;
    let repo = client.model(owner, name);
    for filename in spec
        .metadata_files()
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(spec.filename()))
    {
        if directory.join(filename).is_file() {
            continue;
        }
        let path = repo
            .download_file()
            .filename(filename)
            .revision(spec.revision())
            .local_dir(staging.path().to_path_buf())
            .send()
            // Do not expose signed download URLs or response bodies in errors.
            .map_err(|_| {
                format!(
                    "Hugging Face download failed for {filename}; check connectivity and the selected repository/revision"
                )
            })?;
        if filename == spec.filename() {
            spec.verify(&path)?;
        } else if path.metadata()?.len() > 1_048_576 {
            return Err(format!("model metadata file {filename} exceeds 1 MiB").into());
        }
        fs::File::open(&path)?.sync_all()?;
        // A no-clobber hard link publishes the complete staged file atomically.
        // Staging is on the same filesystem; an existing concurrent winner is
        // verified below before we return. The temporary link is then removed.
        if let Err(error) = fs::hard_link(&path, directory.join(filename))
            && error.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(error.into());
        }
    }
    if !verify_cached(spec, directory)? {
        return Err("model package is incomplete after download".into());
    }
    Ok(directory.join(spec.filename()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ModelSpec {
        ModelSpec::from_json(
            r#"{"id":"fixture","repository":"example/model",
            "revision":"0000000000000000000000000000000000000000","filename":"model.onnx",
            "sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "bytes":3,"interface":"t0-grouped-v1"}"#,
        )
        .unwrap()
    }

    #[test]
    fn offline_cache_requires_complete_verified_package() {
        let directory = tempfile::tempdir().unwrap();
        let spec = fixture();
        assert!(
            provision(&spec, directory.path(), true)
                .unwrap_err()
                .to_string()
                .contains("TFC_OFFLINE")
        );
        fs::write(directory.path().join(spec.filename()), b"abc").unwrap();
        assert!(provision(&spec, directory.path(), true).is_err());
        for name in ["LICENSE", "NOTICE", "config.json"] {
            fs::write(directory.path().join(name), b"metadata").unwrap();
        }
        provision(&spec, directory.path(), true).unwrap();
        fs::write(directory.path().join(spec.filename()), b"bad").unwrap();
        assert!(
            provision(&spec, directory.path(), true)
                .unwrap_err()
                .to_string()
                .contains("SHA-256")
        );
    }
}
