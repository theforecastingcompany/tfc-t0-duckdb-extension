//! Verified native-runtime downloads and shared cache utilities.
use crate::ForecastError;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tempfile::NamedTempFile;

/// Native runtime archive entries have a pinned checksum and exact byte size.
#[derive(Clone, Copy)]
pub(crate) struct Artifact {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: Option<&'static str>,
}

pub(crate) fn cache_root() -> Result<PathBuf, ForecastError> {
    if let Some(path) = env::var_os("TFC_CACHE_DIR") {
        Ok(PathBuf::from(path))
    } else if let Some(path) = env::var_os("XDG_CACHE_HOME").or_else(|| env::var_os("LOCALAPPDATA"))
    {
        Ok(PathBuf::from(path).join("tfc-forecast"))
    } else {
        Ok(PathBuf::from(
            env::var_os("HOME").ok_or("Set TFC_CACHE_DIR or explicit artifact paths")?,
        )
        .join(".cache/tfc-forecast"))
    }
}

pub(crate) fn offline() -> bool {
    env::var("TFC_OFFLINE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<(), ForecastError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    if format!("{:x}", hasher.finalize()) != expected {
        return Err(format!("artifact SHA-256 mismatch: {}", path.display()).into());
    }
    Ok(())
}

pub(crate) fn verify_file(path: &Path, file: &Artifact) -> Result<(), ForecastError> {
    if let Some(hash) = file.sha256 {
        verify_sha256(path, hash)?;
        if path.metadata()?.len() != file.bytes {
            return Err(format!("Unexpected artifact size: {}", path.display()).into());
        }
    }
    Ok(())
}

/// A shared HTTPS client; archive downloads also use the same timeout policy.
pub(crate) fn download_client() -> Result<Client, ForecastError> {
    Ok(Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(900))
        .user_agent("tfc-duckdb-extension/0.1.0")
        .build()?)
}

pub(crate) fn download_file(
    client: &Client,
    url: &str,
    file: &Artifact,
    directory: &Path,
) -> Result<NamedTempFile, ForecastError> {
    // This client is only used for Microsoft's public runtime archives. It has
    // no Hugging Face credentials; hf-hub handles authenticated model requests.
    let response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| {
            format!(
                "Artifact download of {} failed: {}",
                file.name,
                e.without_url()
            )
        })?;
    let mut temporary = NamedTempFile::new_in(directory)?;
    let copied = std::io::copy(&mut response.take(file.bytes + 1), &mut temporary)?;
    if copied > file.bytes || (file.sha256.is_some() && copied != file.bytes) {
        return Err(format!("Unexpected size for artifact {}: {copied}", file.name).into());
    }
    temporary.flush()?;
    verify_file(temporary.path(), file)?;
    temporary.as_file().sync_all()?;
    Ok(temporary)
}

pub(crate) fn publish_file(
    temporary: NamedTempFile,
    destination: &Path,
    file: &Artifact,
) -> Result<(), ForecastError> {
    if let Err(error) = temporary.persist_noclobber(destination) {
        if error.error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(error.error.into());
        }
        verify_file(destination, file)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_client_requests_have_no_authorization() {
        let request = download_client()
            .unwrap()
            .get("https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/runtime.tgz")
            .build()
            .unwrap();
        assert!(
            !request
                .headers()
                .contains_key(reqwest::header::AUTHORIZATION)
        );
    }
}
