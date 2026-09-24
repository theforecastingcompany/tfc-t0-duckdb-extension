//! CPU ONNX Runtime distribution, pinned independently from the model artifact.
use crate::{
    ForecastError,
    download::{self, Artifact},
};
use flate2::read::GzDecoder;
use std::{
    env, fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};
use tempfile::NamedTempFile;

pub const VERSION: &str = "1.29.0";

pub(crate) struct Platform {
    name: &'static str,
    library: &'static str,
    archive: Artifact,
    files: &'static [Artifact],
}

const LICENSE: Artifact = Artifact {
    name: "LICENSE",
    bytes: 1073,
    sha256: Some("2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"),
};
const PRIVACY: Artifact = Artifact {
    name: "Privacy.md",
    bytes: 4696,
    sha256: Some("9703b86132bbe407a7138f9a45689e6996a8ab0ac670529d3699ce753f67538e"),
};
const NOTICES: Artifact = Artifact {
    name: "ThirdPartyNotices.txt",
    bytes: 336906,
    sha256: Some("53d3fa5821ac016ac24dd35775c996efec86e2ae0841e9a3a5e146c0ae916845"),
};
const MACOS: Platform = Platform {
    name: "osx_arm64",
    archive: Artifact {
        name: "onnxruntime-osx-arm64-1.29.0.tgz",
        bytes: 41_578_864,
        sha256: Some("d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92"),
    },
    library: "libonnxruntime.1.29.0.dylib",
    files: &[
        LICENSE,
        PRIVACY,
        NOTICES,
        Artifact {
            name: "libonnxruntime.1.29.0.dylib",
            bytes: 43_184_400,
            sha256: Some("68f6e54e695583adc371aef610ec4abb1ffaa3df656582922de7690f7e2000eb"),
        },
    ],
};
const LINUX: Platform = Platform {
    name: "linux_amd64",
    archive: Artifact {
        name: "onnxruntime-linux-x64-1.29.0.tgz",
        bytes: 11_082_880,
        sha256: Some("c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba"),
    },
    library: "libonnxruntime.so.1.29.0",
    files: &[
        LICENSE,
        PRIVACY,
        NOTICES,
        Artifact {
            name: "libonnxruntime_providers_shared.so",
            bytes: 14_632,
            sha256: Some("086ec1d5388f64153d9c63470d126693db9a182c8ce236d3a1119068471b8a0d"),
        },
        Artifact {
            name: "libonnxruntime.so.1.29.0",
            bytes: 28_497_752,
            sha256: Some("5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca"),
        },
    ],
};

fn platform(os: &str, arch: &str, gnu: bool) -> Result<&'static Platform, ForecastError> {
    match (os, arch, gnu) {
        ("macos", "aarch64", _) => Ok(&MACOS),
        ("linux", "x86_64", true) => Ok(&LINUX),
        _ => Err(format!("Automatic ONNX Runtime download is unsupported for {os}/{arch}; set ORT_DYLIB_PATH to a compatible native library").into()),
    }
}

/// Explicit user libraries retain their original override semantics. Managed
/// libraries and notices are size/hash-verified on every LOAD, before any dlopen.
pub fn resolve_runtime() -> Result<PathBuf, ForecastError> {
    if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err("ORT_DYLIB_PATH: ONNX Runtime shared library does not exist".into());
        }
        return Ok(fs::canonicalize(path)?);
    }
    let platform = platform(env::consts::OS, env::consts::ARCH, cfg!(target_env = "gnu"))?;
    provision_runtime(&download::cache_root()?, platform, download::offline())
}

fn provision_runtime(
    root: &Path,
    platform: &Platform,
    offline: bool,
) -> Result<PathBuf, ForecastError> {
    let directory = runtime_directory(root, platform);
    let mut complete = true;
    for file in platform.files {
        let path = directory.join(file.name);
        if path.is_file() {
            download::verify_file(&path, file)?;
        } else {
            complete = false;
        }
    }
    if !complete {
        if offline {
            return Err(format!(
                "TFC_OFFLINE: missing ONNX Runtime package files in {}",
                directory.display()
            )
            .into());
        }
        fs::create_dir_all(&directory)?;
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{}",
            platform.archive.name
        );
        let archive = download::download_file(
            &download::download_client()?,
            &url,
            &platform.archive,
            &directory,
        )?;
        extract_archive(archive.path(), platform, &directory)?;
    }
    Ok(fs::canonicalize(directory.join(platform.library))?)
}

fn runtime_directory(root: &Path, platform: &Platform) -> PathBuf {
    root.join("runtimes/onnxruntime")
        .join(VERSION)
        .join(platform.name)
        .join(
            platform
                .archive
                .sha256
                .expect("runtime archives are hash-pinned"),
        )
}

/// The verified archive is never unpacked wholesale. Only exact regular-file
/// paths are copied, with compiled-in size limits; archive links are not followed.
fn extract_archive(
    archive_path: &Path,
    platform: &Platform,
    directory: &Path,
) -> Result<(), ForecastError> {
    download::verify_file(archive_path, &platform.archive)?;
    let mut archive = tar::Archive::new(GzDecoder::new(fs::File::open(archive_path)?));
    let root = platform.archive.name.trim_end_matches(".tgz");
    let mut staged = Vec::new();
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path: PathBuf = entry
            .path()?
            .components()
            .filter(|component| !matches!(component, Component::CurDir))
            .collect();
        let Some(file) = platform.files.iter().find(|file| {
            let relative = if file.name.starts_with("libonnxruntime") {
                Path::new("lib").join(file.name)
            } else {
                PathBuf::from(file.name)
            };
            path == Path::new(root).join(relative)
        }) else {
            continue;
        };
        if !entry.header().entry_type().is_file() || entry.size() != file.bytes {
            return Err(format!("Unexpected runtime archive entry: {}", path.display()).into());
        }
        if staged
            .iter()
            .any(|(other, _): &(&Artifact, NamedTempFile)| other.name == file.name)
        {
            return Err(format!("Duplicate runtime archive entry: {}", file.name).into());
        }
        let mut temporary = NamedTempFile::new_in(directory)?;
        let copied = std::io::copy(&mut (&mut entry).take(file.bytes + 1), &mut temporary)?;
        if copied != file.bytes {
            return Err(format!("Truncated runtime archive entry: {}", file.name).into());
        }
        temporary.flush()?;
        download::verify_file(temporary.path(), file)?;
        temporary.as_file().sync_all()?;
        staged.push((file, temporary));
    }
    if staged.len() != platform.files.len() {
        return Err("Runtime archive is missing required library or licensing files".into());
    }
    // Publish only after every selected file has passed verification.
    for (file, temporary) in staged {
        download::publish_file(temporary, &directory.join(file.name), file)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_only_supported_native_platforms() {
        assert_eq!(
            platform("macos", "aarch64", false).unwrap().name,
            "osx_arm64"
        );
        assert_eq!(
            platform("linux", "x86_64", true).unwrap().name,
            "linux_amd64"
        );
        for (os, arch, gnu) in [
            ("linux", "x86_64", false),
            ("linux", "aarch64", true),
            ("macos", "x86_64", false),
            ("windows", "x86_64", false),
        ] {
            assert!(platform(os, arch, gnu).is_err());
        }
    }

    #[test]
    fn offline_runtime_requires_complete_verified_package() {
        let directory = tempfile::tempdir().unwrap();
        assert!(
            provision_runtime(directory.path(), &MACOS, true)
                .unwrap_err()
                .to_string()
                .contains("TFC_OFFLINE")
        );
        let cache = runtime_directory(directory.path(), &MACOS);
        fs::create_dir_all(&cache).unwrap();
        fs::write(cache.join(MACOS.library), b"corrupt runtime").unwrap();
        assert!(
            provision_runtime(directory.path(), &MACOS, true)
                .unwrap_err()
                .to_string()
                .contains("SHA-256")
        );
    }
    const FIXTURE_FILE: Artifact = Artifact {
        name: "libonnxruntime.fixture",
        bytes: 3,
        sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    };

    fn fixture(directory: &Path, entries: &[(&str, &[u8], tar::EntryType)]) -> (PathBuf, Platform) {
        use sha2::{Digest, Sha256};
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for (path, bytes, kind) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(*kind);
            header.set_size(bytes.len() as u64);
            header.set_mode(0o600);
            if kind.is_symlink() {
                header.set_link_name("../../outside").unwrap();
            }
            // Deliberately permit adversarial paths in these synthetic fixtures.
            header.as_mut_bytes()[..100].fill(0);
            header.as_mut_bytes()[..path.len()].copy_from_slice(path.as_bytes());
            header.set_cksum();
            builder.append(&header, *bytes).unwrap();
        }
        let bytes = builder.into_inner().unwrap().finish().unwrap();
        let sha = Box::leak(format!("{:x}", Sha256::digest(&bytes)).into_boxed_str());
        let path = directory.join("fixture.tgz");
        fs::write(&path, &bytes).unwrap();
        (
            path,
            Platform {
                name: "fixture",
                library: FIXTURE_FILE.name,
                files: &[FIXTURE_FILE],
                archive: Artifact {
                    name: "fixture.tgz",
                    bytes: bytes.len() as u64,
                    sha256: Some(sha),
                },
            },
        )
    }

    #[test]
    fn extract_only_whitelisted_files_and_ignore_traversal() {
        let directory = tempfile::tempdir().unwrap();
        let (archive, platform) = fixture(
            directory.path(),
            &[
                ("../outside", b"unrelated", tar::EntryType::Regular),
                ("/absolute", b"unrelated", tar::EntryType::Regular),
                (
                    "./fixture/lib/libonnxruntime.fixture",
                    b"abc",
                    tar::EntryType::Regular,
                ),
            ],
        );
        let output = directory.path().join("output");
        fs::create_dir(&output).unwrap();
        extract_archive(&archive, &platform, &output).unwrap();
        assert_eq!(fs::read(output.join(FIXTURE_FILE.name)).unwrap(), b"abc");
        assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
        assert!(!directory.path().join("outside").exists());
    }

    #[test]
    fn reject_bad_entries_without_publishing_partial_files() {
        let cases: &[&[(&str, &[u8], tar::EntryType)]] = &[
            &[(
                "fixture/lib/libonnxruntime.fixture",
                b"",
                tar::EntryType::Symlink,
            )],
            &[(
                "fixture/lib/libonnxruntime.fixture",
                b"bad",
                tar::EntryType::Regular,
            )],
            &[(
                "fixture/lib/libonnxruntime.fixture",
                b"abcd",
                tar::EntryType::Regular,
            )],
            &[("fixture/unexpected", b"abc", tar::EntryType::Regular)],
            &[
                (
                    "fixture/lib/libonnxruntime.fixture",
                    b"abc",
                    tar::EntryType::Regular,
                ),
                (
                    "fixture/lib/libonnxruntime.fixture",
                    b"abc",
                    tar::EntryType::Regular,
                ),
            ],
        ];
        for entries in cases {
            let directory = tempfile::tempdir().unwrap();
            let (archive, platform) = fixture(directory.path(), entries);
            let output = directory.path().join("output");
            fs::create_dir(&output).unwrap();
            assert!(extract_archive(&archive, &platform, &output).is_err());
            assert_eq!(fs::read_dir(output).unwrap().count(), 0);
        }
    }

    #[test]
    fn reject_tampered_archive_before_extraction() {
        let directory = tempfile::tempdir().unwrap();
        let (archive, platform) = fixture(directory.path(), &[]);
        fs::write(&archive, b"not the pinned archive").unwrap();
        assert!(
            extract_archive(&archive, &platform, directory.path())
                .unwrap_err()
                .to_string()
                .contains("SHA-256")
        );
    }

    #[test]
    #[ignore = "requires the two official release archives in TFC_TEST_RUNTIME_ARCHIVES"]
    fn official_archive_fixtures() {
        let root = PathBuf::from(env::var_os("TFC_TEST_RUNTIME_ARCHIVES").unwrap());
        for platform in [&MACOS, &LINUX] {
            let output = tempfile::tempdir().unwrap();
            extract_archive(&root.join(platform.archive.name), platform, output.path()).unwrap();
            for file in platform.files {
                download::verify_file(&output.path().join(file.name), file).unwrap();
            }
            assert_eq!(
                fs::read_dir(output.path()).unwrap().count(),
                platform.files.len()
            );
        }
    }
}
