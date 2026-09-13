// SPDX-License-Identifier: MIT

//! Fetches test fixtures from the `crane-local-ai/test-data` `HuggingFace`
//! dataset, for use by `#[cfg(test)]` modules and `benches/`.
//!
//! Test data files (G2P word/IPA dictionaries, etc.) are hosted on
//! `HuggingFace` rather than checked into the repository, since some grow too
//! large for a git checkout as more languages are added.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use hf_hub::{HFClientSync, split_id};

/// Name of the environment variable pointing at a local checkout of the
/// `crane-local-ai/test-data` dataset, bypassing the `HuggingFace` download.
pub const CRANE_TEST_DATA_DIR_ENV: &str = "CRANE_TEST_DATA_DIR";

/// `HuggingFace` dataset repo hosting Crane's test fixtures.
const CRANE_TEST_DATA_REPO: &str = "crane-local-ai/test-data";

/// Returns the local path to `path` within the `crane-local-ai/test-data`
/// `HuggingFace` dataset.
///
/// If the `CRANE_TEST_DATA_DIR` environment variable is set, `path` is
/// resolved relative to it (for CI or a pre-downloaded checkout). Otherwise
/// the file is fetched from the `HuggingFace` Hub, using the hub's local
/// cache on repeat calls.
///
/// # Errors
///
/// Returns an error if `CRANE_TEST_DATA_DIR` is set but `path` doesn't exist
/// under it, or if the `HuggingFace` API client cannot be created or the
/// file cannot be downloaded (e.g. no network access and no local checkout).
pub fn get_test_data_file(path: &str) -> Result<PathBuf> {
    if let Ok(dir) = std::env::var(CRANE_TEST_DATA_DIR_ENV) {
        let file = PathBuf::from(dir).join(path);
        if !file.is_file() {
            bail!(
                "{CRANE_TEST_DATA_DIR_ENV} is set but {} does not exist",
                file.display()
            );
        }
        return Ok(file);
    }

    let (owner, name) = split_id(CRANE_TEST_DATA_REPO);
    let file = HFClientSync::new()?
        .dataset(owner, name)
        .download_file()
        .filename(path)
        .send()?;
    Ok(file)
}

/// Loads and parses `g2p/kokoro_vocab.json` from the test-data dataset into
/// a `char -> token id` map.
///
/// # Errors
///
/// Returns an error if the dataset file can't be fetched (see
/// [`get_test_data_file`]) or read, or if it contains a key that isn't
/// exactly one Unicode codepoint.
pub fn load_kokoro_vocab() -> Result<HashMap<char, i64>> {
    let path = get_test_data_file("g2p/kokoro_vocab.json")?;
    let json = std::fs::read_to_string(path)?;
    let raw: HashMap<String, i64> = serde_json::from_str(&json)?;
    raw.into_iter()
        .map(|(k, v)| {
            let mut chars = k.chars();
            let c = chars
                .next()
                .ok_or_else(|| anyhow::anyhow!("vocab key must not be empty"))?;
            if chars.next().is_some() {
                bail!("vocab key {k:?} is not a single codepoint");
            }
            Ok((c, v))
        })
        .collect()
}
