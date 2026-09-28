// SPDX-License-Identifier: MIT

//! Shared GGUF quantized-weight loading infrastructure, used by every model
//! that supports loading from a GGUF checkpoint (`hunyuan_dense`, `gemma4`,
//! `qwen3`, `qwen3_5`, `minicpm5`, `minicpmo`).

pub mod extended_gguf;
pub mod gguf_file;
pub mod iquant;
pub mod ternary;
