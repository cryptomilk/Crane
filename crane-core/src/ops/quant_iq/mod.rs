// SPDX-License-Identifier: MIT

//! Native IQ4_XS / IQ4_NL inference kernels (see `crate::quantized::iquant`).

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "sycl")]
pub mod sycl;
