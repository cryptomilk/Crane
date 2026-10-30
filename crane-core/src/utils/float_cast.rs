// SPDX-License-Identifier: MIT
//! Saturating float-to-integer conversions for sizes derived from float math.
//!
//! Rust's `as` from a float to an integer saturates: negative values and NaN
//! become `0`, values above the integer's range become `MAX`. These helpers
//! name that behaviour once, so call sites need no cast lint suppression.
//!
//! ```
//! use crane_core::utils::float_cast::f64_to_usize;
//!
//! assert_eq!(f64_to_usize((10.0_f64 / 4.0).ceil()), 3);
//! assert_eq!(f64_to_usize(-1.5), 0);
//! ```

/// Converts `value` to `usize`, saturating at `0` and `usize::MAX`.
///
/// NaN converts to `0`. The fractional part is truncated, so round first
/// with `floor`/`ceil`/`round` where a specific rounding is wanted.
#[must_use]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn f64_to_usize(value: f64) -> usize {
    // `as` from float saturates (Rust reference, numeric cast semantics).
    value as usize
}

/// Converts `value` to `usize`, saturating at `0` and `usize::MAX`.
///
/// Same semantics as [`f64_to_usize`].
#[must_use]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn f32_to_usize(value: f32) -> usize {
    // `as` from float saturates (Rust reference, numeric cast semantics).
    value as usize
}

/// Converts `value` to `u32`, saturating at `0` and `u32::MAX`.
///
/// Same semantics as [`f64_to_usize`].
#[must_use]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn f64_to_u32(value: f64) -> u32 {
    // `as` from float saturates (Rust reference, numeric cast semantics).
    value as u32
}

/// Converts `value` to `u32`, saturating at `0` and `u32::MAX`.
///
/// Same semantics as [`f64_to_usize`].
#[must_use]
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn f32_to_u32(value: f32) -> u32 {
    // `as` from float saturates (Rust reference, numeric cast semantics).
    value as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Negative values and NaN clamp to zero instead of wrapping.
    #[test]
    fn negative_and_nan_become_zero() {
        assert_eq!(f64_to_usize(-3.7), 0);
        assert_eq!(f32_to_usize(f32::NAN), 0);
        assert_eq!(f64_to_u32(-1.0), 0);
        assert_eq!(f32_to_u32(f32::NEG_INFINITY), 0);
    }

    // Values past the target range clamp to MAX, and fractions truncate.
    #[test]
    fn overflow_saturates_and_fraction_truncates() {
        assert_eq!(f64_to_u32(1e20), u32::MAX);
        assert_eq!(f32_to_usize(f32::INFINITY), usize::MAX);
        assert_eq!(f64_to_usize(2.9), 2);
        assert_eq!(f32_to_u32(7.0), 7);
    }
}
