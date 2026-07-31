// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Resize` as a native eval op.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `Resize`: nearest-neighbor or linear interpolation.
///
/// `scales` (input 2) or `sizes` (input 3) determine the output shape --
/// exactly one of the two must be present.
///
/// `mode="nearest"` supports rank-3 and rank-4 tensors only, with
/// `nearest_mode="floor"` and `coordinate_transformation_mode="asymmetric"`.
/// Rank-3 inputs (`[N, C, L]`) resize the trailing dimension via
/// `upsample_nearest1d`; rank-4 inputs (`[N, C, H, W]`) resize the trailing
/// two dimensions via `upsample_nearest2d`.
///
/// `mode="linear"` supports `coordinate_transformation_mode="half_pixel"`
/// (the ONNX default for linear mode) and a `scales` input (not `sizes`)
/// with exactly one axis scaled by a non-`1.0` factor -- see
/// [`linear_resize`] for the single-axis restriction's rationale.
pub(crate) fn resize(
    node: &NodeProto,
    input: &Tensor,
    scales: Option<&Tensor>,
    sizes: Option<&Tensor>,
) -> Result<Tensor> {
    let mode = string_attribute(node, "mode")?.unwrap_or("nearest");
    let coordinate_transformation_mode =
        string_attribute(node, "coordinate_transformation_mode")?.unwrap_or("half_pixel");

    match mode {
        "nearest" => {
            let output_dims = resolve_output_dims(node, input, scales, sizes)?;
            let nearest_mode = string_attribute(node, "nearest_mode")?.unwrap_or("round_prefer_floor");
            nearest_resize(node, input, &output_dims, nearest_mode, coordinate_transformation_mode)
        },
        "linear" => {
            if coordinate_transformation_mode != "half_pixel" {
                bail!(
                    "Resize node '{}': only coordinate_transformation_mode=\"half_pixel\" is \
                     supported for linear-mode resize",
                    node.name
                );
            }
            let Some(scales) = scales else {
                bail!(
                    "Resize node '{}': only a 'scales' input is supported (not 'sizes') for \
                     linear-mode resize",
                    node.name
                );
            };
            linear_resize(node, input, scales)
        },
        _ => bail!("Resize node '{}': unsupported mode '{mode}'", node.name),
    }
}

/// Resolves `scales`/`sizes` (exactly one must be present) into a per-axis
/// output dimension vector, used by [`nearest_resize`].
fn resolve_output_dims(
    node: &NodeProto,
    input: &Tensor,
    scales: Option<&Tensor>,
    sizes: Option<&Tensor>,
) -> Result<Vec<usize>> {
    match (scales, sizes) {
        (Some(_), Some(_)) => {
            bail!(
                "Resize node '{}': scales and sizes cannot both be set",
                node.name
            )
        },
        (Some(scales_tensor), None) => {
            let scale_values = scales_tensor.to_vec1::<f32>()?;
            if scale_values.len() != input.rank() {
                bail!(
                    "Resize node '{}': scales has {} value(s) but input has rank {} (per-axis `axes` scoping is not supported)",
                    node.name,
                    scale_values.len(),
                    input.rank()
                );
            }
            let mut output_dims = Vec::with_capacity(input.rank());
            for (i, &d) in input.dims().iter().enumerate() {
                // Tensor dims are always small non-negative values for real
                // models, so the round-trip through f32 cannot lose
                // meaningful precision here; sign is checked explicitly
                // below rather than relying on the cast.
                #[allow(clippy::cast_precision_loss)]
                let scaled_dim = d as f32 * scale_values[i];
                if scaled_dim < 0.0 {
                    bail!(
                        "Resize node '{}': computed negative output dimension {scaled_dim} at axis {i}",
                        node.name
                    );
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let output_dim = scaled_dim as usize;
                output_dims.push(output_dim);
            }
            Ok(output_dims)
        },
        (None, Some(sizes_tensor)) => {
            let size_values = sizes_tensor.to_vec1::<i64>()?;
            if size_values.len() != input.rank() {
                bail!(
                    "Resize node '{}': sizes has {} value(s) but input has rank {} (per-axis `axes` scoping is not supported)",
                    node.name,
                    size_values.len(),
                    input.rank()
                );
            }
            let mut output_dims = Vec::with_capacity(size_values.len());
            for &d in &size_values {
                if d < 0 {
                    bail!(
                        "Resize node '{}': sizes contains negative value {d}",
                        node.name
                    );
                }
                // Validated non-negative above, so the cast cannot wrap.
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let dim = d as usize;
                output_dims.push(dim);
            }
            Ok(output_dims)
        },
        (None, None) => bail!(
            "Resize node '{}': either scales or sizes must be present",
            node.name
        ),
    }
}

/// ONNX `Resize` with `mode="nearest"`: only `nearest_mode="floor"` and
/// `coordinate_transformation_mode="asymmetric"` are supported. Rank-3
/// inputs (`[N, C, L]`) resize the trailing dimension via
/// `upsample_nearest1d`; rank-4 inputs (`[N, C, H, W]`) resize the trailing
/// two dimensions via `upsample_nearest2d`.
fn nearest_resize(
    node: &NodeProto,
    input: &Tensor,
    output_dims: &[usize],
    nearest_mode: &str,
    coordinate_transformation_mode: &str,
) -> Result<Tensor> {
    if nearest_mode != "floor" {
        bail!(
            "Resize node '{}': unsupported nearest_mode '{nearest_mode}'",
            node.name
        );
    }
    if coordinate_transformation_mode != "asymmetric" {
        bail!(
            "Resize node '{}': unsupported coordinate_transformation_mode '{coordinate_transformation_mode}'",
            node.name
        );
    }

    match input.rank() {
        3 => {
            let target_l = output_dims[2];
            input.upsample_nearest1d(target_l)
        },
        4 => {
            let target_h = output_dims[2];
            let target_w = output_dims[3];
            input.upsample_nearest2d(target_h, target_w)
        },
        rank => bail!(
            "Resize node '{}': unsupported input rank {rank} (expected 3 or 4)",
            node.name
        ),
    }
}

/// ONNX `Resize` with `mode="linear"` and `coordinate_transformation_mode=
/// "half_pixel"`: linear interpolation along a single axis.
///
/// Only one axis may have a non-`1.0` `scales` entry (every other axis is
/// left unchanged) -- this isn't a general N-linear implementation, just
/// what single-axis signal resampling (e.g. upsampling/downsampling a
/// waveform) needs. For output index `i`, the corresponding input
/// coordinate is `(i + 0.5) / scale - 0.5` (boundary-clamped), matching
/// `coordinate_transformation_mode="half_pixel"`; the two neighboring input
/// rows are gathered via `index_select` and blended with `broadcast` ops, so
/// the data never leaves tensor storage for a host-side `Vec` round-trip.
fn linear_resize(node: &NodeProto, input: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let scale_values = scales.to_vec1::<f32>()?;
    if scale_values.len() != input.rank() {
        bail!(
            "Resize node '{}': scales has {} value(s) but input has rank {} (per-axis `axes` scoping is not supported)",
            node.name,
            scale_values.len(),
            input.rank()
        );
    }

    let mut axis = None;
    for (i, &s) in scale_values.iter().enumerate() {
        if (s - 1.0).abs() > 1e-6 {
            if axis.is_some() {
                bail!(
                    "Resize node '{}': more than one axis has a non-1.0 scale {scale_values:?}; \
                     only single-axis linear resize is supported",
                    node.name
                );
            }
            axis = Some(i);
        }
    }
    let Some(axis) = axis else {
        bail!("Resize node '{}': no axis has a non-1.0 scale", node.name);
    };
    let scale = scale_values[axis];

    let in_len = input.dim(axis)?;
    // `scale` and `in_len` are both bounded, ordinary model dimensions (at
    // most a few hundred thousand samples), so this product stays far
    // inside f32's exact-integer range and the truncating cast to usize
    // matches ONNX Resize's `floor(input_dim * scale)` spec.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let out_len = (in_len as f32 * scale) as usize;

    // `in_len` is a real tensor dimension (well under i64::MAX in practice),
    // so this cast never wraps.
    #[allow(clippy::cast_possible_wrap)]
    let max_idx = in_len as i64 - 1;

    let mut floor_idx = Vec::with_capacity(out_len);
    let mut ceil_idx = Vec::with_capacity(out_len);
    let mut fracs = Vec::with_capacity(out_len);
    for i in 0..out_len {
        #[allow(clippy::cast_precision_loss)]
        let coord = (i as f32 + 0.5) / scale - 0.5;
        let f = coord.floor();
        #[allow(clippy::cast_possible_truncation)]
        let f_i64 = f as i64;
        floor_idx.push(f_i64.clamp(0, max_idx));
        ceil_idx.push((f_i64 + 1).clamp(0, max_idx));
        fracs.push(coord - f);
    }

    let dev = input.device();
    let floor_t = Tensor::new(floor_idx, dev)?;
    let ceil_t = Tensor::new(ceil_idx, dev)?;

    let left = input.index_select(&floor_t, axis)?;
    let right = input.index_select(&ceil_t, axis)?;

    let mut frac_shape = vec![1usize; input.rank()];
    frac_shape[axis] = out_len;
    let frac_t = Tensor::new(fracs, dev)?.reshape(frac_shape)?;

    left.broadcast_add(&right.broadcast_sub(&left)?.broadcast_mul(&frac_t)?)
}

fn string_attribute<'a>(node: &'a NodeProto, name: &str) -> Result<Option<&'a str>> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(None);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::String {
        bail!(
            "Resize node '{}' has a non-STRING '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type()
        );
    }
    std::str::from_utf8(&attribute.s)
        .map(Some)
        .map_err(candle_core::Error::wrap)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::resize;

    fn resize_node() -> NodeProto {
        NodeProto {
            name: "Resize.0".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"nearest".to_vec(),
                    ..Default::default()
                },
                AttributeProto {
                    name: "nearest_mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"floor".to_vec(),
                    ..Default::default()
                },
                AttributeProto {
                    name: "coordinate_transformation_mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"asymmetric".to_vec(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // The motivating case: rank-3 `[N, C, L]` inputs used to hard-bail
    // before candle's native `upsample_nearest1d` was wired in.
    #[test]
    fn rank3_scales_upsamples_last_dim() -> Result<()> {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu)?;
        let scales = Tensor::new(&[1.0f32, 1.0, 2.0], &Device::Cpu)?;

        let y = resize(&node, &x, Some(&scales), None)?;

        assert_eq!(y.dims(), &[1, 2, 6]);
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        );
        Ok(())
    }

    // Regression: rank-4 `[N, C, H, W]` resize must keep working exactly as
    // before the rank-3 dispatch was added.
    #[test]
    fn rank4_scales_upsamples_spatial_dims() -> Result<()> {
        let node = resize_node();
        let x = Tensor::new(&[[[[1.0f32, 2.0], [3.0, 4.0]]]], &Device::Cpu)?;
        let scales = Tensor::new(&[1.0f32, 1.0, 2.0, 2.0], &Device::Cpu)?;

        let y = resize(&node, &x, Some(&scales), None)?;

        assert_eq!(y.dims(), &[1, 1, 4, 4]);
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
            ]
        );
        Ok(())
    }

    // Same rank-3 upsample as above, but driven by the `sizes` input
    // instead of `scales`.
    #[test]
    fn rank3_sizes_upsamples_last_dim() -> Result<()> {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu)?;
        let sizes = Tensor::new(&[1i64, 2, 6], &Device::Cpu)?;

        let y = resize(&node, &x, None, Some(&sizes))?;

        assert_eq!(y.dims(), &[1, 2, 6]);
        Ok(())
    }

    #[test]
    fn unsupported_rank_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[2.0f32, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("unsupported input rank"));
    }

    // A `scales` tensor with fewer elements than the input rank (e.g. an
    // opset-18 `axes`-scoped export) must be rejected instead of indexing
    // out of bounds into `scale_values`.
    #[test]
    fn scales_length_mismatch_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("scales"));
    }

    // A `sizes` tensor with fewer elements than the input rank must be
    // rejected instead of indexing out of bounds into `output_dims` later.
    #[test]
    fn sizes_length_mismatch_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let sizes = Tensor::new(&[2i64, 6], &Device::Cpu).unwrap();

        let err = resize(&node, &x, None, Some(&sizes)).unwrap_err();

        assert!(err.to_string().contains("sizes"));
    }

    // A negative `sizes` value must be rejected rather than wrapping to a
    // huge `usize` via `as usize`.
    #[test]
    fn negative_size_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let sizes = Tensor::new(&[1i64, 2, -6], &Device::Cpu).unwrap();

        let err = resize(&node, &x, None, Some(&sizes)).unwrap_err();

        assert!(err.to_string().contains("negative"));
    }

    // A negative scale must be rejected rather than producing a negative
    // computed dimension that wraps to a huge `usize`.
    #[test]
    fn negative_scale_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 1.0, -2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("negative"));
    }

    // Both `scales` and `sizes` present is invalid per the ONNX spec.
    #[test]
    fn both_scales_and_sizes_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 1.0, 2.0], &Device::Cpu).unwrap();
        let sizes = Tensor::new(&[1i64, 2, 6], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), Some(&sizes)).unwrap_err();

        assert!(err.to_string().contains("cannot both be set"));
    }

    // Neither `scales` nor `sizes` present is invalid per the ONNX spec.
    #[test]
    fn neither_scales_nor_sizes_bails() {
        let node = resize_node();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();

        let err = resize(&node, &x, None, None).unwrap_err();

        assert!(err.to_string().contains("must be present"));
    }

    // Only `mode="nearest"` and `mode="linear"` are supported; other
    // interpolation modes must be rejected rather than silently treated as
    // nearest-neighbor.
    #[test]
    fn unsupported_mode_bails() {
        let mut node = resize_node();
        node.attribute[0].s = b"cubic".to_vec();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 1.0, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("unsupported mode"));
    }

    fn linear_resize_node() -> NodeProto {
        NodeProto {
            name: "Resize.0".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"linear".to_vec(),
                    ..Default::default()
                },
                AttributeProto {
                    name: "coordinate_transformation_mode".to_string(),
                    r#type: AttributeType::String as i32,
                    s: b"half_pixel".to_vec(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // Half-pixel linear upsample: coord(i) = (i + 0.5) / 2 - 0.5, e.g. i=0
    // -> coord=-0.25 -> clamps to index 0 -> value 0.0; i=1 -> coord=0.25 ->
    // lerp(0, 10, 0.25) = 2.5.
    #[test]
    fn linear_upsample_doubles_last_axis() -> Result<()> {
        let node = linear_resize_node();
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu)?;
        let scales = Tensor::new(&[1.0f32, 2.0], &Device::Cpu)?;

        let y = resize(&node, &x, Some(&scales), None)?;

        assert_eq!(y.dims(), &[1, 8]);
        let got = y.to_vec2::<f32>()?[0].clone();
        let expected = [0.0f32, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {got:?}, expected {expected:?}");
        }
        Ok(())
    }

    #[test]
    fn linear_downsample_halves_last_axis() -> Result<()> {
        let node = linear_resize_node();
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu)?;
        let scales = Tensor::new(&[1.0f32, 0.5], &Device::Cpu)?;

        let y = resize(&node, &x, Some(&scales), None)?;

        assert_eq!(y.dims(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn linear_rejects_sizes_input() {
        let node = linear_resize_node();
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu).unwrap();
        let sizes = Tensor::new(&[1i64, 8], &Device::Cpu).unwrap();

        let err = resize(&node, &x, None, Some(&sizes)).unwrap_err();

        assert!(err.to_string().contains("'scales' input is supported"));
    }

    #[test]
    fn linear_rejects_multi_axis_scales() {
        let node = linear_resize_node();
        let x = Tensor::new(&[[0f32, 10.], [20., 30.]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[2.0f32, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("more than one axis"));
    }

    #[test]
    fn linear_rejects_non_half_pixel_coordinate_transformation_mode() {
        let mut node = linear_resize_node();
        node.attribute[1].s = b"asymmetric".to_vec();
        let x = Tensor::new(&[[0f32, 10., 20., 30.]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("half_pixel"));
    }
}
