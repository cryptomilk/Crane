// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Resize` as a native eval op.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `Resize`: nearest-neighbor interpolation for rank-3 and rank-4 tensors.
///
/// `scales` (input 2) or `sizes` (input 3) determine the output shape --
/// exactly one of the two must be present. Only `mode="nearest"`,
/// `nearest_mode="floor"`, and `coordinate_transformation_mode="asymmetric"`
/// are supported. Rank-3 inputs (`[N, C, L]`) resize the trailing dimension
/// via `upsample_nearest1d`; rank-4 inputs (`[N, C, H, W]`) resize the
/// trailing two dimensions via `upsample_nearest2d`.
pub(crate) fn resize(
    node: &NodeProto,
    input: &Tensor,
    scales: Option<&Tensor>,
    sizes: Option<&Tensor>,
) -> Result<Tensor> {
    let output_dims = match (scales, sizes) {
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
            output_dims
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
            output_dims
        },
        (None, None) => bail!(
            "Resize node '{}': either scales or sizes must be present",
            node.name
        ),
    };

    let mode = string_attribute(node, "mode")?.unwrap_or("nearest");
    let nearest_mode = string_attribute(node, "nearest_mode")?.unwrap_or("round_prefer_floor");
    let coordinate_transformation_mode =
        string_attribute(node, "coordinate_transformation_mode")?.unwrap_or("half_pixel");

    if mode != "nearest" {
        bail!("Resize node '{}': unsupported mode '{mode}'", node.name);
    }
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

    // Only `mode="nearest"` is supported; other interpolation modes must be
    // rejected rather than silently treated as nearest-neighbor.
    #[test]
    fn unsupported_mode_bails() {
        let mut node = resize_node();
        node.attribute[0].s = b"linear".to_vec();
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[1.0f32, 1.0, 2.0], &Device::Cpu).unwrap();

        let err = resize(&node, &x, Some(&scales), None).unwrap_err();

        assert!(err.to_string().contains("unsupported mode"));
    }
}
