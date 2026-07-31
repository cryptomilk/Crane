// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `ReduceMean` as a native eval op.

use candle_core::{Result, Tensor};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `ReduceMean`: averages `input` over the given axes.
///
/// Axes are resolved from `axes_input` (opset 13+ second input) if present,
/// falling back to the `axes` attribute (opset 13 and below) otherwise.
/// `keepdims` (attribute, default `1`) keeps reduced axes as size-1 dims.
/// `noop_with_empty_axes` (attribute, default `0`) controls behavior when no
/// axes are provided at all: `1` returns `input` unchanged, `0` reduces over
/// all axes. Negative axis values are normalized relative to `input`'s rank
/// rather than cast directly to `usize`, which would otherwise wrap to a
/// huge out-of-range index.
pub(crate) fn reduce_mean(
    node: &NodeProto,
    input: &Tensor,
    axes_input: Option<&Tensor>,
) -> Result<Tensor> {
    let keepdims = int_attribute(node, "keepdims", 1)?;
    let noop_with_empty_axes = int_attribute(node, "noop_with_empty_axes", 0)?;

    let axes = match axes_input {
        Some(axes) => normalize_axes(input, axes.to_vec1::<i64>()?)?,
        None => match ints_attribute(node, "axes")? {
            Some(axes) => normalize_axes(input, axes)?,
            None => {
                if noop_with_empty_axes == 1 {
                    vec![]
                } else {
                    (0..input.rank()).collect()
                }
            },
        },
    };

    if keepdims == 1 {
        input.mean_keepdim(axes)
    } else {
        input.mean(axes)
    }
}

fn normalize_axes(input: &Tensor, axes: Vec<i64>) -> Result<Vec<usize>> {
    axes.into_iter().map(|a| input.normalize_axis(a)).collect()
}

fn int_attribute(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Int {
        candle_core::bail!(
            "ReduceMean node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(attribute.i)
}

fn ints_attribute(node: &NodeProto, name: &str) -> Result<Option<Vec<i64>>> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(None);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Ints {
        candle_core::bail!(
            "ReduceMean node '{}' has a non-INTS '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(Some(attribute.ints.clone()))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::reduce_mean;

    fn node(keepdims: i64, noop_with_empty_axes: i64) -> NodeProto {
        NodeProto {
            name: "ReduceMean.0".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "keepdims".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: keepdims,
                    ..Default::default()
                },
                AttributeProto {
                    name: "noop_with_empty_axes".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: noop_with_empty_axes,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // Verifies a basic mean over a positive axis with keepdims enabled,
    // with axes provided as the opset-18+ second input tensor.
    #[test]
    fn mean_positive_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[1i64], &Device::Cpu)?;
        let node = node(1, 0);

        let y = reduce_mean(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![2.0], vec![5.0]]);
        Ok(())
    }

    // The motivating bug: axis -1 provided via the opset-18+ input tensor
    // must resolve to the last axis, not wrap to a huge index.
    #[test]
    fn mean_negative_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[-1i64], &Device::Cpu)?;
        let node = node(1, 0);

        let y = reduce_mean(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![2.0], vec![5.0]]);
        Ok(())
    }

    // Verifies keepdims=0 removes the reduced dimension instead of keeping
    // it as size-1.
    #[test]
    fn keepdims_zero() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[1i64], &Device::Cpu)?;
        let node = node(0, 0);

        let y = reduce_mean(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2]);
        assert_eq!(y.to_vec1::<f32>()?, vec![2.0, 5.0]);
        Ok(())
    }

    // Verifies noop_with_empty_axes=1 returns the input unchanged when no
    // axes are given via either the input tensor or the attribute.
    #[test]
    fn noop_with_empty_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let node = node(1, 1);

        let y = reduce_mean(&node, &x, None)?;

        assert_eq!(y.dims(), x.dims());
        assert_eq!(y.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        Ok(())
    }

    // Verifies the opset-13 fallback: axes provided as a node attribute
    // (no second input tensor) are still honored.
    #[test]
    fn axes_from_attribute() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let mut node = node(1, 0);
        node.attribute.push(AttributeProto {
            name: "axes".to_string(),
            r#type: AttributeType::Ints as i32,
            ints: vec![-1],
            ..Default::default()
        });

        let y = reduce_mean(&node, &x, None)?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![2.0], vec![5.0]]);
        Ok(())
    }
}
