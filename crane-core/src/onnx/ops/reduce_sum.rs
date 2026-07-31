// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `ReduceSum` as a native eval op.

use candle_core::{Result, Tensor};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `ReduceSum`: sums `input` over `axes_input` (opset 13+ second input).
///
/// `keepdims` (attribute, default `1`) keeps reduced axes as size-1 dims.
/// `noop_with_empty_axes` (attribute, default `0`) controls behavior when no
/// axes are provided: `1` returns `input` unchanged, `0` reduces over all
/// axes. Negative axis values are normalized relative to `input`'s rank
/// rather than cast directly to `usize`, which would otherwise wrap to a
/// huge out-of-range index.
pub(crate) fn reduce_sum(
    node: &NodeProto,
    input: &Tensor,
    axes_input: Option<&Tensor>,
) -> Result<Tensor> {
    let keepdims = int_attribute(node, "keepdims", 1)?;
    let noop_with_empty_axes = int_attribute(node, "noop_with_empty_axes", 0)?;

    let axes = match axes_input {
        Some(axes) => axes
            .to_vec1::<i64>()?
            .into_iter()
            .map(|a| input.normalize_axis(a))
            .collect::<Result<Vec<_>>>()?,
        None => {
            if noop_with_empty_axes == 1 {
                vec![]
            } else {
                (0..input.rank()).collect()
            }
        },
    };

    if keepdims == 1 {
        input.sum_keepdim(axes)
    } else {
        input.sum(axes)
    }
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
            "ReduceSum node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(attribute.i)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::reduce_sum;

    fn node(keepdims: i64, noop_with_empty_axes: i64) -> NodeProto {
        NodeProto {
            name: "ReduceSum.0".to_string(),
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

    // Verifies a basic sum over a positive axis with keepdims enabled.
    #[test]
    fn sum_positive_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[1i64], &Device::Cpu)?;
        let node = node(1, 0);

        let y = reduce_sum(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![6.0], vec![15.0]]);
        Ok(())
    }

    // The motivating bug: axis -1 was cast directly to usize, wrapping to a
    // huge out-of-range index instead of resolving to the last axis.
    #[test]
    fn sum_negative_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[-1i64], &Device::Cpu)?;
        let node = node(1, 0);

        let y = reduce_sum(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![6.0], vec![15.0]]);
        Ok(())
    }

    // Verifies keepdims=0 removes the reduced dimension instead of keeping
    // it as size-1.
    #[test]
    fn keepdims_zero() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axes = Tensor::new(&[1i64], &Device::Cpu)?;
        let node = node(0, 0);

        let y = reduce_sum(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2]);
        assert_eq!(y.to_vec1::<f32>()?, vec![6.0, 15.0]);
        Ok(())
    }

    // Verifies noop_with_empty_axes=1 returns the input unchanged when no
    // axes are given.
    #[test]
    fn noop_with_empty_axes() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let node = node(1, 1);

        let y = reduce_sum(&node, &x, None)?;

        assert_eq!(y.dims(), x.dims());
        assert_eq!(y.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        Ok(())
    }
}
