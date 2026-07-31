// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `CumSum` as a native eval op.

use candle_core::{DType, IndexOp, Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `CumSum`: cumulative sum of `input` along the axis given by
/// `axis_tensor` (the op's required second input).
///
/// `exclusive` and `reverse` (attributes, default `0`) are not supported and
/// cause a bail if non-zero. Negative axis values are normalized relative to
/// `input`'s rank rather than cast directly to `usize`, which would
/// otherwise wrap to a huge out-of-range index. `candle_core::Tensor::cumsum`
/// is implemented via `matmul` against a triangular ones-matrix, and
/// `candle-core`'s `matmul` kernel only supports floating-point dtypes, even
/// though ONNX's `CumSum` spec allows integer inputs (e.g. `int64`) -- so
/// integer inputs are cast to `F64`, summed, then cast back. `F64` (exact up
/// to 2^53) rather than `F32` (exact only up to 2^24) is used so that
/// cumulative sums of realistic integer magnitudes don't silently lose
/// precision.
pub(crate) fn cumsum(node: &NodeProto, input: &Tensor, axis_tensor: &Tensor) -> Result<Tensor> {
    let exclusive = int_attribute(node, "exclusive", 0)?;
    if exclusive != 0 {
        bail!("CumSum node '{}' only supports exclusive == 0", node.name)
    }
    let reverse = int_attribute(node, "reverse", 0)?;
    if reverse != 0 {
        bail!("CumSum node '{}' only supports reverse == 0", node.name)
    }

    let axis = input.normalize_axis(scalar_i64(axis_tensor)?)?;

    if input.dtype().is_int() {
        let orig_dtype = input.dtype();
        input
            .to_dtype(DType::F64)?
            .cumsum(axis)?
            .to_dtype(orig_dtype)
    } else {
        input.cumsum(axis)
    }
}

fn scalar_i64(t: &Tensor) -> Result<i64> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_vec0::<i64>()
    } else {
        t.to_vec0::<i64>()
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
        bail!(
            "CumSum node '{}' has a non-INT '{}' attribute ({:?})",
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

    use super::cumsum;

    fn node() -> NodeProto {
        NodeProto {
            name: "CumSum.0".to_string(),
            ..Default::default()
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            r#type: AttributeType::Int as i32,
            i: value,
            ..Default::default()
        }
    }

    // Verifies a basic cumulative sum over a positive axis.
    #[test]
    fn cumsum_float_positive_axis() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axis = Tensor::new(1i64, &Device::Cpu)?;

        let y = cumsum(&node(), &x, &axis)?;

        assert_eq!(
            y.to_vec2::<f32>()?,
            vec![vec![1.0, 3.0, 6.0], vec![4.0, 9.0, 15.0]]
        );
        Ok(())
    }

    // Cumulative sum over axis 0 instead of the trailing axis.
    #[test]
    fn cumsum_float_axis_zero() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axis = Tensor::new(0i64, &Device::Cpu)?;

        let y = cumsum(&node(), &x, &axis)?;

        assert_eq!(
            y.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0, 3.0], vec![5.0, 7.0, 9.0]]
        );
        Ok(())
    }

    // The motivating bug: axis -1 was cast directly to usize, wrapping to a
    // huge out-of-range index instead of resolving to the last axis.
    #[test]
    fn cumsum_float_negative_axis() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axis = Tensor::new(-1i64, &Device::Cpu)?;

        let y = cumsum(&node(), &x, &axis)?;

        assert_eq!(
            y.to_vec2::<f32>()?,
            vec![vec![1.0, 3.0, 6.0], vec![4.0, 9.0, 15.0]]
        );
        Ok(())
    }

    // The other motivating bug: candle's cumsum uses matmul internally,
    // which only supports floating-point dtypes, so int64 input used to
    // fail outright.
    #[test]
    fn cumsum_int64() -> Result<()> {
        let x = Tensor::new(&[1i64, 2, 3, 4], &Device::Cpu)?;
        let axis = Tensor::new(0i64, &Device::Cpu)?;

        let y = cumsum(&node(), &x, &axis)?;

        assert_eq!(y.dtype(), candle_core::DType::I64);
        assert_eq!(y.to_vec1::<i64>()?, vec![1, 3, 6, 10]);
        Ok(())
    }

    // Axis given as a rank-1, single-element tensor (e.g. shape [1]), as
    // commonly produced by ONNX exporters, should normalize the same way as
    // a rank-0 scalar axis tensor.
    #[test]
    fn cumsum_axis_rank1() -> Result<()> {
        let x = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
        let axis = Tensor::new(&[-1i64], &Device::Cpu)?;

        let y = cumsum(&node(), &x, &axis)?;

        assert_eq!(
            y.to_vec2::<f32>()?,
            vec![vec![1.0, 3.0, 6.0], vec![4.0, 9.0, 15.0]]
        );
        Ok(())
    }

    // Verifies that a non-zero `exclusive` attribute is rejected rather than
    // silently ignored.
    #[test]
    fn exclusive_nonzero_is_rejected() -> Result<()> {
        let x = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let axis = Tensor::new(0i64, &Device::Cpu)?;
        let mut n = node();
        n.attribute.push(int_attr("exclusive", 1));

        let err = cumsum(&n, &x, &axis).expect_err("non-zero exclusive should be rejected");
        assert!(err.to_string().contains("exclusive"));
        Ok(())
    }

    // Verifies that a non-zero `reverse` attribute is rejected rather than
    // silently ignored.
    #[test]
    fn reverse_nonzero_is_rejected() -> Result<()> {
        let x = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        let axis = Tensor::new(0i64, &Device::Cpu)?;
        let mut n = node();
        n.attribute.push(int_attr("reverse", 1));

        let err = cumsum(&n, &x, &axis).expect_err("non-zero reverse should be rejected");
        assert!(err.to_string().contains("reverse"));
        Ok(())
    }
}
