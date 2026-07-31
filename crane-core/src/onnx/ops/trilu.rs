// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `Trilu` as a native eval op.

use candle_core::{DType, IndexOp, Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `Trilu`: zeroes out the upper or lower triangular part of `input`'s
/// trailing two dimensions.
///
/// `upper` (attribute, default `1`) selects upper (`1`) vs lower (`0`)
/// triangular; `k_input` (optional second input, default `0`) shifts the
/// diagonal. Batched inputs (rank > 2) apply the same 2D mask to every
/// leading dimension via broadcasting. The triangle is selected with
/// `where_cond` rather than multiplied by a 0/1 mask, so `+/-inf` inputs
/// (e.g. causal attention masks) become `0` outside the triangle instead of
/// `NaN`.
pub(crate) fn trilu(node: &NodeProto, input: &Tensor, k_input: Option<&Tensor>) -> Result<Tensor> {
    let k = match k_input {
        Some(k) => scalar_i64(k)?,
        None => 0,
    };
    let upper = int_attribute(node, "upper", 1)?;

    let dims = input.dims();
    if dims.len() < 2 {
        bail!("Trilu expects input with at least 2 dimensions: {:?}", dims);
    }
    let n = dims[dims.len() - 2];
    let m = dims[dims.len() - 1];
    let max_dim = std::cmp::max(n, m);

    // Build a U8 0/1 mask used as a where_cond condition (not multiplied
    // against input, so it doesn't need input's dtype).
    let mask = if k != 0 {
        let mut data = vec![0u8; n * m];
        for i in 0..n {
            for j in 0..m {
                // i and j are matrix indices bounded by n/m, always far
                // below i64::MAX; the cast cannot wrap in practice.
                #[allow(clippy::cast_possible_wrap)]
                let (signed_i, signed_j) = (i as i64, j as i64);
                if (upper != 0 && signed_j >= signed_i + k) || (upper == 0 && signed_j <= signed_i + k) {
                    data[i * m + j] = 1u8;
                }
            }
        }
        Tensor::from_vec(data, (n, m), input.device())?
    } else if upper == 0 {
        Tensor::tril2(max_dim, DType::U8, input.device())?
    } else {
        Tensor::triu2(max_dim, DType::U8, input.device())?
    };

    let final_mask = if n == m {
        mask
    } else {
        mask.narrow(0, 0, n)?.narrow(1, 0, m)?
    };

    let zeros = Tensor::zeros(input.dims(), input.dtype(), input.device())?;
    final_mask.broadcast_as(input.dims())?.where_cond(input, &zeros)
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
            "Trilu node '{}' has a non-INT '{}' attribute ({:?})",
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

    use super::trilu;

    fn node_with_upper(upper: i64) -> NodeProto {
        NodeProto {
            name: "Trilu.0".to_string(),
            attribute: vec![AttributeProto {
                name: "upper".to_string(),
                r#type: AttributeType::Int as i32,
                i: upper,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // The motivating case: the old `input * mask` implementation produced
    // `NaN` (not `0`) wherever the mask is `0` and `input` is `-inf` --
    // exactly the standard causal-attention-mask pattern.
    #[test]
    fn upper_avoids_nan_with_infinite_input() -> Result<()> {
        let neg_inf = f32::NEG_INFINITY;
        let x = Tensor::new(&[[neg_inf; 3], [neg_inf; 3], [neg_inf; 3]], &Device::Cpu)?;
        let node = node_with_upper(1);

        let y = trilu(&node, &x, None)?.to_vec2::<f32>()?;

        assert_eq!(y[0], vec![neg_inf, neg_inf, neg_inf]);
        assert_eq!(y[1], vec![0.0, neg_inf, neg_inf]);
        assert_eq!(y[2], vec![0.0, 0.0, neg_inf]);
        Ok(())
    }

    #[test]
    fn lower_avoids_nan_with_infinite_input() -> Result<()> {
        let neg_inf = f32::NEG_INFINITY;
        let x = Tensor::new(&[[neg_inf; 3], [neg_inf; 3], [neg_inf; 3]], &Device::Cpu)?;
        let node = node_with_upper(0);

        let y = trilu(&node, &x, None)?.to_vec2::<f32>()?;

        assert_eq!(y[0], vec![neg_inf, 0.0, 0.0]);
        assert_eq!(y[1], vec![neg_inf, neg_inf, 0.0]);
        assert_eq!(y[2], vec![neg_inf, neg_inf, neg_inf]);
        Ok(())
    }

    // Verifies a k=1 diagonal offset (keep strictly above the diagonal)
    // with -inf input still avoids NaN.
    #[test]
    fn upper_with_diagonal_offset_avoids_nan() -> Result<()> {
        let neg_inf = f32::NEG_INFINITY;
        let x = Tensor::new(&[[neg_inf; 3], [neg_inf; 3], [neg_inf; 3]], &Device::Cpu)?;
        let k = Tensor::new(1i64, &Device::Cpu)?;
        let node = node_with_upper(1);

        let y = trilu(&node, &x, Some(&k))?.to_vec2::<f32>()?;

        assert_eq!(y[0], vec![0.0, neg_inf, neg_inf]);
        assert_eq!(y[1], vec![0.0, 0.0, neg_inf]);
        assert_eq!(y[2], vec![0.0, 0.0, 0.0]);
        Ok(())
    }
}
