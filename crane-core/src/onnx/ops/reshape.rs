//! Crane Added 20260731: ONNX Reshape shape resolution.

use candle_core::{Result, Tensor, bail};

/// Applies ONNX's default Reshape semantics.
///
/// A target dimension of zero copies the corresponding input dimension and
/// must participate in the inferred `-1` dimension calculation.
pub(crate) fn reshape(input: &Tensor, target: &[i64]) -> Result<Tensor> {
    let mut shape = Vec::with_capacity(target.len());
    let mut inferred_axis = None;
    let mut known_elements = 1usize;

    for (axis, &dimension) in target.iter().enumerate() {
        match dimension {
            -1 => {
                if inferred_axis.replace(axis).is_some() {
                    bail!("ONNX Reshape target contains more than one -1 dimension");
                }
                shape.push(1);
            },
            0 => {
                let dimension = input.dim(axis)?;
                known_elements = known_elements.checked_mul(dimension).ok_or_else(|| {
                    candle_core::Error::Msg(
                        "ONNX Reshape target element count overflowed".to_string(),
                    )
                })?;
                shape.push(dimension);
            },
            dimension if dimension > 0 => {
                // The match guard above ensures `dimension > 0`.
                #[allow(clippy::cast_sign_loss)]
                let dimension = dimension as usize;
                known_elements = known_elements.checked_mul(dimension).ok_or_else(|| {
                    candle_core::Error::Msg(
                        "ONNX Reshape target element count overflowed".to_string(),
                    )
                })?;
                shape.push(dimension);
            },
            dimension => bail!("ONNX Reshape target contains invalid dimension {dimension}"),
        }
    }

    if let Some(axis) = inferred_axis {
        if known_elements == 0 || input.elem_count() % known_elements != 0 {
            bail!(
                "cannot infer ONNX Reshape dimension for {} elements and known product {known_elements}",
                input.elem_count()
            );
        }
        shape[axis] = input.elem_count() / known_elements;
    }
    input.reshape(shape)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::reshape;

    #[test]
    fn copied_batch_dimension_participates_in_inference() -> Result<()> {
        let input = Tensor::zeros((12, 95, 360), DType::F32, &Device::Cpu)?;
        let output = reshape(&input, &[0, -1, 3, 8, 15])?;
        assert_eq!(output.dims(), &[12, 95, 3, 8, 15]);
        Ok(())
    }
}
