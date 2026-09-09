//! Qwen 3.5 image preprocessor.
//!
//! Mirrors `Qwen2VLImageProcessor` from HF Transformers: smart-resize to a
//! shape that's a multiple of `(patch_size * spatial_merge_size)` and whose
//! total pixel count sits in `[min_pixels, max_pixels]`, then convert to
//! float32 CHW and normalize by `image_mean` / `image_std`. The result is
//! reshaped into
//! `[num_patches, temporal_patch_size * patch_size * patch_size * in_channels]`
//! row-major, ready to feed into `Qwen3_5VisionModel`.
//!
//! Video input is intentionally **not** supported in this MVP — that requires
//! frame extraction (ffmpeg) and temporal batching. The single-image path
//! duplicates the frame `temporal_patch_size` times to match the expected
//! input shape, which is what HF's processor does for a single still image.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use serde::Deserialize;

/// Mirror of HF `preprocessor_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub size: Size,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Size {
    pub shortest_edge: usize,
    pub longest_edge: usize,
}

/// Result of preprocessing one image: a row-major `[num_patches, in_dim]`
/// tensor plus the `(t, h, w)` grid for the text-model `MRoPE` bookkeeping.
pub struct ProcessedImage {
    pub pixel_values: Tensor,
    pub grid_thw: (u32, u32, u32),
}

/// Load `preprocessor_config.json` next to a Qwen 3.5 multimodal checkpoint.
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not parse as a valid config.
pub fn load_preprocessor_config(model_dir: &str) -> Result<PreprocessorConfig> {
    let path = std::path::Path::new(model_dir).join("preprocessor_config.json");
    let data = std::fs::read(&path)
        .with_context(|| format!("read preprocessor_config.json at {}", path.display()))?;
    let cfg: PreprocessorConfig =
        serde_json::from_slice(&data).with_context(|| format!("parse {}", path.display()))?;
    Ok(cfg)
}

/// Smart-resize an image so that both dimensions are multiples of `factor`
/// and the total pixel count sits in `[min_pixels, max_pixels]`.
///
/// Mirrors HF `Qwen2VL`'s `smart_resize` exactly:
///   1. Round h, w to the *nearest* multiple of `factor` (HF `round_by_factor`,
///      i.e. `round(x / factor) * factor` — NOT a ceil).
///   2. If that exceeds `max_pixels`, scale DOWN by
///      `beta = sqrt((h*w)/max_pixels)`, then floor to a multiple of factor.
///   3. If it is below `min_pixels`, scale UP by
///      `beta = sqrt(min_pixels/(h*w))`, then ceil to a multiple of factor.
fn smart_resize(h: u32, w: u32, factor: u32, min_pixels: usize, max_pixels: usize) -> (u32, u32) {
    let h_f = f64::from(h);
    let w_f = f64::from(w);
    let factor_f = f64::from(factor);

    let round_by_factor = |x: f64| {
        // Pixel dimensions are always non-negative and far below u32::MAX,
        // so round()-then-cast cannot truncate or lose sign here.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rounded = (x / factor_f).round() as u32;
        rounded.max(1) * factor
    };
    let mut h_bar = round_by_factor(h_f);
    let mut w_bar = round_by_factor(w_f);

    // u64 so the product can't overflow on large inputs.
    let area = u64::from(h_bar) * u64::from(w_bar);
    if area > max_pixels as u64 {
        // max_pixels/min_pixels come from preprocessor_config.json and are
        // always realistic pixel-count bounds (well under 2^52), so this is
        // lossless in practice.
        #[allow(clippy::cast_precision_loss)]
        let max_pixels_f = max_pixels as f64;
        let beta = (h_f * w_f / max_pixels_f).sqrt();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            h_bar = (h_f / beta / factor_f).floor() as u32 * factor;
            w_bar = (w_f / beta / factor_f).floor() as u32 * factor;
        }
    } else if area < min_pixels as u64 {
        #[allow(clippy::cast_precision_loss)]
        let min_pixels_f = min_pixels as f64;
        let beta = (min_pixels_f / (h_f * w_f)).sqrt();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            h_bar = (h_f * beta / factor_f).ceil() as u32 * factor;
            w_bar = (w_f * beta / factor_f).ceil() as u32 * factor;
        }
    }

    (h_bar.max(factor), w_bar.max(factor))
}

impl PreprocessorConfig {
    #[must_use]
    pub fn factor(&self) -> u32 {
        // patch_size * merge_size is a small config constant (e.g. 16*2=32),
        // far below u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let factor = (self.patch_size * self.merge_size) as u32;
        factor
    }

    /// Resize `image` to multiples of `factor` while keeping pixels in bounds.
    #[must_use]
    pub fn smart_resize(&self, image: &DynamicImage) -> DynamicImage {
        let (w, h) = image.dimensions();
        let (h_new, w_new) = smart_resize(
            h,
            w,
            self.factor(),
            self.size.shortest_edge,
            self.size.longest_edge,
        );
        if (h_new, w_new) == (h, w) {
            image.clone()
        } else {
            // HF's preprocessor_config sets `resample: 3` = PIL BICUBIC, whose
            // kernel (a = -0.5) is Catmull-Rom. Lanczos3 here shifted every
            // pixel by up to 20/255 against the reference.
            image.resize_exact(w_new, h_new, FilterType::CatmullRom)
        }
    }

    /// Convert one image into the `[num_patches, in_dim]` tensor the vision
    /// tower expects, plus the `grid_thw` for the text model.
    ///
    /// # Errors
    ///
    /// Returns an error if building the output tensors fails.
    pub fn process(
        &self,
        image: &DynamicImage,
        device: &Device,
        dtype: DType,
    ) -> Result<ProcessedImage> {
        // 1. Convert to RGB (drop alpha) and resize.
        let rgb = image.to_rgb8();
        let resized = self.smart_resize(&DynamicImage::ImageRgb8(rgb));
        let (w, h) = resized.dimensions();
        // patch_size is a small config constant (e.g. 16), far below u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
        let patch_size_u32 = self.patch_size as u32;
        let (h_p, w_p) = (h / patch_size_u32, w / patch_size_u32);
        let n_patches = (h_p * w_p) as usize;

        // 2. CHW float, normalized.
        let mean = &self.image_mean;
        let std = &self.image_std;
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();
        let mut chw = vec![0f32; 3 * (w * h) as usize];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            for c in 0..3 {
                let v = f32::from(pixel[c]) / 255.0;
                chw[c * (w * h) as usize + (y * w + x) as usize] = (v - mean[c]) / std[c];
            }
        }
        let chw = Tensor::from_vec(chw, (3, h as usize, w as usize), device)?.to_dtype(dtype)?;

        // 3. Reshape into patches. Each patch covers `patch_size × patch_size`
        //    pixels in 3 channels, repeated `temporal_patch_size` times
        //    (duplicated for a still image).
        let patch = self.patch_size;
        let t_patch = self.temporal_patch_size;
        let in_dim = t_patch * 3 * patch * patch;
        let mut patches = vec![0f32; n_patches * in_dim];

        let patch_size_u = patch;
        let w_u = w as usize;
        let h_u = h as usize;
        let h_p_u = h_p as usize;
        let w_p_u = w_p as usize;

        // Materialize chw once (CPU vec), then index in-place.
        let chw_data: Vec<f32> = chw
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
            .into_iter()
            .flatten()
            .flatten()
            .collect();

        // Row order and within-row layout both follow HF Qwen2VL's
        //   patches.reshape(t, temporal, C, h/m, m, p, w/m, m, p)
        //          .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        // i.e. rows are ordered (t, h_block, w_block, m_row, m_col) — each 2x2
        // spatial-merge block's 4 patches are CONTIGUOUS, not raster-scan — and
        // each row is laid out (channel, temporal, patch_y, patch_x).
        let merge = self.merge_size;
        let pp = patch_size_u * patch_size_u;
        let mut patch_idx = 0usize;
        for h_blk in 0..(h_p_u / merge) {
            for w_blk in 0..(w_p_u / merge) {
                for m_row in 0..merge {
                    for m_col in 0..merge {
                        let py = h_blk * merge + m_row;
                        let px = w_blk * merge + m_col;
                        let base = patch_idx * in_dim;
                        for c in 0..3 {
                            for t in 0..t_patch {
                                for j in 0..patch_size_u {
                                    for i in 0..patch_size_u {
                                        let y = py * patch_size_u + j;
                                        let x = px * patch_size_u + i;
                                        let src = c * (h_u * w_u) + y * w_u + x;
                                        let dst = base
                                            + c * (t_patch * pp)
                                            + t * pp
                                            + j * patch_size_u
                                            + i;
                                        patches[dst] = chw_data[src];
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }

        let pixel_values =
            Tensor::from_vec(patches, (n_patches, in_dim), device)?.to_dtype(dtype)?;
        Ok(ProcessedImage {
            pixel_values,
            grid_thw: (1, h_p, w_p),
        })
    }
}

/// Batch-concatenate a list of processed images into the flat tensors
/// `Qwen3_5VisionModel::forward` expects.
///
/// # Panics
///
/// Panics if a `ProcessedImage`'s `pixel_values` tensor is not 2-D, which
/// cannot happen because [`PreprocessorConfig::process`] always builds a
/// `[num_patches, in_dim]` tensor.
///
/// # Errors
///
/// Returns an error if building the output tensors fails.
pub fn batch_images(images: &[ProcessedImage]) -> Result<(Tensor, Tensor)> {
    let mut pix = Vec::new();
    let mut grids = Vec::new();
    for img in images {
        let data = img.pixel_values.to_vec2::<f32>()?;
        for row in data {
            pix.extend(row);
        }
        grids.push(img.grid_thw);
    }
    let total_patches: usize = images.iter().map(|i| i.pixel_values.dim(0).unwrap()).sum();
    let in_dim = if total_patches > 0 {
        images[0].pixel_values.dim(1).unwrap()
    } else {
        0
    };
    let device = images
        .first()
        .map_or(Device::Cpu, |i| i.pixel_values.device().clone());
    let pixel_values = Tensor::from_vec(pix, (total_patches, in_dim), &device)?;
    let flat_grids: Vec<u32> = grids.iter().flat_map(|g| [g.0, g.1, g.2]).collect();
    let image_grid_thw = Tensor::from_vec(flat_grids, (grids.len(), 3), &device)?;
    Ok((pixel_values, image_grid_thw))
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `shortest_edge` (= min_pixels) is deliberately tiny so the small
    /// synthetic test images below are not smart-resized at all.
    fn cfg() -> PreprocessorConfig {
        PreprocessorConfig {
            size: Size {
                shortest_edge: 16,
                longest_edge: 16777216,
            },
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
        }
    }

    /// HF `smart_resize` rounds to the NEAREST multiple of `patch * merge`
    /// (`round_by_factor`). A ceil here silently changes the patch grid — and
    /// therefore the number of `<|image_pad|>` tokens — versus the reference.
    #[test]
    fn smart_resize_rounds_to_nearest_not_up() {
        let f = 32; // patch 16 * merge 2
        let (min, max) = (65536usize, 16777216usize);
        // 294 / 32 = 9.19 -> nearest is 9 (288), NOT 10 (320).
        assert_eq!(smart_resize(294, 1024, f, min, max), (288, 1024));
        // 300 / 32 = 9.375 -> still 9.
        assert_eq!(smart_resize(300, 1024, f, min, max), (288, 1024));
        // 310 / 32 = 9.6875 -> rounds up to 10 (320).
        assert_eq!(smart_resize(310, 1024, f, min, max), (320, 1024));
        // Exact multiples are untouched.
        assert_eq!(smart_resize(288, 1024, f, min, max), (288, 1024));
    }

    /// Patch rows must be ordered 2x2-merge-block-major (HF's
    /// `.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)`), and each row laid out
    /// (channel, temporal, patch_y, patch_x) — NOT raster order with a
    /// temporal-major row. A permutation bug here leaves every summary
    /// statistic identical while scrambling the image, so assert the layout
    /// directly with a positionally-encoded image.
    #[test]
    fn patch_layout_is_merge_block_major_and_channel_major() {
        let pp = cfg();
        let patch = pp.patch_size as u32;
        // 4x2 patches => 2x1 merge blocks. Encode each pixel's patch index in
        // the red channel so we can read the ordering back out.
        let (w_p, h_p) = (4u32, 2u32);
        let (w, h) = (w_p * patch, h_p * patch);
        let mut img = image::RgbImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let (py, pxi) = (y / patch, x / patch);
            *px = image::Rgb([(py * w_p + pxi) as u8, 7, 9]);
        }
        let processed = pp
            .process(&DynamicImage::ImageRgb8(img), &Device::Cpu, DType::F32)
            .expect("process");
        assert_eq!(processed.grid_thw, (1, h_p, w_p));

        let rows = processed.pixel_values.to_vec2::<f32>().expect("to_vec2");
        assert_eq!(rows.len(), (h_p * w_p) as usize);
        let pp_sz = (patch * patch) as usize;
        let t_patch = pp.temporal_patch_size;
        assert_eq!(rows[0].len(), 3 * t_patch * pp_sz);

        // Row order: block (0,0) contributes patches (0,0) (0,1) (1,0) (1,1)
        // i.e. raster patch indices 0, 1, 4, 5 — then block (0,1) gives 2, 3, 6, 7.
        let decode = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u32;
        let got: Vec<u32> = rows.iter().map(|r| decode(r[0])).collect();
        assert_eq!(
            got,
            vec![0, 1, 4, 5, 2, 3, 6, 7],
            "patch rows must be merge-block-major"
        );

        // Within a row: channel-major, then temporal. Channel 1 == 7, ch 2 == 9,
        // and both temporal copies of a channel must be identical.
        for row in &rows {
            for (c, expect) in [(1usize, 7u32), (2usize, 9u32)] {
                for t in 0..t_patch {
                    assert_eq!(
                        decode(row[c * (t_patch * pp_sz) + t * pp_sz]),
                        expect,
                        "row layout must be (channel, temporal, y, x)"
                    );
                }
            }
        }
    }
}
