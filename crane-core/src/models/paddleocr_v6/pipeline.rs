//! Complete two-stage `PaddleOCR` v6 pipeline.
//!
//! This module owns all v6-specific preprocessing, detector postprocessing,
//! crop recognition, CTC decoding, and reading-order assembly.

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{DEFAULT_CHECKPOINT_DIR, DICTIONARY_FILE, PaddleOcrV6};
use crate::utils::float_cast::{f32_to_u32, f32_to_usize};

#[derive(Debug, Clone)]
pub struct OcrRegion {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct OcrDocument {
    pub text: String,
    pub regions: Vec<OcrRegion>,
}

pub struct PaddleOcrV6Pipeline {
    model: PaddleOcrV6,
    dictionary: Vec<String>,
}

impl PaddleOcrV6Pipeline {
    /// # Errors
    ///
    /// Returns an error if [`PaddleOcrV6::from_dir`] fails to load the
    /// detector/recognizer models, or if the dictionary file can't be read.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let requested = path.as_ref();
        let directory = if requested.as_os_str().is_empty() || requested == Path::new("checkpoints")
        {
            PathBuf::from(DEFAULT_CHECKPOINT_DIR)
        } else {
            requested.to_owned()
        };
        let model = PaddleOcrV6::from_dir(&directory)?;
        let mut dictionary = vec![String::new()];
        dictionary.extend(
            std::fs::read_to_string(directory.join(DICTIONARY_FILE))?
                .lines()
                .map(str::to_owned),
        );
        dictionary.push(" ".to_owned());
        Ok(Self { model, dictionary })
    }

    /// # Errors
    ///
    /// Returns an error if `path` can't be opened as an image, if the
    /// detector produces an unexpected output shape, or if detection or
    /// recognition inference fails.
    pub fn recognize(&self, path: impl AsRef<Path>) -> Result<OcrDocument> {
        let started = Instant::now();
        let image = image::open(path.as_ref())
            .map_err(|e| anyhow!("cannot read OCR image {}: {e}", path.as_ref().display()))?
            .to_rgb8();
        let loaded = Instant::now();
        let boxes = self.detect_regions(&image)?;
        let detected = Instant::now();
        let mut regions = self.recognize_regions(&image, boxes)?;
        let recognized = Instant::now();
        if std::env::var_os("CRANE_PADDLEOCR_PROFILE").is_some() {
            eprintln!(
                "PaddleOCR v6 timing: load={:.3}s detect={:.3}s recognize={:.3}s",
                (loaded - started).as_secs_f64(),
                (detected - loaded).as_secs_f64(),
                (recognized - detected).as_secs_f64()
            );
        }
        sort_reading_order(&mut regions);
        let text = regions
            .iter()
            .filter(|region| !region.text.trim().is_empty())
            .map(|region| region.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(OcrDocument { text, regions })
    }

    fn detect_regions(&self, image: &image::RgbImage) -> Result<Vec<Candidate>> {
        let (source_width, source_height) = image.dimensions();
        let limit_side = std::env::var("CRANE_PADDLEOCR_DET_LIMIT")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value >= 32)
            .unwrap_or(960);
        let ratio = (limit_side as f32 / source_width.max(source_height) as f32).min(1.0);
        let width = round_to_32(source_width as f32 * ratio);
        let height = round_to_32(source_height as f32 * ratio);
        let resized =
            image::imageops::resize(image, width, height, image::imageops::FilterType::Triangle);

        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut values = Vec::with_capacity((3 * width * height) as usize);
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    values.push(
                        (resized.get_pixel(x, y)[channel] as f32 / 255.0 - mean[channel])
                            / std[channel],
                    );
                }
            }
        }
        let input = Tensor::from_vec(
            values,
            (1, 3, height as usize, width as usize),
            &Device::Cpu,
        )
        .map_err(model_error)?;
        let output = self.model.detect(&input).map_err(model_error)?;
        let shape = output.dims();
        if shape.len() != 4 || shape[0] != 1 || shape[1] < 1 {
            return Err(anyhow!(
                "PaddleOCR detector returned invalid shape {shape:?}"
            ));
        }
        let map = output
            .narrow(0, 0, 1)
            .and_then(|tensor| tensor.narrow(1, 0, 1))
            .and_then(|tensor| tensor.reshape((shape[2], shape[3])))
            .and_then(|tensor| tensor.to_vec2::<f32>())
            .map_err(model_error)?;
        Ok(db_regions(&map, width, height, source_width, source_height))
    }

    fn recognize_regions(
        &self,
        image: &image::RgbImage,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<OcrRegion>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut crops = candidates
            .into_iter()
            .map(|candidate| {
                let crop = image::imageops::crop_imm(
                    image,
                    candidate.left,
                    candidate.top,
                    candidate.right - candidate.left,
                    candidate.bottom - candidate.top,
                )
                .to_image();
                let width =
                    f32_to_usize((crop.width() as f32 / crop.height() as f32) * 48.0).max(1);
                PreparedCrop {
                    candidate,
                    crop,
                    width,
                }
            })
            .collect::<Vec<_>>();
        crops.sort_by_key(|crop| crop.width);
        let batch_size = std::env::var("CRANE_PADDLEOCR_REC_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(crops.len());

        // Crane Added 20260731: recognize all crops in one call by default.
        // Memory-constrained callers can request smaller aspect-ratio-sorted
        // batches without changing the model or evaluator.
        let mut regions = Vec::with_capacity(crops.len());
        for batch in crops.chunks(batch_size) {
            self.recognize_batch(batch, &mut regions)?;
        }
        Ok(regions)
    }

    fn recognize_batch(&self, batch: &[PreparedCrop], regions: &mut Vec<OcrRegion>) -> Result<()> {
        let max_width = batch.iter().map(|crop| crop.width).max().unwrap_or(1);
        let mut values = vec![-1.0f32; batch.len() * 3 * 48 * max_width];
        for (batch_index, prepared) in batch.iter().enumerate() {
            let resized = image::imageops::resize(
                &prepared.crop,
                prepared.width as u32,
                48,
                image::imageops::FilterType::Triangle,
            );
            // Python reference receives BGR crops from cv2.imread.
            for channel in 0..3 {
                for y in 0..48 {
                    for x in 0..prepared.width {
                        let index = (((batch_index * 3 + channel) * 48 + y) * max_width) + x;
                        values[index] = (resized.get_pixel(x as u32, y as u32)[2 - channel] as f32
                            / 255.0
                            - 0.5)
                            / 0.5;
                    }
                }
            }
        }

        let input = Tensor::from_vec(values, (batch.len(), 3, 48usize, max_width), &Device::Cpu)
            .map_err(model_error)?;
        let logits = self
            .model
            .recognize(&input)
            .map_err(model_error)?
            .to_vec3::<f32>()
            .map_err(model_error)?;
        if logits.len() != batch.len() {
            return Err(anyhow!(
                "PaddleOCR recognizer returned batch {}, expected {}",
                logits.len(),
                batch.len()
            ));
        }

        regions.extend(batch.iter().zip(logits).filter_map(|(prepared, logits)| {
            let candidate = prepared.candidate;
            let (text, recognition_score) = ctc_decode(&logits, &self.dictionary);
            (!text.trim().is_empty()).then_some(OcrRegion {
                left: candidate.left,
                top: candidate.top,
                right: candidate.right,
                bottom: candidate.bottom,
                text,
                confidence: recognition_score.min(candidate.score),
            })
        }));
        Ok(())
    }
}

struct PreparedCrop {
    candidate: Candidate,
    crop: image::RgbImage,
    width: usize,
}

#[derive(Clone, Copy)]
struct Candidate {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    score: f32,
}

fn round_to_32(value: f32) -> u32 {
    f32_to_u32(((value / 32.0).round() * 32.0).max(32.0))
}

fn db_regions(
    map: &[Vec<f32>],
    resized_width: u32,
    resized_height: u32,
    source_width: u32,
    source_height: u32,
) -> Vec<Candidate> {
    let height = map.len();
    let width = map.first().map_or(0, Vec::len);
    let mut visited = vec![false; width * height];
    let mut boxes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if visited[y * width + x] || map[y][x] <= 0.3 {
                continue;
            }
            let mut stack = vec![(x, y)];
            visited[y * width + x] = true;
            let (mut x0, mut y0, mut x1, mut y1) = (x, y, x, y);
            let (mut score, mut count) = (0.0f32, 0usize);
            while let Some((cx, cy)) = stack.pop() {
                x0 = x0.min(cx);
                y0 = y0.min(cy);
                x1 = x1.max(cx);
                y1 = y1.max(cy);
                score += map[cy][cx];
                count += 1;
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as isize + dx;
                    let ny = cy as isize + dy;
                    if nx >= 0 && ny >= 0 && nx < width as isize && ny < height as isize {
                        let index = ny as usize * width + nx as usize;
                        if !visited[index] && map[ny as usize][nx as usize] > 0.3 {
                            visited[index] = true;
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            let score = score / count as f32;
            if count < 3 || score < 0.6 {
                continue;
            }
            // Approximate DB unclip_ratio=1.5 for an axis-aligned component.
            let grow_x = f32_to_usize(((x1 - x0 + 1) as f32 * 0.25).ceil());
            let grow_y = f32_to_usize(((y1 - y0 + 1) as f32 * 0.25).ceil());
            x0 = x0.saturating_sub(grow_x);
            y0 = y0.saturating_sub(grow_y);
            x1 = (x1 + grow_x).min(width.saturating_sub(1));
            y1 = (y1 + grow_y).min(height.saturating_sub(1));

            let map_to_source_x =
                source_width as f32 / resized_width as f32 * resized_width as f32 / width as f32;
            let map_to_source_y = source_height as f32 / resized_height as f32
                * resized_height as f32
                / height as f32;
            let left = f32_to_u32((x0 as f32 * map_to_source_x).floor());
            let top = f32_to_u32((y0 as f32 * map_to_source_y).floor());
            let right = f32_to_u32(((x1 + 1) as f32 * map_to_source_x).ceil()).min(source_width);
            let bottom = f32_to_u32(((y1 + 1) as f32 * map_to_source_y).ceil()).min(source_height);
            if right > left + 2 && bottom > top + 2 {
                boxes.push(Candidate {
                    left,
                    top,
                    right,
                    bottom,
                    score,
                });
            }
        }
    }
    boxes
}

fn ctc_decode(logits: &[Vec<f32>], dictionary: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut scores = Vec::new();
    let mut previous = usize::MAX;
    for timestep in logits {
        let Some((index, score)) = timestep
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
        else {
            continue;
        };
        if index != 0 && index != previous {
            if let Some(character) = dictionary.get(index) {
                text.push_str(character);
                scores.push(*score);
            }
        }
        previous = index;
    }
    let confidence = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f32>() / scores.len() as f32
    };
    (text, confidence)
}

fn sort_reading_order(regions: &mut [OcrRegion]) {
    regions.sort_by(|left, right| {
        let average_height = ((left.bottom - left.top) + (right.bottom - right.top)) as f32 / 2.0;
        if (left.top as i64 - right.top as i64).unsigned_abs() as f32 <= average_height * 0.6 {
            left.left.cmp(&right.left)
        } else {
            left.top.cmp(&right.top)
        }
    });
}

fn model_error(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{error}")
}
