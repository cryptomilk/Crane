//! Vision-tower sanity check for Qwen 3.5: load the ViT from a real
//! multimodal checkpoint, run forward on a dummy image tensor, verify the
//! output shape matches the expected `[merged_tokens, out_hidden_size]`.
//!
//! Gated by `CRANE_QWEN35_VL_DIR` so it doesn't run by default. Used to catch
//! regressions when the vision module's shape conventions change.

use candle_core::Device;

#[test]
#[ignore = "needs a local Qwen3.5 multimodal checkpoint (CRANE_QWEN35_VL_DIR)"]
fn qwen3_5_vision_forward_shape() {
    use candle_core::{DType, IndexOp, Tensor};
    use candle_nn::VarBuilder;

    use crane_core::models::qwen3_5::{Qwen3_5VisionModel, load_config};
    use crane_core::utils::utils::get_safetensors_files;

    let dir = std::env::var("CRANE_QWEN35_VL_DIR")
        .expect("set CRANE_QWEN35_VL_DIR to a Qwen3.5 multimodal checkpoint dir");

    let cfg = load_config(&format!("{dir}/config.json")).expect("load config");
    let vcfg = cfg
        .vision_config
        .as_ref()
        .expect("config.json has no vision_config — need a multimodal checkpoint");

    println!(
        "vision: depth={} hidden={} out_hidden={} heads={} patch={} merge={} temporal={}",
        vcfg.depth,
        vcfg.hidden_size,
        vcfg.out_hidden_size,
        vcfg.num_heads,
        vcfg.patch_size,
        vcfg.spatial_merge_size,
        vcfg.temporal_patch_size,
    );

    // Resolve device/dtype: CUDA + F16 if available, else CPU + F32.
    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (Device::new_cuda(0).unwrap(), DType::F16)
    } else {
        (Device::Cpu, DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (Device::Cpu, DType::F32);

    let filenames = get_safetensors_files(&dir).expect("locate safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) }
        .expect("mmap safetensors");

    let vision =
        Qwen3_5VisionModel::new(vcfg, &vb.pp("model").pp("visual")).expect("build vision tower");

    // 408x408 image → resize to multiple of 32 (patch * merge) → 416x416.
    // patches per side = 416 / 16 = 26.
    let h_patches = 26usize;
    let w_patches = 26usize;
    let t_patches = 1usize; // single image
    let n_patches = t_patches * h_patches * w_patches;
    let in_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;

    let pixel_values =
        Tensor::zeros((n_patches, in_dim), dtype, &device).expect("zero pixel_values");
    let grid_thw = Tensor::from_vec(
        vec![t_patches as u32, h_patches as u32, w_patches as u32],
        (1, 3),
        &device,
    )
    .expect("grid_thw");

    let (out, _deepstack) = vision
        .forward(&pixel_values, &grid_thw)
        .expect("vision forward");

    // After 2x2 spatial merge: n_patches / 4 tokens.
    let merged = n_patches / vcfg.spatial_merge_size.pow(2);
    let dims = out.dims();
    assert_eq!(
        dims,
        &[merged, vcfg.out_hidden_size],
        "expected [{merged}, {}], got {dims:?}",
        vcfg.out_hidden_size
    );
    println!("vision forward OK: out shape = {dims:?}");
    let sample_f32: Vec<f32> = out
        .i((0, ..))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    println!("first-token sample (first 8): {:?}", &sample_f32[..8]);
    assert!(sample_f32.iter().any(|x| x.is_finite() && *x != 0.0));
}
