// This implementation was adapted from the fastembed-rs repository (https://github.com/Anush008/fastembed-rs).
// Thank you for incredible work!

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, Module, VarBuilder};
use image::{imageops::FilterType, DynamicImage};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use unicode_categories::UnicodeCategories;

#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiBuilder;

use crate::models::qwen3_vl::{Qwen3VLVisionModel, VisionConfig};

pub enum QLinear {
    Quantized(QMatMul),
    QuantizedWithBias(QMatMul, Tensor),
    Unquantized(candle_nn::Linear),
}

pub enum VisualInput {
    Image(DynamicImage),
    Video(Vec<DynamicImage>),
}

impl candle_nn::Module for QLinear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Quantized(qm) => {
                if xs.dtype() == DType::F32 {
                    qm.forward(xs)
                } else {
                    let xs_f32 = xs.to_dtype(DType::F32)?;
                    let out_f32 = qm.forward(&xs_f32)?;
                    out_f32.to_dtype(xs.dtype())
                }
            }
            Self::QuantizedWithBias(qm, bias) => {
                let out = if xs.dtype() == DType::F32 {
                    qm.forward(xs)?
                } else {
                    let xs_f32 = xs.to_dtype(DType::F32)?;
                    let out_f32 = qm.forward(&xs_f32)?;
                    out_f32.to_dtype(xs.dtype())?
                };

                let bias_cast = bias.to_dtype(out.dtype())?;
                out.broadcast_add(&bias_cast)
            }
            Self::Unquantized(l) => l.forward(xs),
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub head_dim: Option<usize>,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<serde_json::Value>,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub use_cache: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_window_layers: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(Deserialize)]
struct Qwen3VLConfig {
    text_config: Config,
}

fn parse_config_and_weight_prefix(config_bytes: &[u8]) -> Result<(Config, Option<&'static str>)> {
    if let Ok(cfg) = serde_json::from_slice::<Config>(config_bytes) {
        return Ok((cfg, None));
    }

    if let Ok(cfg) = serde_json::from_slice::<Qwen3VLConfig>(config_bytes) {
        return Ok((cfg.text_config, Some("model.language_model")));
    }

    Err(candle_core::Error::Msg(
        "Failed to parse config as Qwen3 or Qwen3-VL text config".into(),
    ))
}

fn find_safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut files = Vec::new();
    for shard_idx in 1..=20 {
        let entries =
            std::fs::read_dir(model_dir).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        for entry in entries.flatten() {
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            if fname_str.starts_with(&format!("model-{:05}-of-", shard_idx))
                && fname_str.ends_with(".safetensors")
            {
                files.push(entry.path());
                break;
            }
        }
    }

    if files.is_empty() {
        return Err(candle_core::Error::Msg(
            "Could not locate model.safetensors or sharded weight files".into(),
        ));
    }
    Ok(files)
}

#[derive(Deserialize, Debug, Clone)]
struct Qwen3VLFullConfig {
    text_config: Config,
    vision_config: VisionConfig,
    image_token_id: u32,
    #[allow(dead_code)]
    vision_start_token_id: u32,
    #[allow(dead_code)]
    vision_end_token_id: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct Qwen3VLPreprocessorConfig {
    min_pixels: usize,
    max_pixels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    rescale_factor: f32,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

struct PreparedImage {
    pixel_values: Vec<f32>,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    num_llm_tokens: usize,
}

fn scalar_f32(device: &Device, v: f32) -> Result<Tensor> {
    Tensor::from_slice(&[v], (1,), device)?.to_dtype(DType::F32)
}

fn scalar_f64_as_f32(device: &Device, v: f64) -> Result<Tensor> {
    scalar_f32(device, v as f32)
}

fn map_err<E: std::fmt::Display>(err: E) -> candle_core::Error {
    candle_core::Error::Msg(err.to_string())
}

fn build_attention_mask_4d(attention_mask_2d: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len) = attention_mask_2d.dims2()?;
    let device = attention_mask_2d.device();
    let mask_value = -1e4f32;

    let causal = {
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[i * seq_len + j] = mask_value;
            }
        }
        Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?
    };

    let pad_mask_f32 = attention_mask_2d.to_dtype(DType::F32)?;
    let ones = Tensor::ones_like(&pad_mask_f32)?;
    let inverted_mask = ones.sub(&pad_mask_f32)?;

    let mask_val_t = Tensor::new(&[mask_value], device)?;
    let pad_additive = inverted_mask.broadcast_mul(&mask_val_t)?;

    let pad_additive = pad_additive.unsqueeze(1)?.unsqueeze(2)?;

    let causal_broadcast = causal.broadcast_as((batch_size, 1, seq_len, seq_len))?;
    causal_broadcast.broadcast_add(&pad_additive)
}

fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let xs_f32 = xs.to_dtype(DType::F32)?;
    let sum_sq = xs_f32.sqr()?.sum_keepdim(1)?;
    let eps_tensor = Tensor::new(&[1e-12f32], xs.device())?.broadcast_as(sum_sq.shape())?;
    let norm = sum_sq.add(&eps_tensor)?.sqrt()?;
    // xs.broadcast_div(&norm)
    xs_f32.broadcast_div(&norm)?.to_dtype(xs.dtype())
}

fn last_token_pool(hidden: &Tensor, attention_mask_2d: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, _) = hidden.dims3()?;
    let masks = attention_mask_2d.to_vec2::<f32>()?;
    if masks.len() != batch_size {
        return Err(candle_core::Error::Msg(
            "attention mask batch size mismatch".into(),
        ));
    }

    let mut pooled_rows = Vec::with_capacity(batch_size);
    for (batch_idx, row) in masks.iter().enumerate() {
        let last_idx = row.iter().rposition(|&v| v > 0.0).unwrap_or(seq_len - 1);
        pooled_rows.push(hidden.i((batch_idx, last_idx))?);
    }
    let pooled_refs: Vec<&Tensor> = pooled_rows.iter().collect();
    Tensor::stack(&pooled_refs, 0)
}

fn find_token_spans(ids: &[u32], token_id: u32) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == token_id {
            let start = i;
            while i < ids.len() && ids[i] == token_id {
                i += 1;
            }
            spans.push((start, i));
        } else {
            i += 1;
        }
    }
    spans
}

fn round_ties_to_even(value: f64) -> usize {
    let floor = value.floor();
    let frac = value - floor;
    if frac < 0.5 {
        floor as usize
    } else if frac > 0.5 {
        (floor + 1.0) as usize
    } else if (floor as i64) % 2 == 0 {
        floor as usize
    } else {
        (floor + 1.0) as usize
    }
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(candle_core::Error::Msg(
            "Image dimensions must be greater than zero".into(),
        ));
    }
    let aspect = (height.max(width) as f64) / (height.min(width) as f64);
    if aspect > 200.0 {
        return Err(candle_core::Error::Msg(
            "Absolute aspect ratio must be <= 200".into(),
        ));
    }

    // Match Python `round()` behavior used by qwen-vl-utils (ties-to-even).
    let mut h_bar = round_ties_to_even(height as f64 / factor as f64) * factor;
    let mut w_bar = round_ties_to_even(width as f64 / factor as f64) * factor;
    h_bar = h_bar.max(factor);
    w_bar = w_bar.max(factor);

    let area = (height * width) as f64;
    if h_bar * w_bar > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / area).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize * factor).max(factor);
    }

    Ok((h_bar, w_bar))
}

fn preprocess_image(img: &DynamicImage, cfg: &Qwen3VLPreprocessorConfig) -> Result<PreparedImage> {
    if cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 {
        return Err(candle_core::Error::Msg(
            "Expected image_mean and image_std length to be 3".into(),
        ));
    }
    if cfg.patch_size == 0 || cfg.temporal_patch_size == 0 || cfg.merge_size == 0 {
        return Err(candle_core::Error::Msg(
            "patch_size, temporal_patch_size and merge_size must be > 0".into(),
        ));
    }

    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = rgb.dimensions();
    let factor = cfg.patch_size * cfg.merge_size;
    let (resized_h, resized_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    )?;

    let resized = image::imageops::resize(
        &rgb,
        resized_w as u32,
        resized_h as u32,
        FilterType::Triangle,
    );

    let scale = [
        cfg.rescale_factor / cfg.image_std[0],
        cfg.rescale_factor / cfg.image_std[1],
        cfg.rescale_factor / cfg.image_std[2],
    ];
    let offset = [
        cfg.image_mean[0] / cfg.image_std[0],
        cfg.image_mean[1] / cfg.image_std[1],
        cfg.image_mean[2] / cfg.image_std[2],
    ];

    let width = resized.width() as usize;
    let height = resized.height() as usize;
    let pixels = resized.as_raw();

    let grid_t = 1usize;
    let grid_h = resized_h / cfg.patch_size;
    let grid_w = resized_w / cfg.patch_size;
    let merge = cfg.merge_size;

    if grid_h % merge != 0 || grid_w % merge != 0 {
        return Err(candle_core::Error::Msg(
            "grid_h and grid_w must be divisible by merge_size".into(),
        ));
    }

    let channels = 3usize;
    let patch_size = cfg.patch_size;
    let temporal_patch_size = cfg.temporal_patch_size;
    let patch_dim = channels * temporal_patch_size * patch_size * patch_size;
    let total_patch_tokens = grid_t * grid_h * grid_w;
    let total_values = total_patch_tokens * patch_dim;
    let mut out = Vec::with_capacity(total_values);
    out.resize(total_values, 0.0);

    let merged_h = grid_h / merge;
    let merged_w = grid_w / merge;
    let patch_size_sq = patch_size * patch_size;
    let patch_size_times_channels = patch_size_sq * channels;

    let mut idx = 0;
    for _t in 0..grid_t {
        for gh_block in 0..merged_h {
            for gw_block in 0..merged_w {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let gh = gh_block * merge + mh;
                        let gw = gw_block * merge + mw;
                        let base_y = gh * patch_size;
                        let base_x = gw * patch_size;

                        for c in 0..channels {
                            let scale_c = scale[c];
                            let offset_c = offset[c];
                            for _tp in 0..temporal_patch_size {
                                for ph in 0..patch_size {
                                    let y = base_y + ph;
                                    let row_offset = y * width;
                                    for pw in 0..patch_size {
                                        let x = base_x + pw;
                                        let pixel_idx = (row_offset + x) * 3 + c;
                                        let pixel = pixels[pixel_idx] as f32;
                                        out[idx] = pixel * scale_c - offset_c;
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let num_llm_tokens = total_patch_tokens / (merge * merge);
    Ok(PreparedImage {
        pixel_values: out,
        grid_t: grid_t as u32,
        grid_h: grid_h as u32,
        grid_w: grid_w as u32,
        num_llm_tokens,
    })
}

fn preprocess_video(
    frames: &[DynamicImage],
    cfg: &Qwen3VLPreprocessorConfig,
) -> Result<PreparedImage> {
    if frames.is_empty() {
        return Err(candle_core::Error::Msg("Video frames list is empty".into()));
    }

    // Config validation
    if cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 {
        return Err(candle_core::Error::Msg("Invalid mean/std config".into()));
    }

    let mut processed_frames: Vec<std::borrow::Cow<DynamicImage>> = frames
        .iter()
        .map(|f| std::borrow::Cow::Borrowed(f))
        .collect();

    if processed_frames.len() % cfg.temporal_patch_size != 0 {
        if let Some(last) = processed_frames.last() {
            processed_frames.push(last.clone());
        }
    }

    let num_frames = processed_frames.len();
    let grid_t = num_frames / cfg.temporal_patch_size;

    // 2. Using FIRST frame as reference for resizing
    let first_frame = &processed_frames[0];
    let (orig_w, orig_h) = (first_frame.width(), first_frame.height());
    let factor = cfg.patch_size * cfg.merge_size;

    let (resized_h, resized_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
    )?;

    // 3. Resizing all frames
    let resized_frames: Vec<image::RgbImage> = processed_frames
        .iter()
        .map(|f| {
            let rgb = f.to_rgb8();
            let resized = image::imageops::resize(
                &rgb,
                resized_w as u32,
                resized_h as u32,
                FilterType::Triangle,
            );
            resized
        })
        .collect();

    let grid_h = resized_h / cfg.patch_size;
    let grid_w = resized_w / cfg.patch_size;
    let merge = cfg.merge_size;

    if grid_h % merge != 0 || grid_w % merge != 0 {
        return Err(candle_core::Error::Msg(
            "grid_h and grid_w must be divisible by merge_size".into(),
        ));
    }

    let channels = 3usize;
    let patch_size = cfg.patch_size;
    let temporal_patch_size = cfg.temporal_patch_size;
    let patch_dim = channels * temporal_patch_size * patch_size * patch_size;

    let scale = [
        cfg.rescale_factor / cfg.image_std[0],
        cfg.rescale_factor / cfg.image_std[1],
        cfg.rescale_factor / cfg.image_std[2],
    ];
    let offset = [
        cfg.image_mean[0] / cfg.image_std[0],
        cfg.image_mean[1] / cfg.image_std[1],
        cfg.image_mean[2] / cfg.image_std[2],
    ];

    let total_patch_tokens = grid_t * grid_h * grid_w;
    let mut out = Vec::with_capacity(total_patch_tokens * patch_dim);

    let merged_h = grid_h / merge;
    let merged_w = grid_w / merge;

    for t_block in 0..grid_t {
        for gh_block in 0..merged_h {
            for gw_block in 0..merged_w {
                for mh in 0..merge {
                    for mw in 0..merge {
                        let gh = gh_block * merge + mh;
                        let gw = gw_block * merge + mw;
                        let base_y = gh * patch_size;
                        let base_x = gw * patch_size;

                        for c in 0..channels {
                            let scale_c = scale[c];
                            let offset_c = offset[c];
                            for tp in 0..temporal_patch_size {
                                let frame_idx = t_block * temporal_patch_size + tp;
                                let frame = &resized_frames[frame_idx];
                                let frame_width = frame.width() as usize;
                                let frame_height = frame.height() as usize;

                                for ph in 0..patch_size {
                                    let y = base_y + ph;
                                    for pw in 0..patch_size {
                                        let x = base_x + pw;

                                        let pixel = if x < frame_width && y < frame_height {
                                            frame.get_pixel(x as u32, y as u32).0[c]
                                        } else {
                                            0
                                        };

                                        let value = (pixel as f32) * scale_c - offset_c;
                                        out.push(value);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let num_llm_tokens = total_patch_tokens / (merge * merge);

    Ok(PreparedImage {
        pixel_values: out,
        grid_t: grid_t as u32,
        grid_h: grid_h as u32,
        grid_w: grid_w as u32,
        num_llm_tokens,
    })
}

fn build_vl_prompt(text: Option<&str>, include_image: bool, instruction: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(instruction);
    prompt.push_str("<|im_end|>\n<|im_start|>user\n");
    if include_image {
        prompt.push_str("<|vision_start|><|image_pad|><|vision_end|>");
    }
    if let Some(text) = text {
        prompt.push_str(text);
    }
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

fn prepare_instruction(instruction: Option<&str>, default_instruction: &str) -> String {
    if let Some(inst) = instruction {
        let trimmed = inst.trim();
        if !trimmed.is_empty() {
            let mut result = trimmed.to_string();

            if let Some(last_char) = result.chars().last() {
                if !last_char.is_punctuation() {
                    result.push('.');
                }
            }
            return result;
        }
    }

    default_instruction.to_string()
}

fn expand_image_token_placeholders(prompt: &str, num_image_tokens: usize) -> Result<String> {
    if num_image_tokens == 0 {
        return Ok(prompt.to_string());
    }
    let image_token = "<|image_pad|>";
    if !prompt.contains(image_token) {
        return Err(candle_core::Error::Msg(
            "Prompt contains no <|image_pad|> placeholder".into(),
        ));
    }
    Ok(prompt.replacen(image_token, &image_token.repeat(num_image_tokens), 1))
}

fn build_image_position_ids(
    encodings: &[tokenizers::Encoding],
    image_spans_per_batch: &[Option<(usize, usize)>],
    prepared_images: &[Option<PreparedImage>],
    merge_size: usize,
    device: &Device,
) -> Result<Tensor> {
    if encodings.is_empty() {
        return Err(candle_core::Error::Msg(
            "encodings cannot be empty when building position ids".into(),
        ));
    }
    if encodings.len() != image_spans_per_batch.len() || encodings.len() != prepared_images.len() {
        return Err(candle_core::Error::Msg(
            "batch size mismatch while building position ids".into(),
        ));
    }

    let batch_size = encodings.len();
    let seq_len = encodings[0].len();
    let mut data = vec![1u32; 3 * batch_size * seq_len];
    let index = |dim: usize, batch: usize, pos: usize| -> usize {
        (dim * batch_size + batch) * seq_len + pos
    };

    for (batch_idx, encoding) in encodings.iter().enumerate() {
        let visible_len = encoding
            .get_attention_mask()
            .iter()
            .filter(|&&m| m != 0)
            .count();

        let Some((start, end)) = image_spans_per_batch[batch_idx] else {
            for pos in 0..visible_len {
                let val = pos as u32;
                data[index(0, batch_idx, pos)] = val;
                data[index(1, batch_idx, pos)] = val;
                data[index(2, batch_idx, pos)] = val;
            }
            continue;
        };

        let prepared = prepared_images[batch_idx].as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "Found image token span for a sample without prepared image".into(),
            )
        })?;
        if end > visible_len {
            return Err(candle_core::Error::Msg(
                "Image token span exceeds visible sequence length".into(),
            ));
        }

        let llm_t = prepared.grid_t as usize;
        let llm_h = prepared.grid_h as usize / merge_size;
        let llm_w = prepared.grid_w as usize / merge_size;
        let image_len = end - start;
        if image_len != llm_t * llm_h * llm_w {
            return Err(candle_core::Error::Msg(format!(
                "Image token span length {} does not match expected LLM grid {}x{}x{}",
                image_len, llm_t, llm_h, llm_w
            )));
        }

        for pos in 0..start {
            let val = pos as u32;
            data[index(0, batch_idx, pos)] = val;
            data[index(1, batch_idx, pos)] = val;
            data[index(2, batch_idx, pos)] = val;
        }

        let mut seq_pos = start;
        for t in 0..llm_t {
            for h in 0..llm_h {
                for w in 0..llm_w {
                    data[index(0, batch_idx, seq_pos)] = (start + t) as u32;
                    data[index(1, batch_idx, seq_pos)] = (start + h) as u32;
                    data[index(2, batch_idx, seq_pos)] = (start + w) as u32;
                    seq_pos += 1;
                }
            }
        }
        if seq_pos != end {
            return Err(candle_core::Error::Msg(
                "Image token position construction consumed the wrong number of tokens".into(),
            ));
        }

        let prefix_max = start.saturating_sub(1);
        let image_max = start + llm_t.max(llm_h).max(llm_w).saturating_sub(1);
        let st_idx = prefix_max.max(image_max) + 1;
        for offset in 0..(visible_len - end) {
            let pos = end + offset;
            let val = (st_idx + offset) as u32;
            data[index(0, batch_idx, pos)] = val;
            data[index(1, batch_idx, pos)] = val;
            data[index(2, batch_idx, pos)] = val;
        }
    }

    Tensor::from_vec(data, (3, batch_size, seq_len), device)
}

fn load_image_from_path(path: &Path) -> Result<DynamicImage> {
    image::ImageReader::open(path)
        .map_err(map_err)?
        .decode()
        .map_err(map_err)
}

pub struct Qwen3RMSNorm {
    weight: Tensor, // [dim]
    eps: f64,
}

impl Qwen3RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for Qwen3RMSNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let dev = xs.device();

        let xs_f = xs.to_dtype(DType::F32)?;
        let var = xs_f.powf(2.0)?.mean_keepdim(D::Minus1)?; // [..., 1]

        let eps_t = scalar_f64_as_f32(dev, self.eps)?;
        let var_eps = var.broadcast_add(&eps_t)?;

        // xs * rsqrt(var + eps)  == xs * (1/sqrt(...))
        let inv_rms = var_eps.sqrt()?.recip()?; // [..., 1]
        let normed = xs_f.broadcast_mul(&inv_rms)?; // [..., dim]

        // weight * back_to_input_dtype
        let normed = normed.to_dtype(in_dtype)?;
        let w = self.weight.to_dtype(in_dtype)?;
        normed.broadcast_mul(&w)
    }
}

pub struct Qwen3MLP {
    gate_proj: QLinear, // hidden -> intermediate
    up_proj: QLinear,   // hidden -> intermediate
    down_proj: QLinear, // intermediate -> hidden
    act_fn: Activation,
}

impl Qwen3MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = QLinear::Unquantized(linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("gate_proj"),
        )?);
        let up_proj = QLinear::Unquantized(linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("up_proj"),
        )?);
        let down_proj = QLinear::Unquantized(linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
        )?);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct Qwen3RotaryEmbedding {
    inv_freq: Tensor,      // [dim/2] f32
    attention_factor: f32, // HF: attention_scaling (1.0 for default)
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
}

impl Qwen3RotaryEmbedding {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let base = cfg.rope_theta; // f64 for precision
        let dim = cfg.head_dim(); // head_dim
        assert!(dim.is_multiple_of(2), "head_dim must be even, got {dim}");

        // t = [0,2,4,...,dim-2] as f32
        let t = Tensor::arange_step(0u32, dim as u32, 2u32, device)?.to_dtype(DType::F32)?;

        // exponent = t / dim
        let dim_t = scalar_f32(device, dim as f32)?;
        let exponent = t.broadcast_div(&dim_t)?; // [dim/2]

        // inv_freq = 1 / (base ** exponent) = exp(-ln(base) * exponent)
        let ln_base = (base as f32).ln();
        let ln_base_t = scalar_f32(device, ln_base)?;
        let inv_freq = exponent.broadcast_mul(&ln_base_t.neg()?)?.exp()?; // [dim/2]

        let mut mrope_interleaved = false;
        let mut mrope_section = [24usize, 20usize, 20usize];
        if let Some(rope_scaling) = cfg.rope_scaling.as_ref() {
            if let Some(v) = rope_scaling
                .get("mrope_interleaved")
                .and_then(|v| v.as_bool())
            {
                mrope_interleaved = v;
            }
            if let Some(arr) = rope_scaling.get("mrope_section").and_then(|v| v.as_array()) {
                if arr.len() == 3 {
                    mrope_section = [
                        arr[0].as_u64().unwrap_or(mrope_section[0] as u64) as usize,
                        arr[1].as_u64().unwrap_or(mrope_section[1] as u64) as usize,
                        arr[2].as_u64().unwrap_or(mrope_section[2] as u64) as usize,
                    ];
                }
            }
        }

        Ok(Self {
            inv_freq,
            attention_factor: 1.0,
            mrope_interleaved,
            mrope_section,
        })
    }

    /// position_ids: [B,T] or [3,B,T] (MRoPE)
    /// returns (cos, sin): [B,T,dim] in xs.dtype()
    pub fn forward(&self, xs: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dev = xs.device();
        let inv_freq = self.inv_freq.to_device(dev)?.to_dtype(DType::F32)?;
        let d2 = inv_freq.dims1()?;

        let freqs = if position_ids.rank() == 2 {
            let (b, t) = position_ids.dims2()?;
            let pos = position_ids
                .to_device(dev)?
                .to_dtype(DType::F32)?
                .contiguous()?;

            let inv_freq_expanded = inv_freq
                .reshape((1, d2, 1))?
                .expand((b, d2, 1))?
                .contiguous()?;
            let pos_expanded = pos.reshape((b, 1, t))?.contiguous()?;

            inv_freq_expanded
                .matmul(&pos_expanded)?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            let (dims, b, t) = position_ids.dims3()?;
            if dims != 3 {
                return Err(candle_core::Error::Msg(
                    "Expected position_ids first dimension to be 3 for MRoPE".into(),
                ));
            }

            let inv = inv_freq.to_vec1::<f32>()?;
            let pos = position_ids.to_device(dev)?.to_vec3::<u32>()?;
            let mut freqs = vec![0f32; b * t * d2];

            // Not sure if it will work fine on GPU

            for batch_idx in 0..b {
                for tok_idx in 0..t {
                    let base = (batch_idx * t + tok_idx) * d2;
                    let temporal = pos[0][batch_idx][tok_idx] as f32;
                    for i in 0..d2 {
                        freqs[base + i] = temporal * inv[i];
                    }

                    if self.mrope_interleaved {
                        for dim in 1..=2 {
                            let pos_dim = pos[dim][batch_idx][tok_idx] as f32;
                            let mut i = dim;
                            let limit = (self.mrope_section[dim] * 3).min(d2);
                            while i < limit {
                                freqs[base + i] = pos_dim * inv[i];
                                i += 3;
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(freqs, (b, t, d2), dev)?
        };

        // emb = cat(freqs, freqs) -> [B,T,dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 2)?;

        let scale = scalar_f32(dev, self.attention_factor)?;
        let cos = emb.cos()?.broadcast_mul(&scale)?;
        let sin = emb.sin()?.broadcast_mul(&scale)?;

        let out_dtype = xs.dtype();
        Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
    }
}

// rotate_half + apply_rotary_pos_emb (HF)
// rotate_half: cat(-x2, x1) along last dim (half-split style)
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x
        .dims()
        .last()
        .copied()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".into()))?;
    assert!(d % 2 == 0, "rotate_half requires even last dim, got {d}");
    let half = d / 2;

    // x1 = x[..., :half], x2 = x[..., half:]
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let nx2 = x2.neg()?;
    Tensor::cat(&[&nx2, &x1], x.rank() - 1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,   // [B, Hq, T, D]
    k: &Tensor,   // [B, Hk, T, D]
    cos: &Tensor, // [B, T, D]
    sin: &Tensor, // [B, T, D]
) -> Result<(Tensor, Tensor)> {
    // unsqueeze_dim=1 -> [B,1,T,D] then broadcast
    let cos_u = cos.unsqueeze(1)?;
    let sin_u = sin.unsqueeze(1)?;

    let q_embed = (q.broadcast_mul(&cos_u)? + rotate_half(q)?.broadcast_mul(&sin_u)?)?;
    let k_embed = (k.broadcast_mul(&cos_u)? + rotate_half(k)?.broadcast_mul(&sin_u)?)?;
    Ok((q_embed, k_embed))
}

// repeat_kv (HF): [B, Nkv, T, D] -> [B, Nh, T, D]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, t, d) = x.dims4()?;
    // [B, Nkv, 1, T, D]
    let x = x.unsqueeze(2)?;
    // broadcast to [B, Nkv, n_rep, T, D]
    let x = x.broadcast_as((b, n_kv, n_rep, t, d))?;
    // reshape to [B, Nkv*n_rep, T, D]
    x.reshape((b, n_kv * n_rep, t, d))
}

fn eager_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scaling: f32,
) -> Result<Tensor> {
    let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let scale_tensor = Tensor::new(scaling, q.device())?.to_dtype(q.dtype())?;
    let attn_weights = attn_weights.broadcast_mul(&scale_tensor)?;
    let attn_weights = match attention_mask {
        None => attn_weights,
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
        .to_dtype(attn_weights.dtype())?;
    attn_weights.matmul(v)
}

// - q_norm/k_norm on head_dim after reshape
// - RoPE on q,k
// - repeat_kv for key/value (GQA)
pub struct Qwen3Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Qwen3RMSNorm, // dim = head_dim
    k_norm: Qwen3RMSNorm, // dim = head_dim

    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f32,
}

impl Qwen3Attention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = cfg.num_kv_groups();
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "num_heads must be multiple of num_kv_heads"
        );

        let q_out = num_heads * head_dim;
        let kv_out = num_kv_heads * head_dim;

        let q_proj = if cfg.attention_bias {
            QLinear::Unquantized(linear(cfg.hidden_size, q_out, vb.pp("q_proj"))?)
        } else {
            QLinear::Unquantized(linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?)
        };
        let k_proj = if cfg.attention_bias {
            QLinear::Unquantized(linear(cfg.hidden_size, kv_out, vb.pp("k_proj"))?)
        } else {
            QLinear::Unquantized(linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?)
        };
        let v_proj = if cfg.attention_bias {
            QLinear::Unquantized(linear(cfg.hidden_size, kv_out, vb.pp("v_proj"))?)
        } else {
            QLinear::Unquantized(linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?)
        };
        let o_proj = if cfg.attention_bias {
            QLinear::Unquantized(linear(q_out, cfg.hidden_size, vb.pp("o_proj"))?)
        } else {
            QLinear::Unquantized(linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj"))?)
        };

        // q_norm/k_norm are RMSNorm(head_dim)
        let q_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3RMSNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            scaling: (head_dim as f32).powf(-0.5),
        })
    }

    /// hidden_states: [B,T,H]
    /// position_embeddings: (cos,sin) both [B,T,D]
    /// attention_mask: additive mask [B,1,T,T] (0 or -inf)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _h) = hidden_states.dims3()?;
        let d = self.head_dim;

        // hidden_shape = [B,T,-1,D]
        // q: [B,T,Nh*D] -> [B,T,Nh,D] -> q_norm -> transpose -> [B,Nh,T,D]
        let q = hidden_states
            .apply(&self.q_proj)?
            .reshape((b, t, self.num_heads, d))?;
        let q = q.apply(&self.q_norm)?.transpose(1, 2)?;

        // k: [B,T,Nkv*D] -> [B,T,Nkv,D] -> k_norm -> [B,Nkv,T,D]
        let k = hidden_states
            .apply(&self.k_proj)?
            .reshape((b, t, self.num_kv_heads, d))?;
        let k = k.apply(&self.k_norm)?.transpose(1, 2)?;

        // v: [B,Nkv,T,D]
        let v = hidden_states
            .apply(&self.v_proj)?
            .reshape((b, t, self.num_kv_heads, d))?
            .transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // GQA expand
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;

        // Use SDPA for Metal only, eager attention for CPU/CUDA
        // Note: candle_nn::ops::sdpa is Metal-only, not available for CUDA
        let is_metal = q.device().is_metal();
        let out = if is_metal {
            let mask = if let Some(mask) = attention_mask {
                Some(mask.broadcast_as((b, self.num_heads, t, t))?)
            } else {
                None
            };
            candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, self.scaling, 1.0)?
        } else {
            // Use eager attention for CPU and CUDA (SDPA is Metal-only in Candle)
            let mask = if let Some(mask) = attention_mask {
                Some(mask.broadcast_as((b, self.num_heads, t, t))?)
            } else {
                None
            };
            eager_attention_forward(&q, &k, &v, mask.as_ref(), self.scaling)?
        };

        // transpose back -> [B,T,Nh,D] -> reshape [B,T,Nh*D] -> o_proj -> [B,T,H]
        let out = out.transpose(1, 2)?.reshape((b, t, self.num_heads * d))?;
        out.apply(&self.o_proj)
    }
}

pub struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,          // dim=hidden_size
    post_attention_layernorm: Qwen3RMSNorm, // dim=hidden_size
}

impl Qwen3DecoderLayer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen3RMSNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_embeddings: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        // Pre-norm + Attention with residual
        let hs = hidden_states.apply(&self.input_layernorm)?;
        let hs = self
            .self_attn
            .forward(&hs, position_embeddings, attention_mask)?;
        let hs = hidden_states.add(&hs)?;

        // MLP with residual
        let hs2 = hs.apply(&self.post_attention_layernorm)?;
        let hs2 = hs2.apply(&self.mlp)?;
        hs.add(&hs2)
    }
}

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    cfg: Config,
    device: Device,
}

impl Qwen3Model {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(&cfg, vb.pp(format!("layers.{i}")))?);
        }

        let norm = Qwen3RMSNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = Qwen3RotaryEmbedding::new(&cfg, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            cfg,
            device,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_with_inputs_embeds(
        &self,
        inputs_embeds: &Tensor,
        attention_mask_4d: Option<&Tensor>,
        deepstack_additions: Option<&[Tensor]>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = inputs_embeds.dims3()?;
        let mut hs = inputs_embeds.clone();

        let attention_mask_4d_cast = match attention_mask_4d {
            Some(m) => Some(m.to_dtype(hs.dtype())?),
            None => None,
        };

        let default_position_ids = if position_ids.is_none() {
            let pos_1d = Tensor::arange(0u32, t as u32, hs.device())?;
            Some(pos_1d.unsqueeze(0)?.expand((b, t))?.contiguous()?)
        } else {
            None
        };
        let position_ids = if let Some(position_ids) = position_ids {
            position_ids
        } else {
            default_position_ids.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("missing default position ids".to_string())
            })?
        };

        let (cos, sin) = self.rotary_emb.forward(&hs, &position_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hs = layer.forward(&hs, attention_mask_4d_cast.as_ref(), (&cos, &sin))?;
            if let Some(additions) = deepstack_additions {
                if let Some(visual_add) = additions.get(layer_idx) {
                    hs = hs.add(visual_add)?;
                }
            }
        }

        hs.apply(&self.norm)
    }
    /// input_ids: [B,T]
    /// attention_mask: optionally [B,T] (1=token,0=pad) OR already expanded additive mask [B,1,T,T]
    /// Here we assume additive mask [B,1,T,T] for simplicity (HF does that).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask_4d: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hs = self.embed_tokens(input_ids)?;
        self.forward_with_inputs_embeds(&hs, attention_mask_4d, None, None)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn get_gguf_tensor(
        name: &str,
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        // Pass `device` as the 3rd argument
        let qt = content
            .tensor(file, name, device)
            .map_err(|e| candle_core::Error::Msg(format!("Missing {name}: {e}")))?;
        qt.dequantize(device)?.to_dtype(dtype)
    }

    fn get_gguf_qlinear(
        name: &str,
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<QLinear> {
        // Pass `device` as the 3rd argument
        let qt = content
            .tensor(file, &format!("{name}.weight"), device)
            .map_err(|e| candle_core::Error::Msg(format!("Missing {name}.weight: {e}")))?;
        let inner = QMatMul::from_qtensor(qt)?;

        // Pass `device` as the 3rd argument here as well
        if let Ok(qt_bias) = content.tensor(file, &format!("{name}.bias"), device) {
            let bias = qt_bias.dequantize(device)?.to_dtype(dtype)?;
            Ok(QLinear::QuantizedWithBias(inner, bias))
        } else {
            Ok(QLinear::Quantized(inner))
        }
    }

    pub fn from_gguf(
        cfg: Config,
        content: &gguf_file::Content,
        file: &mut std::fs::File,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Self> {
        let tok_emb = Self::get_gguf_tensor("token_embd.weight", content, file, device, dtype)?;
        let embed_tokens = candle_nn::Embedding::new(tok_emb, cfg.hidden_size);

        let norm_weight =
            Self::get_gguf_tensor("output_norm.weight", content, file, device, dtype)?;
        let norm = Qwen3RMSNorm::from_tensor(norm_weight, cfg.rms_norm_eps);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let q_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.attn_q"), content, file, device, dtype)?;
            let k_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.attn_k"), content, file, device, dtype)?;
            let v_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.attn_v"), content, file, device, dtype)?;
            let o_proj = Self::get_gguf_qlinear(
                &format!("blk.{i}.attn_output"),
                content,
                file,
                device,
                dtype,
            )?;

            let q_norm_w = Self::get_gguf_tensor(
                &format!("blk.{i}.attn_q_norm.weight"),
                content,
                file,
                device,
                dtype,
            )?;
            let k_norm_w = Self::get_gguf_tensor(
                &format!("blk.{i}.attn_k_norm.weight"),
                content,
                file,
                device,
                dtype,
            )?;

            let gate_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.ffn_gate"), content, file, device, dtype)?;
            let up_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.ffn_up"), content, file, device, dtype)?;
            let down_proj =
                Self::get_gguf_qlinear(&format!("blk.{i}.ffn_down"), content, file, device, dtype)?;

            let input_norm_w = Self::get_gguf_tensor(
                &format!("blk.{i}.attn_norm.weight"),
                content,
                file,
                device,
                dtype,
            )?;
            let post_attn_norm_w = Self::get_gguf_tensor(
                &format!("blk.{i}.ffn_norm.weight"),
                content,
                file,
                device,
                dtype,
            )?;

            layers.push(Qwen3DecoderLayer {
                self_attn: Qwen3Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Qwen3RMSNorm::from_tensor(q_norm_w, cfg.rms_norm_eps),
                    k_norm: Qwen3RMSNorm::from_tensor(k_norm_w, cfg.rms_norm_eps),
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_kv_groups: cfg.num_kv_groups(),
                    head_dim: cfg.head_dim(),
                    scaling: (cfg.head_dim() as f32).powf(-0.5),
                },
                mlp: Qwen3MLP {
                    gate_proj,
                    up_proj,
                    down_proj,
                    act_fn: cfg.hidden_act,
                },
                input_layernorm: Qwen3RMSNorm::from_tensor(input_norm_w, cfg.rms_norm_eps),
                post_attention_layernorm: Qwen3RMSNorm::from_tensor(
                    post_attn_norm_w,
                    cfg.rms_norm_eps,
                ),
            });
        }

        let rotary_emb = Qwen3RotaryEmbedding::new(&cfg, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            cfg,
            device: device.clone(),
        })
    }

    pub fn load_vision_mmproj_varbuilder<'a>(
        path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<candle_nn::VarBuilder<'a>> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut tensors = std::collections::HashMap::new();

        // We need to catch the two halves of the 3D patch embedding kernel
        let mut patch_embd_0 = None;
        let mut patch_embd_1 = None;

        for (name, _) in content.tensor_infos.iter() {
            let qtensor = content
                .tensor(&mut file, name, device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let tensor = qtensor.dequantize(device)?.to_dtype(dtype)?;

            // Intercept the split patch embedding weights
            if name == "v.patch_embd.weight" {
                patch_embd_0 = Some(tensor);
                continue;
            } else if name == "v.patch_embd.weight.1" {
                patch_embd_1 = Some(tensor);
                continue;
            }

            let mut mapped_name = name.to_string();

            // 1. Patch Embed Bias
            if mapped_name == "v.patch_embd.bias" {
                mapped_name = "patch_embed.proj.bias".to_string();
            }
            // 2. Position Embeddings
            else if mapped_name == "v.position_embd.weight" {
                mapped_name = "pos_embed.weight".to_string();
            }
            // 3. Final Merger (mm.0 = fc1, mm.2 = fc2, post_ln = norm)
            else if mapped_name.starts_with("mm.0.") {
                mapped_name = mapped_name.replace("mm.0.", "merger.linear_fc1.");
            } else if mapped_name.starts_with("mm.2.") {
                mapped_name = mapped_name.replace("mm.2.", "merger.linear_fc2.");
            } else if mapped_name == "v.post_ln.weight" {
                mapped_name = "merger.norm.weight".to_string();
            } else if mapped_name == "v.post_ln.bias" {
                mapped_name = "merger.norm.bias".to_string();
            }
            // 4. Deepstack Mergers
            else if mapped_name.starts_with("v.deepstack.") {
                mapped_name = mapped_name
                    .replace("v.deepstack.5.", "deepstack_merger_list.0.")
                    .replace("v.deepstack.11.", "deepstack_merger_list.1.")
                    .replace("v.deepstack.17.", "deepstack_merger_list.2.")
                    .replace(".fc1.", ".linear_fc1.")
                    .replace(".fc2.", ".linear_fc2.");
            }
            // 5. Vision Blocks
            else if mapped_name.starts_with("v.blk.") {
                mapped_name = mapped_name
                    .replace("v.blk.", "blocks.")
                    .replace(".attn_out.", ".attn.proj.")
                    .replace(".attn_qkv.", ".attn.qkv.")
                    .replace(".ffn_up.", ".mlp.linear_fc1.") // llama.cpp up = safetensors fc1
                    .replace(".ffn_down.", ".mlp.linear_fc2.") // llama.cpp down = safetensors fc2
                    .replace(".ln1.", ".norm1.")
                    .replace(".ln2.", ".norm2.");
            }

            tensors.insert(mapped_name, tensor);
        }

        // 6. Reconstruct the 3D Patch Embedding Kernel by stacking the temporal dimension
        if let Some(t0) = patch_embd_0 {
            if let Some(t1) = patch_embd_1 {
                let t0 = t0.unsqueeze(2)?;
                let t1 = t1.unsqueeze(2)?;
                let stacked = candle_core::Tensor::cat(&[&t0, &t1], 2)?;
                tensors.insert("patch_embed.proj.weight".to_string(), stacked);
            } else {
                tensors.insert("patch_embed.proj.weight".to_string(), t0);
            }
        } else {
            return Err(candle_core::Error::Msg(
                "Missing v.patch_embd.weight in the mmproj file!".to_string(),
            ));
        }

        Ok(candle_nn::VarBuilder::from_tensors(tensors, dtype, device))
    }
}

/// High-level embedding wrapper around [`Qwen3Model`] that handles tokenization,
/// batching, mean-pooling, and L2-normalization.
pub struct Qwen3TextEmbedding {
    model: Qwen3Model,
    tokenizer: tokenizers::Tokenizer,
}

impl Qwen3TextEmbedding {
    /// Build from an already loaded model and tokenizer.
    pub fn new(model: Qwen3Model, tokenizer: tokenizers::Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load from a Hugging Face repo.
    ///
    /// Supported model families:
    /// - Qwen3 text embedding checkpoints (e.g. `Qwen/Qwen3-Embedding-0.6B`)
    /// - Qwen3-VL embedding checkpoints in text-only mode
    ///   (e.g. `Qwen/Qwen3-VL-Embedding-2B`)
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        // Load config
        let cfg_path: PathBuf = repo
            .get("config.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cfg_bytes =
            std::fs::read(&cfg_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let (cfg, weight_prefix) = parse_config_and_weight_prefix(&cfg_bytes)?;

        // Load weights (single or sharded)
        let weight_files: Vec<PathBuf> = if let Ok(p) = repo.get("model.safetensors") {
            vec![p]
        } else {
            let mut files = Vec::new();
            for i in 1.. {
                let candidates: Vec<_> = (1..=20)
                    .filter_map(|total| {
                        let fname = format!("model-{:05}-of-{:05}.safetensors", i, total);
                        repo.get(&fname).ok()
                    })
                    .collect();
                if candidates.is_empty() {
                    break;
                }
                files.extend(candidates.into_iter().take(1));
            }
            if files.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Could not locate model.safetensors or sharded weight files".into(),
                ));
            }
            files
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let vb = match weight_prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let model = Qwen3Model::new(cfg, vb)?;

        // Load tokenizer
        let tok_path: PathBuf = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Use LEFT padding so the last position is always the real last token.
        // This simplifies last-token pooling and matches the official implementation.
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        // Note: Unlike some other embedding models, Qwen3-Embedding does NOT
        // add an EOS token. The last_token_pool simply takes the last real token.

        Ok(Self { model, tokenizer })
    }

    /// Load from a local directory containing model files.
    ///
    /// Required files in directory:
    /// - config.json
    /// - model.safetensors (or model-XXXXX-of-XXXXX.safetensors)
    /// - tokenizer.json
    pub fn from_path(
        model_dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let model_dir = model_dir.as_ref();

        let cfg_bytes = std::fs::read(model_dir.join("config.json"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let (cfg, weight_prefix) = parse_config_and_weight_prefix(&cfg_bytes)?;

        let weight_files = find_safetensor_files(model_dir)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let vb = match weight_prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let model = Qwen3Model::new(cfg, vb)?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        Ok(Self { model, tokenizer })
    }

    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        model_dir_for_configs: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let mut file = std::fs::File::open(gguf_path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let cfg_bytes = std::fs::read(model_dir_for_configs.as_ref().join("config.json"))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let (cfg, _) = parse_config_and_weight_prefix(&cfg_bytes)?;

        let model = Qwen3Model::from_gguf(cfg, &content, &mut file, device, dtype)?;

        let mut tokenizer =
            tokenizers::Tokenizer::from_file(model_dir_for_configs.as_ref().join("tokenizer.json"))
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Left,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        Ok(Self { model, tokenizer })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Embed a batch of texts, returning normalized embeddings.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.iter().map(|s| s.as_ref()).collect::<Vec<_>>(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        let mut input_ids_vec: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            input_ids_vec.extend(enc.get_ids().iter().copied());
            attention_mask_vec.extend(enc.get_attention_mask().iter().map(|&m| m as f32));
        }

        let device = self.model.device();
        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;

        let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)?;

        // Forward pass -> [B, T, H]
        let hidden = self.model.forward(&input_ids, Some(&attention_mask_4d))?;

        // Last token pooling: with left padding, the last position is always the real last token
        let pooled = hidden.i((.., seq_len - 1))?; // [B, H]

        // L2 normalize (add epsilon for numerical stability)
        let normalized = l2_normalize(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let normalized = normalized.to_dtype(DType::F32)?;
        let data = normalized.to_vec2::<f32>()?;
        Ok(data)
    }
}

/// Multimodal embedding wrapper for Qwen3-VL embedding checkpoints.
///
/// This supports text-only and image-only inputs. Image+text mixed inputs can be
/// added later with the same backend.
pub struct Qwen3VLEmbedding {
    model: Qwen3Model,
    vision: Qwen3VLVisionModel,
    tokenizer: tokenizers::Tokenizer,
    preprocessor: Qwen3VLPreprocessorConfig,
    image_token_id: u32,
    default_instruction: String,
}

impl Qwen3VLEmbedding {
    /// Load a Qwen3-VL embedding model from Hugging Face.
    #[cfg(feature = "hf-hub")]
    pub fn from_hf(
        repo_id: &str,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams};

        let api = ApiBuilder::new()
            .with_progress(true)
            .build()
            .map_err(map_err)?;
        let repo = api.model(repo_id.to_string());

        let cfg_path: PathBuf = repo.get("config.json").map_err(map_err)?;
        let cfg_bytes = std::fs::read(&cfg_path).map_err(map_err)?;
        let cfg: Qwen3VLFullConfig = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let preproc_path: PathBuf = repo.get("preprocessor_config.json").map_err(map_err)?;
        let preprocessor_bytes = std::fs::read(&preproc_path).map_err(map_err)?;
        let preprocessor: Qwen3VLPreprocessorConfig =
            serde_json::from_slice(&preprocessor_bytes).map_err(map_err)?;

        let weight_files: Vec<PathBuf> = if let Ok(p) = repo.get("model.safetensors") {
            vec![p]
        } else {
            let mut files = Vec::new();
            for i in 1.. {
                let candidates: Vec<_> = (1..=20)
                    .filter_map(|total| {
                        let fname = format!("model-{:05}-of-{:05}.safetensors", i, total);
                        repo.get(&fname).ok()
                    })
                    .collect();
                if candidates.is_empty() {
                    break;
                }
                files.extend(candidates.into_iter().take(1));
            }
            if files.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Could not locate model.safetensors or sharded weight files".into(),
                ));
            }
            files
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = Qwen3Model::new(cfg.text_config, vb.pp("model").pp("language_model"))?;
        let vision = Qwen3VLVisionModel::new(&cfg.vision_config, vb.pp("model").pp("visual"))?;

        let tok_path: PathBuf = repo.get("tokenizer.json").map_err(map_err)?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path).map_err(map_err)?;
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        Ok(Self {
            model,
            vision,
            tokenizer,
            preprocessor,
            image_token_id: cfg.image_token_id,
            default_instruction: "Represent the user's input.".to_string(),
        })
    }

    /// Load from a local directory containing model files.
    ///
    /// Required files in directory:
    /// - config.json
    /// - model.safetensors (or model-XXXXX-of-XXXXX.safetensors)
    /// - tokenizer.json
    /// - preprocessor_config.json
    pub fn from_path(
        model_dir: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        max_length: usize,
    ) -> Result<Self> {
        use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams};

        let model_dir = model_dir.as_ref();

        let cfg_bytes = std::fs::read(model_dir.join("config.json")).map_err(map_err)?;
        let cfg: Qwen3VLFullConfig = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let preprocessor_bytes =
            std::fs::read(model_dir.join("preprocessor_config.json")).map_err(map_err)?;
        let preprocessor: Qwen3VLPreprocessorConfig =
            serde_json::from_slice(&preprocessor_bytes).map_err(map_err)?;

        let weight_files = find_safetensor_files(model_dir)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, device)? };
        let model = Qwen3Model::new(cfg.text_config.clone(), vb.pp("model").pp("language_model"))?;
        let vision = Qwen3VLVisionModel::new(&cfg.vision_config, vb.pp("model").pp("visual"))?;

        let mut tokenizer =
            tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(map_err)?;
        let _ = tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            ..Default::default()
        }));
        let _ = tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }));

        Ok(Self {
            model,
            vision,
            tokenizer,
            preprocessor,
            image_token_id: cfg.image_token_id,
            default_instruction: "Represent the user's input.".to_string(),
        })
    }

    pub fn from_gguf_and_mmproj(
        text_gguf_path: impl AsRef<std::path::Path>,
        mmproj_path: impl AsRef<std::path::Path>,
        model_dir_for_configs: impl AsRef<std::path::Path>, // Needs the dir for configs/tokenizer
        device: &Device,
        dtype: DType, // Pass DType::F32 or DType::BF16
    ) -> candle_core::Result<Self> {
        // 1. Load Configurations
        let model_dir = model_dir_for_configs.as_ref();
        let cfg_bytes = std::fs::read(model_dir.join("config.json")).map_err(map_err)?;
        let cfg: Qwen3VLFullConfig = serde_json::from_slice(&cfg_bytes).map_err(map_err)?;

        let preprocessor_bytes =
            std::fs::read(model_dir.join("preprocessor_config.json")).map_err(map_err)?;
        let preprocessor: Qwen3VLPreprocessorConfig =
            serde_json::from_slice(&preprocessor_bytes).map_err(map_err)?;

        // 2. Load Quantized Text GGUF
        let mut text_file = std::fs::File::open(text_gguf_path.as_ref())?;
        let text_content = gguf_file::Content::read(&mut text_file).map_err(map_err)?;
        let text_model = Qwen3Model::from_gguf(
            cfg.text_config.clone(),
            &text_content,
            &mut text_file,
            device,
            dtype,
        )?;

        // 3. Load F16 Vision mmproj
        let vision_vb =
            Qwen3Model::load_vision_mmproj_varbuilder(mmproj_path.as_ref(), device, dtype)?;
        let vision_model = Qwen3VLVisionModel::new(&cfg.vision_config, vision_vb)?;

        // 4. Tokenizer
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(map_err)?;
        let _ = tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            ..Default::default()
        }));

        Ok(Self {
            model: text_model,
            vision: vision_model,
            tokenizer,
            preprocessor,
            image_token_id: cfg.image_token_id,
            default_instruction: "Represent the user's input.".to_string(),
        })
    }

    pub fn config(&self) -> &Config {
        self.model.config()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn max_pixels(&self) -> usize {
        self.preprocessor.max_pixels
    }

    pub fn min_pixels(&self) -> usize {
        self.preprocessor.min_pixels
    }

    pub fn set_max_pixels(&mut self, max_pixels: usize) {
        self.preprocessor.max_pixels = max_pixels;
    }

    pub fn set_min_pixels(&mut self, min_pixels: usize) {
        self.preprocessor.min_pixels = min_pixels;
    }

    pub fn embed_batch<S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        visuals: &[Option<VisualInput>],
        texts: &[Option<S1>],
        instructions: &[Option<S2>],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = visuals.len();
        if texts.len() != batch_size || instructions.len() != batch_size {
            return Err(candle_core::Error::Msg(
                "visuals, texts, and instructions must have the same length".into(),
            ));
        }

        let mut prepared_inputs = Vec::with_capacity(batch_size);

        for visual in visuals {
            let prepared = match visual {
                Some(VisualInput::Image(img)) => Some(preprocess_image(img, &self.preprocessor)?),
                Some(VisualInput::Video(frames)) => {
                    if frames.is_empty() {
                        return Err(candle_core::Error::Msg("Empty video frames".into()));
                    }
                    Some(preprocess_video(frames, &self.preprocessor)?)
                }
                None => None,
            };
            prepared_inputs.push(prepared);
        }

        let mut prompts = Vec::with_capacity(batch_size);

        for (i, prepared) in prepared_inputs.iter().enumerate() {
            let user_text = texts[i].as_ref().map(|s| s.as_ref());

            let inst = prepare_instruction(
                instructions[i].as_ref().map(|s| s.as_ref()),
                &self.default_instruction,
            );

            let mut prompt = build_vl_prompt(user_text, prepared.is_some(), &inst);

            if let Some(p) = prepared {
                prompt = expand_image_token_placeholders(&prompt, p.num_llm_tokens)?;
            }
            prompts.push(prompt);
        }

        // Inference
        self.run_inference_on_prepared(prompts, prepared_inputs)
    }

    /// Embed a batch of texts with custom instructions for each.
    /// You can pass `None` for a specific item to use the default instruction.
    pub fn embed_texts_with_instructions<S1: AsRef<str>, S2: AsRef<str>>(
        &self,
        texts: &[S1],
        instructions: &[Option<S2>],
    ) -> Result<Vec<Vec<f32>>> {
        if texts.len() != instructions.len() {
            return Err(candle_core::Error::Msg(
                "texts and instructions must have the same length".into(),
            ));
        }

        let text_inputs: Vec<Option<String>> =
            texts.iter().map(|t| Some(t.as_ref().to_string())).collect();
        let image_inputs: Vec<Option<DynamicImage>> = (0..texts.len()).map(|_| None).collect();

        let inst_inputs: Vec<Option<String>> = instructions
            .iter()
            .map(|inst| inst.as_ref().map(|s| s.as_ref().to_string()))
            .collect();

        self.embed_internal(text_inputs, image_inputs, inst_inputs)
    }

    /// Embed a batch of image paths with custom instructions for each.
    pub fn embed_images_with_instructions<S1: AsRef<Path>, S2: AsRef<str>>(
        &self,
        images: &[S1],
        instructions: &[Option<S2>],
    ) -> Result<Vec<Vec<f32>>> {
        if images.len() != instructions.len() {
            return Err(candle_core::Error::Msg(
                "images and instructions must have the same length".into(),
            ));
        }

        let mut image_inputs = Vec::with_capacity(images.len());
        for path in images {
            image_inputs.push(Some(load_image_from_path(path.as_ref())?));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();

        let inst_inputs: Vec<Option<String>> = instructions
            .iter()
            .map(|inst| inst.as_ref().map(|s| s.as_ref().to_string()))
            .collect();

        self.embed_internal(text_inputs, image_inputs, inst_inputs)
    }

    /// Embed a list of videos. Each video is represented as a list of frames (images).
    /// Returns embeddings for each video.
    pub fn embed_video_frames<S: AsRef<str>>(
        &self,
        videos: &[Vec<DynamicImage>],
        instructions: &[Option<S>],
    ) -> Result<Vec<Vec<f32>>> {
        if videos.len() != instructions.len() {
            return Err(candle_core::Error::Msg(
                "videos and instructions must have the same length".into(),
            ));
        }

        if videos.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Video preprocessing
        let mut prepared_videos = Vec::with_capacity(videos.len());
        for frames in videos {
            if frames.is_empty() {
                prepared_videos.push(None);
            } else {
                let prepared = preprocess_video(frames, &self.preprocessor)?;
                prepared_videos.push(Some(prepared));
            }
        }

        // 2. Prompting
        let mut prompts = Vec::with_capacity(videos.len());
        for (i, prepared) in prepared_videos.iter().enumerate() {
            let inst = prepare_instruction(
                instructions[i].as_ref().map(|s| s.as_ref()),
                &self.default_instruction,
            );

            let mut prompt = build_vl_prompt(None, prepared.is_some(), &inst);

            if let Some(p) = prepared {
                prompt = expand_image_token_placeholders(&prompt, p.num_llm_tokens)?;
            }
            prompts.push(prompt);
        }

        // 3. Inference
        self.run_inference_on_prepared(prompts, prepared_videos)
    }

    /// Embed a batch of texts using Qwen3-VL prompt formatting.
    pub fn embed_texts<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Vec<f32>>> {
        let text_inputs: Vec<Option<String>> =
            texts.iter().map(|t| Some(t.as_ref().to_string())).collect();
        let image_inputs: Vec<Option<DynamicImage>> = (0..texts.len()).map(|_| None).collect();
        let instructions: Vec<Option<String>> = vec![None; texts.len()];
        self.embed_internal(text_inputs, image_inputs, instructions)
    }

    /// Embed a batch of image paths.
    pub fn embed_images<S: AsRef<Path>>(&self, images: &[S]) -> Result<Vec<Vec<f32>>> {
        let mut image_inputs = Vec::with_capacity(images.len());
        for path in images {
            image_inputs.push(Some(load_image_from_path(path.as_ref())?));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();
        let instructions: Vec<Option<String>> = vec![None; images.len()];
        self.embed_internal(text_inputs, image_inputs, instructions)
    }

    /// Embed a batch of image bytes.
    pub fn embed_image_bytes(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        let mut image_inputs = Vec::with_capacity(images.len());
        for bytes in images {
            let image = image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .map_err(map_err)?
                .decode()
                .map_err(map_err)?;
            image_inputs.push(Some(image));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();
        let instructions: Vec<Option<String>> = vec![None; images.len()];
        self.embed_internal(text_inputs, image_inputs, instructions)
    }

    /// Embed a batch of image bytes with custom instructions for each.
    pub fn embed_image_bytes_with_instructions<S: AsRef<str>>(
        &self,
        images: &[&[u8]],
        instructions: &[Option<S>],
    ) -> Result<Vec<Vec<f32>>> {
        if images.len() != instructions.len() {
            return Err(candle_core::Error::Msg(
                "images and instructions must have the same length".into(),
            ));
        }

        let mut image_inputs = Vec::with_capacity(images.len());
        for bytes in images {
            let image = image::ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .map_err(map_err)?
                .decode()
                .map_err(map_err)?;
            image_inputs.push(Some(image));
        }
        let text_inputs: Vec<Option<String>> = (0..images.len()).map(|_| None).collect();

        let inst_inputs: Vec<Option<String>> = instructions
            .iter()
            .map(|inst| inst.as_ref().map(|s| s.as_ref().to_string()))
            .collect();

        self.embed_internal(text_inputs, image_inputs, inst_inputs)
    }

    fn run_inference_on_prepared(
        &self,
        prompts: Vec<String>,
        prepared_inputs: Vec<Option<PreparedImage>>,
    ) -> Result<Vec<Vec<f32>>> {
        let device = self.model.device();

        // 1. Tokenizing text
        let encodings = self
            .tokenizer
            .encode_batch(prompts, true)
            .map_err(map_err)?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].len();

        // 2. Finding where to paste images to text (<|image_pad|>)
        let mut image_spans_per_batch = Vec::with_capacity(batch_size);
        for (batch_idx, encoding) in encodings.iter().enumerate() {
            let Some(prepared) = prepared_inputs[batch_idx].as_ref() else {
                image_spans_per_batch.push(None);
                continue;
            };

            let spans = find_token_spans(encoding.get_ids(), self.image_token_id);
            if spans.len() != 1 {
                return Err(candle_core::Error::Msg(
                    "Expected exactly one image token span per image/video input".into(),
                ));
            }
            let (start, end) = spans[0];
            let span_len = end - start;
            if span_len != prepared.num_llm_tokens {
                return Err(candle_core::Error::Msg(format!(
                    "Token span mismatch: prompt reserved {}, but visual feature needs {}",
                    span_len, prepared.num_llm_tokens
                )));
            }
            image_spans_per_batch.push(Some((start, end)));
        }

        // 3. Preparing text tensors (input_ids, attention_mask)
        let mut input_ids_vec = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len);
        for encoding in &encodings {
            input_ids_vec.extend(encoding.get_ids().iter().copied());
            attention_mask_vec.extend(encoding.get_attention_mask().iter().map(|&m| m as f32));
        }

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), device)?;
        let attention_mask_2d =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), device)?;

        // Getting text embeddings
        let mut inputs_embeds = self.model.embed_tokens(&input_ids)?;
        let hidden_size = self.model.config().hidden_size;

        let mut deepstack_additions: Option<Vec<Tensor>> = None;
        let mut position_ids: Option<Tensor> = None;

        // 4. Vision preparing (if any)
        let num_inputs = prepared_inputs.iter().filter(|p| p.is_some()).count();
        if num_inputs > 0 {
            let patch_dim = 3
                * self.preprocessor.temporal_patch_size
                * self.preprocessor.patch_size
                * self.preprocessor.patch_size;

            let mut pixel_values = Vec::new();
            let mut grid_thw = Vec::new();

            // Collecting all pixels of all images to one
            for prepared in prepared_inputs.iter().flatten() {
                pixel_values.extend_from_slice(&prepared.pixel_values);
                grid_thw.extend_from_slice(&[prepared.grid_t, prepared.grid_h, prepared.grid_w]);
            }

            let num_patch_tokens = pixel_values.len() / patch_dim;
            let pixel_values =
                Tensor::from_vec(pixel_values, (num_patch_tokens, patch_dim), device)?
                    .to_dtype(inputs_embeds.dtype())?;

            let image_grid_thw = Tensor::from_vec(grid_thw, (num_inputs, 3), device)?;

            // mRoPE
            position_ids = Some(build_image_position_ids(
                &encodings,
                &image_spans_per_batch,
                &prepared_inputs,
                self.preprocessor.merge_size,
                device,
            )?);

            // Vision Tower
            let (vision_embeds, deepstack_image_embeds) =
                self.vision.forward(&pixel_values, &image_grid_thw)?;

            // 5. Injection
            let mut offset = 0usize;

            for (batch_idx, image_span) in image_spans_per_batch.iter().enumerate() {
                let Some((start, end)) = image_span else {
                    continue;
                };
                let span_len = end - start;

                let image_chunk = vision_embeds.narrow(0, offset, span_len)?;
                offset += span_len;

                // Replacing placeholders with real vectors of image
                inputs_embeds = inputs_embeds.slice_assign(
                    &[batch_idx..batch_idx + 1, *start..*end, 0..hidden_size],
                    &image_chunk.unsqueeze(0)?,
                )?;
            }

            if offset != vision_embeds.dim(0)? {
                return Err(candle_core::Error::Msg(
                    "Unconsumed image embeddings remain after token injection".into(),
                ));
            }

            // Deepstack (if used) - this allocates full tensors but is needed for LLM addition
            if !deepstack_image_embeds.is_empty() {
                let mut per_layer_additions = Vec::with_capacity(deepstack_image_embeds.len());
                for deepstack_layer in deepstack_image_embeds {
                    let mut addition = Tensor::zeros(
                        (batch_size, seq_len, hidden_size),
                        deepstack_layer.dtype(),
                        device,
                    )?;
                    let mut deep_offset = 0usize;
                    for (batch_idx, image_span) in image_spans_per_batch.iter().enumerate() {
                        let Some((start, end)) = image_span else {
                            continue;
                        };
                        let span_len = end - start;
                        let chunk = deepstack_layer.narrow(0, deep_offset, span_len)?;
                        deep_offset += span_len;
                        addition = addition.slice_assign(
                            &[batch_idx..batch_idx + 1, *start..*end, 0..hidden_size],
                            &chunk.unsqueeze(0)?,
                        )?;
                    }
                    per_layer_additions.push(addition);
                }
                deepstack_additions = Some(per_layer_additions);
            }
        }

        // 6. Final LLM pass
        let attention_mask_4d = build_attention_mask_4d(&attention_mask_2d)?;

        let hidden = self.model.forward_with_inputs_embeds(
            &inputs_embeds,
            Some(&attention_mask_4d),
            deepstack_additions.as_deref(),
            position_ids.as_ref(),
        )?;

        // 7. Pooling
        let pooled = last_token_pool(&hidden, &attention_mask_2d)?;
        let normalized = l2_normalize(&pooled)?.to_dtype(DType::F32)?;
        normalized.to_vec2::<f32>()
    }

    fn embed_internal(
        &self,
        texts: Vec<Option<String>>,
        images: Vec<Option<DynamicImage>>,
        instructions: Vec<Option<String>>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.len() != images.len() || texts.len() != instructions.len() {
            return Err(candle_core::Error::Msg("Batch sizes mismatch".into()));
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut prepared_inputs = Vec::with_capacity(images.len());
        for image in &images {
            prepared_inputs.push(match image {
                Some(img) => Some(preprocess_image(img, &self.preprocessor)?),
                None => None,
            });
        }

        let mut prompts = Vec::with_capacity(texts.len());
        for (i, (text, prepared)) in texts.iter().zip(prepared_inputs.iter()).enumerate() {
            let inst = prepare_instruction(instructions[i].as_deref(), &self.default_instruction);
            let mut prompt = build_vl_prompt(text.as_deref(), prepared.is_some(), &inst);
            if let Some(prepared) = prepared {
                prompt = expand_image_token_placeholders(&prompt, prepared.num_llm_tokens)?;
            }
            prompts.push(prompt);
        }

        self.run_inference_on_prepared(prompts, prepared_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expand_image_token_placeholders, find_token_spans, parse_config_and_weight_prefix,
        round_ties_to_even,
    };

    #[test]
    fn parses_qwen3_config_without_prefix() {
        let config = r#"{
            "attention_bias": false,
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_scaling": null,
            "rope_theta": 1000000,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151669
        }"#;

        let (cfg, prefix) = parse_config_and_weight_prefix(config.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(prefix, None);
    }

    #[test]
    fn parses_qwen3_vl_text_config_with_language_model_prefix() {
        let config = r#"{
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 262144,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                },
                "rope_theta": 5000000,
                "tie_word_embeddings": true,
                "use_cache": true,
                "vocab_size": 151936
            },
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653
        }"#;

        let (cfg, prefix) = parse_config_and_weight_prefix(config.as_bytes()).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(prefix, Some("model.language_model"));
    }

    #[test]
    fn finds_token_spans() {
        let ids = vec![1u32, 42, 42, 7, 42, 42, 42, 9];
        let spans = find_token_spans(&ids, 42);
        assert_eq!(spans, vec![(1, 3), (4, 7)]);
    }

    #[test]
    fn expands_image_placeholders() {
        let prompt = "<|vision_start|><|image_pad|><|vision_end|>";
        let expanded = expand_image_token_placeholders(prompt, 3).unwrap();
        assert_eq!(
            expanded,
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>"
        );
    }

    #[test]
    fn rounds_half_to_even_like_python() {
        assert_eq!(round_ties_to_even(12.5), 12);
        assert_eq!(round_ties_to_even(13.5), 14);
        assert_eq!(round_ties_to_even(9.5625), 10);
    }
}
