use vecbox_core::{download, models, utils};

pub fn get_test_config() -> (String, String) {
    let repo = std::env::var("VECBOX_HF_REPO")
        .unwrap_or_else(|_| "alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox".to_string());
    let quant = std::env::var("VECBOX_HF_QUANT").unwrap_or_else(|_| "Q4_K_M".to_string());
    (repo, quant)
}

pub fn load_embedder() -> anyhow::Result<models::qwen3::Qwen3VLEmbedding> {
    let (repo, quant) = get_test_config();

    println!("Downloading model from HuggingFace (uses cache if available)...");
    println!("  Repo: {}", repo);
    println!("  Quant: {}", quant);

    let downloaded = download::download_model(&repo, &quant)?;

    println!("Model downloaded/loaded from cache:");
    println!("  GGUF: {:?}", downloaded.gguf_path);
    println!("  MMPROJ: {:?}", downloaded.mmproj_path);

    let device = utils::get_device()?;
    let dtype = utils::get_device_dtype(&device)?;

    println!("Loading model into memory...");
    let embedder = models::qwen3::Qwen3VLEmbedding::from_gguf_and_mmproj(
        &downloaded.gguf_path,
        &downloaded.mmproj_path,
        &downloaded.config_dir,
        &device,
        dtype,
    )?;

    println!("Model loaded successfully!");
    Ok(embedder)
}

pub fn get_test_image_path() -> String {
    std::env::var("VECBOX_TEST_IMAGE")
        .unwrap_or_else(|_| "tests/fixtures/images/test_image.png".to_string())
}
