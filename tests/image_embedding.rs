mod common;

#[test]
#[ignore]
fn test_load_model_and_embed_image() {
    let embedder = common::load_embedder().expect("Failed to load model");

    let image_path = common::get_test_image_path();
    println!("Embedding image: {}", image_path);

    let embeddings = embedder
        .embed_images(&[&image_path])
        .expect("Failed to embed image");

    assert_eq!(embeddings.len(), 1, "Should return exactly one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );

    println!("Image embedding successful!");
    println!("  Embedding dimension: {}", embeddings[0].len());
    println!(
        "  First 5 values: {:?}",
        &embeddings[0][..5.min(embeddings[0].len())]
    );
}

#[test]
#[ignore]
fn test_embed_multiple_images() {
    let embedder = common::load_embedder().expect("Failed to load model");

    let image_path = common::get_test_image_path();

    let images: Vec<&str> = vec![&image_path, &image_path, &image_path];

    println!("Embedding {} images...", images.len());

    let embeddings = embedder
        .embed_images(&images)
        .expect("Failed to embed images");

    assert_eq!(
        embeddings.len(),
        images.len(),
        "Should return one embedding per image"
    );

    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "Embedding {} should not be empty", i);
    }

    println!("Multiple image embeddings successful!");
    println!("  Number of embeddings: {}", embeddings.len());
}

#[test]
#[ignore]
fn test_embed_image_with_instruction() {
    let embedder = common::load_embedder().expect("Failed to load model");

    let image_path = common::get_test_image_path();
    let instruction = "Represent this image for visual search";

    println!("Embedding image with instruction...");
    println!("  Image: {}", image_path);
    println!("  Instruction: {}", instruction);

    let embeddings = embedder
        .embed_images_with_instructions::<&str, &str>(&[&image_path], &[Some(instruction)])
        .expect("Failed to embed image with instruction");

    assert_eq!(embeddings.len(), 1, "Should return exactly one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );

    println!("Image embedding with instruction successful!");
}

#[test]
#[ignore]
fn test_embed_image_bytes() {
    let mut embedder = common::load_embedder().expect("Failed to load model");

    let image_path = common::get_test_image_path();
    let image_bytes = std::fs::read(&image_path).expect("Failed to read image file");

    println!("Embedding image from bytes: {}", image_path);
    println!("  Image size: {} bytes", image_bytes.len());

    let embeddings = embedder
        .embed_image_bytes(&[&image_bytes])
        .expect("Failed to embed image bytes");

    assert_eq!(embeddings.len(), 1, "Should return exactly one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );

    println!("Image bytes embedding successful!");
}
