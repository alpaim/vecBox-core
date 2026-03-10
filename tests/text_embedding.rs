mod common;

#[test]
#[ignore]
fn test_load_model_and_embed_text() {
    let mut embedder = common::load_embedder().expect("Failed to load model");

    let text = "Hello, world! This is a test embedding.";
    println!("Embedding text: {}", text);

    let embeddings = embedder.embed_texts(&[text]).expect("Failed to embed text");

    assert_eq!(embeddings.len(), 1, "Should return exactly one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );

    println!("Text embedding successful!");
    println!("  Embedding dimension: {}", embeddings[0].len());
    println!(
        "  First 5 values: {:?}",
        &embeddings[0][..5.min(embeddings[0].len())]
    );
}

#[test]
#[ignore]
fn test_embed_multiple_texts() {
    let mut embedder = common::load_embedder().expect("Failed to load model");

    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Rust is a systems programming language.",
    ];

    println!("Embedding {} texts...", texts.len());

    let embeddings = embedder.embed_texts(&texts).expect("Failed to embed texts");

    assert_eq!(
        embeddings.len(),
        texts.len(),
        "Should return one embedding per text"
    );

    for (i, emb) in embeddings.iter().enumerate() {
        assert!(!emb.is_empty(), "Embedding {} should not be empty", i);
    }

    println!("Multiple text embeddings successful!");
    println!("  Number of embeddings: {}", embeddings.len());
}

#[test]
#[ignore]
fn test_embed_text_with_instruction() {
    let mut embedder = common::load_embedder().expect("Failed to load model");

    let text = "What is the capital of France?";
    let instruction = "Represent this question for semantic search";

    println!("Embedding text with instruction...");
    println!("  Text: {}", text);
    println!("  Instruction: {}", instruction);

    let embeddings = embedder
        .embed_texts_with_instructions::<&str, &str>(&[text], &[Some(instruction)])
        .expect("Failed to embed text with instruction");

    assert_eq!(embeddings.len(), 1, "Should return exactly one embedding");
    assert!(
        !embeddings[0].is_empty(),
        "Embedding vector should not be empty"
    );

    println!("Text embedding with instruction successful!");
}
