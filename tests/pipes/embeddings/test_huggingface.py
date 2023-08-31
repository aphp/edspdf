from edspdf.pipes.embeddings.huggingface_embedding import HuggingfaceEmbedding


def test_huggingface_embedding(pdfdoc):
    embedding = HuggingfaceEmbedding(
        pipeline=None,
        name="huggingface",
        model="hf-tiny-model-private/tiny-random-LayoutLMv3Model",
        window=32,
        stride=16,
        use_image=True,
    )
    # Patch the faulty size in the tiny-random-LayoutLMv3Model
    embedding.image_processor.size = {
        "height": embedding.hf_model.config.input_size,
        "width": embedding.hf_model.config.input_size,
    }
    embedding(pdfdoc)
