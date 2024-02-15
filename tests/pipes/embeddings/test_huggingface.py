from edspdf.pipes.embeddings.huggingface_embedding import HuggingfaceEmbedding
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor


def test_huggingface_embedding(pdf, error_pdf):
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

    extractor = PdfMinerExtractor(render_pages=True)
    embedding(extractor(pdf))
    embedding(extractor(error_pdf))
