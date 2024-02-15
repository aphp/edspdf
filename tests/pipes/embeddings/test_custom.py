from edspdf.pipes.embeddings.box_layout_embedding import BoxLayoutEmbedding
from edspdf.pipes.embeddings.box_transformer import BoxTransformer
from edspdf.pipes.embeddings.embedding_combiner import EmbeddingCombiner
from edspdf.pipes.embeddings.simple_text_embedding import SimpleTextEmbedding
from edspdf.pipes.embeddings.sub_box_cnn_pooler import SubBoxCNNPooler
from edspdf.pipes.extractors.pdfminer import PdfMinerExtractor


def test_custom_embedding(pdf, error_pdf, tmp_path):
    embedding = BoxTransformer(
        num_heads=4,
        dropout_p=0.1,
        activation="gelu",
        init_resweight=0.01,
        head_size=16,
        attention_mode=["c2c", "c2p", "p2c"],
        n_layers=1,
        n_relative_positions=64,
        embedding=EmbeddingCombiner(
            dropout_p=0.1,
            text_encoder=SubBoxCNNPooler(
                out_channels=64,
                kernel_sizes=(3, 4, 5),
                embedding=SimpleTextEmbedding(
                    size=72,
                ),
            ),
            layout_encoder=BoxLayoutEmbedding(
                n_positions=64,
                x_mode="sin",
                y_mode="sin",
                w_mode="learned",
                h_mode="learned",
                size=72,
            ),
        ),
    )
    str(embedding)

    extractor = PdfMinerExtractor(render_pages=True)
    pdfdoc = extractor(pdf)
    pdfdoc.text_boxes[0].text = "Very long word of 150 letters : " + "x" * 150
    embedding.post_init([pdfdoc], set())
    embedding(pdfdoc)
    embedding.save_extra_data(tmp_path, set())
    embedding.load_extra_data(tmp_path, set())

    # Test empty document
    embedding(extractor(error_pdf))
