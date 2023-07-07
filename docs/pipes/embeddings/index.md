# Embeddings

We offer multiple embedding methods to encode the text and layout information of the PDFs. The following components can be added to a pipeline or composed together, and contain preprocessing and postprocessing logic to convert and batch documents.

<!-- --8<-- [start:components] -->

<style>
td:nth-child(1), td:nth-child(2) {
    white-space: nowrap;
}
</style>

| Factory name                                                                                  | Description                                                       |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`simple-text-embedding`][edspdf.pipes.embeddings.simple_text_embedding.SimpleTextEmbedding]  | A module that embeds the textual features of the blocks.          |
| [`embedding-combiner`][edspdf.pipes.embeddings.embedding_combiner.EmbeddingCombiner]          | Encodes boxes using a combination of multiple encoders            |
| [`sub-box-cnn-pooler`][edspdf.pipes.embeddings.sub_box_cnn_pooler.SubBoxCNNPooler]            | Pools the output of a CNN over the elements of a box (like words) |
| [`box-layout-embedding`][edspdf.pipes.embeddings.box_layout_embedding.BoxLayoutEmbedding]     | Encodes the layout of the boxes                                   |
| [`box-transformer`][edspdf.pipes.embeddings.box_transformer.BoxTransformer]                   | Contextualizes box representations using a transformer            |
| [`huggingface-embedding`][edspdf.pipes.embeddings.huggingface_embedding.HuggingfaceEmbedding] | Box representations using a Huggingface multi-modal model.        |

<!-- --8<-- [end:components] -->

!!! warning "Layers"
    These components are not to be confused with [`layers`](/layers), which are standard
    PyTorch modules that can be used to build trainable components, such as the ones
    described here.
