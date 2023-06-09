# Deep learning layers

EDS-PDF provides a set of specialized deep learning layers that can be used to build trainable
components. These layers are built on top of the PyTorch framework and can be used in
any PyTorch model.

| Layer                                                                           | Description                                                                                 |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`BoxTransformerModule`][edspdf.layers.box_transformer.BoxTransformerModule]    | Contextualize box embeddings with a 2d Transformer with relative position representations   |
| [`BoxTransformerLayer`][edspdf.layers.box_transformer.BoxTransformerLayer]      | A single layer of the above `BoxTransformerModule` layer                                    |
| [`RelativeAttention`][edspdf.layers.relative_attention.RelativeAttention]       | A 2d attention layer that optionally uses relative position to compute its attention scores |
| [`SinusoidalEmbedding`][edspdf.layers.sinusoidal_embedding.SinusoidalEmbedding] | A position embedding that uses trigonometric functions to encode positions                  |
| [`Vocabulary`][edspdf.layers.vocabulary.Vocabulary]                             | A non deep learning layer to encodes / decode vocabularies                                  |
