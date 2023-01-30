# Deep learning layers

EDS-PDF provides a set of deep learning layers that can be used to build trainable
components. These layers are built on top of the PyTorch framework and can be used in
any PyTorch model.

| Layer                                                                                    | Description                                                                                |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`box-embedding`][edspdf.layers.box_embedding.BoxEmbedding]                              | High level layer combining multiple box embedding layers together                          |
| [`box-layout-embedding`][edspdf.layers.box_layout_embedding.BoxLayoutEmbedding]          | Embeds the layout features (x/y/w/h) features of a box                                     |
| [`box-text-embedding`][edspdf.layers.box_text_embedding.BoxTextEmbedding]                | Embeds the textual features (shape/prefix/suffix) features of a box                        |
| [`box-layout-preprocessor`][edspdf.layers.box_layout_preprocessor.BoxLayoutPreprocessor] | Performs common preprocessing of box layout features to be used / shared by other components |
| [`box-transformer`][edspdf.layers.box_transformer.BoxTransformer]                        | Contextualize box embeddings with a 2d Transformer with relative position representations  |
| [`cnn-pooler`][edspdf.layers.cnn_pooler.CnnPooler]                                       | A pytorch component that aggregates its inputs by running convolution and max-pooling ops  |
| [`relative-attention`][edspdf.layers.relative_attention.RelativeAttention]               | A 2d attention layer that optionally uses relative position to compute its attention scores |
| [`sinusoidal-embedding`][edspdf.layers.sinusoidal_embedding.SinusoidalEmbedding]         | A position embedding that uses trigonometric functions to encode positions                 |
| [`vocabulary`][edspdf.layers.vocabulary.Vocabulary]                                      | A non deep learning layer to encodes / decode vocabularies                                 |
