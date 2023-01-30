# Deep learning classifier

This component predicts the label of each line over the whole document using machine
learning.

!!! note

    You must train the model your model to use this classifier.
    See [Model training](../../recipes/training.md) for more information

## Configuration example

The classifier is composed of the following blocks:

- a configurable box embedding layer
- a linear classification layer

In this example, we use a `box-embedding` layer to generate the embeddings
of the boxes. It is composed of a text encoder that embeds the text features of the
boxes and a layout encoder that embeds the layout features of the boxes.
These two embeddings are summed and passed through an optional `contextualizer`, here
a `box-transformer`.

=== "API-based"

    ```python
    pipeline.add_pipe(
        "deep-classifier",
        name="classifier",
        config={
            "embedding": {
                "@factory": "box-embedding",
                "size": 72,
                "dropout_p": 0.1,
                "text_encoder": {
                    "@factory": "box-text-embedding",
                    "pooler": {
                        "@factory": "cnn-pooler",
                        "out_channels": 64,
                        "kernel_sizes": (3, 4, 5),
                    },
                },
                "layout_encoder": {
                    "@factory": "box-layout-embedding",
                    "n_positions": 64,
                    "x_mode": "learned",
                    "y_mode": "learned",
                    "w_mode": "learned",
                    "h_mode": "learned",
                },
                "contextualizer": {
                    "@factory": "box-transformer",
                    "n_relative_positions": 64,
                    "input_size": 72,
                    "num_heads": 4,
                    "dropout_p": 0.1,
                    "activation": "gelu",
                    "init_resweight": 0.01,
                    "head_size": 16,
                    "attention_mode": ("c2c", "c2p", "p2c"),
                    "n_layers": 1,
                },
            },
            "labels": [],
            "activation": "relu",
        },
    )
    ```

=== "Configuration-based"

    ```ini
    [components.classifier]
    @factory = "deep-classifier"
    labels = []
    activation = "relu"

    [components.classifier.embedding]
    @factory = "box-embedding"
    size = 72
    dropout_p = 0.1

    [components.classifier.embedding.layout_encoder]
    n_positions = 64
    x_mode = "learned"
    y_mode = "learned"
    w_mode = "learned"
    h_mode = "learned"

    [components.classifier.embedding.text_encoder]

    [components.classifier.embedding.text_encoder.pooler]
    out_channels = 64
    kernel_sizes = [3,4,5]

    [components.classifier.embedding.contextualizer]
    @factory = "box-transformer"
    n_relative_positions = 64
    num_heads = 4
    dropout_p = 0.1
    activation = "gelu"
    init_resweight = 0.01
    head_size = 16
    attention_mode = "c2c,c2p,p2c"
    n_layers = 1
    ```
