# Training a Pipeline

In this chapter, we'll see how we can train a deep-learning based classifier to better classify the lines of the
document and extract texts from the document.

## Step-by-step walkthrough

Training supervised models consists in feeding batches of samples taken from a training corpus
to a model instantiated from a given architecture and optimizing the learnable weights of the
model to decrease a given loss. The process of training a pipeline with EDS-PDF is as follows:

1. We first start by seeding the random states and instantiating a new trainable pipeline. Here we show two examples of pipeline, the first one based on a custom embedding architecture and the second one based on a pre-trained HuggingFace transformer model.

    === "Custom architecture"

        The architecture of the trainable classifier of this recipe is described in the following figure:
        ![Architecture of the trainable classifier](resources/deep-learning-architecture.svg)

        ```{ .python .annotate }
        from edspdf import Pipeline
        from edspdf.utils.random import set_seed

        set_seed(42)

        model = Pipeline()
        model.add_pipe("pdfminer-extractor", name="extractor") # (1)
        model.add_pipe(
            "box-transformer",
            name="embedding",
            config={
                "num_heads": 4,
                "dropout_p": 0.1,
                "activation": "gelu",
                "init_resweight": 0.01,
                "head_size": 16,
                "attention_mode": ["c2c", "c2p", "p2c"],
                "n_layers": 1,
                "n_relative_positions": 64,
                "embedding": {
                    "@factory": "embedding-combiner",
                    "dropout_p": 0.1,
                    "text_encoder": {
                        "@factory": "sub-box-cnn-pooler",
                        "out_channels": 64,
                        "kernel_sizes": (3, 4, 5),
                        "embedding": {
                            "@factory": "simple-text-embedding",
                            "size": 72,
                        },
                    },
                    "layout_encoder": {
                        "@factory": "box-layout-embedding",
                        "n_positions": 64,
                        "x_mode": "learned",
                        "y_mode": "learned",
                        "w_mode": "learned",
                        "h_mode": "learned",
                        "size": 72,
                    },
                },
            },
        )
        model.add_pipe(
            "trainable-classifier",
            name="classifier",
            config={
                "embedding": model.get_pipe("embedding"),
                "labels": [],
            },
        )
        ```

        1. You can choose between multiple extractors, such as "pdfminer-extractor", "mupdf-extractor" or "poppler-extractor" (the latter does not support rendering images). See the extractors list here [extractors](/pipes/extractors) for more details.

    === "Pre-trained HuggingFace transformer"

        ```{ .python .annotate }
        model = Pipeline()
        model.add_pipe(
            "mupdf-extractor",
            name="extractor",
            config={
                "render_pages": True,
            },
        ) # (1)
        model.add_pipe(
            "huggingface-embedding",
            name="embedding",
            config={
                "model": "microsoft/layoutlmv3-base",
                "use_image": False,
                "window": 128,
                "stride": 64,
                "line_pooling": "mean",
            },
        )
        model.add_pipe(
            "trainable-classifier",
            name="classifier",
            config={
                "embedding": model.get_pipe("embedding"),
                "labels": [],
            },
        )
        ```

        1. You can choose between multiple extractors, such as "pdfminer-extractor", "mupdf-extractor" or "poppler-extractor" (the latter does not support rendering images). See the extractors list here [extractors](/pipes/extractors) for more details.

2. We then load and adapt (i.e., convert into PDFDoc) the training and validation dataset, which is often a combination of JSON and PDF files. The recommended way of doing this is to make a Python generator of PDFDoc objects.
    ```python
    train_docs = list(segmentation_adapter(train_path)(model))
    val_docs = list(segmentation_adapter(val_path)(model))
    ```

3. We initialize the missing or incomplete components attributes (such as vocabularies) with the training dataset
    ```python
    model.post_init(train_docs)
    ```

4. The training dataset is then preprocessed into features. The resulting preprocessed dataset is then wrapped into a pytorch DataLoader to be fed to the model during the training loop with the model's own collate method.
    ```python
    preprocessed = list(model.preprocess_many(train_docs, supervision=True))
    dataloader = DataLoader(
        preprocessed,
        batch_size=batch_size,
        collate_fn=model.collate,
        shuffle=True,
    )
    ```

5. We instantiate an optimizer and start the training loop
    ```python
    from itertools import chain, repeat

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
    )

    # We will loop over the dataloader
    iterator = chain.from_iterable(repeat(dataloader))

    for step in tqdm(range(max_steps), "Training model", leave=True):
        batch = next(iterator)
        optimizer.zero_grad()
    ```


6. The trainable components are fed the collated batches from the dataloader with the [`TrainablePipe.module_forward`][edspdf.trainable_pipe.TrainablePipe.module_forward] methods to compute the losses. Since outputs of shared subcomponents are reused between components, we enable caching by wrapping this step in a cache context. The training loop is otherwise carried in a similar fashion to a standard pytorch training loop
    ```python
    with model.cache():
        loss = torch.zeros((), device="cpu")
        for name, component in model.trainable_pipes():
            output = component.module_forward(batch[component.name])
            if "loss" in output:
                loss += output["loss"]

        loss.backward()

        optimizer.step()
    ```

   7. Finally, the model is evaluated on the validation dataset at regular intervals and saved at the end of the training. To score the model, we only want to run "classifier" component and not the extractor, otherwise we would overwrite annotated text boxes on documents in the `val_docs` dataset, and have mismatching text boxes between the gold and predicted documents. To save the model, although you can use `torch.save` to save your model, we provide a safer method to avoid the security pitfalls of pickle models
       ```python
       from edspdf import Pipeline
       from sklearn.metrics import classification_report
       from copy import deepcopy


       def score(golds, preds):
           return classification_report(
               [b.label for gold in golds for b in gold.text_boxes if b.text != ""],
               [b.label for pred in preds for b in pred.text_boxes if b.text != ""],
               output_dict=True,
               zero_division=0,
           )


       ...

       if (step % 100) == 0:
           # we only want to run "classifier" component, not overwrite the text boxes
           with model.select_pipes(enable=["classifier"]):
               print(score(val_docs, model.pipe(deepcopy(val_docs))))

       # torch.save(model, "model.pt")
       model.save("model")
       ```

## Adapting a dataset

The first step of training a pipeline is to adapt the dataset to the pipeline. This is done by converting the dataset into a list of [PDFDoc][edspdf.structures.PDFDoc] objects, using an [extractor](/components/extractors). The following function loads a dataset of `.pdf` and `.json` files, where each `.json` file contain box annotations represented with `page`, `x0`, `x1`, `y0`, `y1` and `label`.

```python
from edspdf.utils.alignment import align_box_labels
from pathlib import Path
from pydantic import DirectoryPath
from edspdf.registry import registry
from edspdf.structures import Box
import json


@registry.adapter.register("my-segmentation-adapter")
def segmentation_adapter(
    path: DirectoryPath,
):
    def adapt_to(model):
        for anns_filepath in sorted(Path(path).glob("*.json")):
            pdf_filepath = str(anns_filepath).replace(".json", ".pdf")
            with open(anns_filepath) as f:
                sample = json.load(f)
            pdf = Path(pdf_filepath).read_bytes()

            if len(sample["annotations"]) == 0:
                continue

            doc = model.components.extractor(pdf)
            doc.id = pdf_filepath.split(".")[0].split("/")[-1]
            doc.lines = [
                line
                for page in sorted(set(b.page for b in doc.lines))
                for line in align_box_labels(
                    src_boxes=[
                        Box(
                            page_num=b["page"],
                            x0=b["x0"],
                            x1=b["x1"],
                            y0=b["y0"],
                            y1=b["y1"],
                            label=b["label"],
                        )
                        for b in sample["annotations"]
                        if b["page"] == page
                    ],
                    dst_boxes=doc.lines,
                    pollution_label=None,
                )
                if line.text == "" or line.label is not None
            ]
            yield doc

    return adapt_to
```

## Full example

Let's wrap the training code in a function, and make it callable from the command line using [confit](https://github.com/aphp/confit) !

???+ example "train.py"
    ```python linenums="1"
    import itertools
    import json
    from copy import deepcopy
    from pathlib import Path

    import torch
    from confit import Cli
    from pydantic import DirectoryPath
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from edspdf import Pipeline, registry
    from edspdf.structures import Box
    from edspdf.utils.alignment import align_box_labels
    from edspdf.utils.random import set_seed

    app = Cli(pretty_exceptions_show_locals=False)


    def score(golds, preds):
        return classification_report(
            [b.label for gold in golds for b in gold.text_boxes if b.text != ""],
            [b.label for pred in preds for b in pred.text_boxes if b.text != ""],
            output_dict=True,
            zero_division=0,
        )


    @registry.adapter.register("my-segmentation-adapter")
    def segmentation_adapter(
        path: str,
    ):
        def adapt_to(model):
            for anns_filepath in sorted(Path(path).glob("*.json")):
                pdf_filepath = str(anns_filepath).replace(".json", ".pdf")
                with open(anns_filepath) as f:
                    sample = json.load(f)
                pdf = Path(pdf_filepath).read_bytes()

                if len(sample["annotations"]) == 0:
                    continue

                doc = model.get_pipe("extractor")(pdf)
                doc.id = pdf_filepath.split(".")[0].split("/")[-1]
                doc.content_boxes = [
                    line
                    for page_num in sorted(set(b.page_num for b in doc.lines))
                    for line in align_box_labels(
                        src_boxes=[
                            Box(
                                page_num=b["page"],
                                x0=b["x0"],
                                x1=b["x1"],
                                y0=b["y0"],
                                y1=b["y1"],
                                label=b["label"],
                            )
                            for b in sample["annotations"]
                            if b["page"] == page_num
                        ],
                        dst_boxes=doc.lines,
                        pollution_label=None,
                    )
                    if line.text == "" or line.label is not None
                ]
                yield doc

        return adapt_to


    @app.command(name="train")
    def train_my_model(
        train_path: DirectoryPath = "dataset/train",
        val_path: DirectoryPath = "dataset/dev",
        max_steps: int = 1000,
        batch_size: int = 4,
        lr: float = 3e-4,
    ):
        set_seed(42)

        # We define the model
        model = Pipeline()
        model.add_pipe("mupdf-extractor", name="extractor")
        model.add_pipe(
            "box-transformer",
            name="embedding",
            config={
                "num_heads": 4,
                "dropout_p": 0.1,
                "activation": "gelu",
                "init_resweight": 0.01,
                "head_size": 16,
                "attention_mode": ["c2c", "c2p", "p2c"],
                "n_layers": 1,
                "n_relative_positions": 64,
                "embedding": {
                    "@factory": "embedding-combiner",
                    "dropout_p": 0.1,
                    "text_encoder": {
                        "@factory": "sub-box-cnn-pooler",
                        "out_channels": 64,
                        "kernel_sizes": (3, 4, 5),
                        "embedding": {
                            "@factory": "simple-text-embedding",
                            "size": 72,
                        },
                    },
                    "layout_encoder": {
                        "@factory": "box-layout-embedding",
                        "n_positions": 64,
                        "x_mode": "learned",
                        "y_mode": "learned",
                        "w_mode": "learned",
                        "h_mode": "learned",
                        "size": 72,
                    },
                },
            },
        )
        model.add_pipe(
            "trainable-classifier",
            name="classifier",
            config={
                "embedding": model.get_pipe("embedding"),
                "labels": [],
            },
        )

        # Loading and adapting the training and validation data
        train_docs = list(segmentation_adapter(train_path)(model))
        val_docs = list(segmentation_adapter(val_path)(model))

        # Taking the first `initialization_subset` samples to initialize the model
        model.post_init(train_docs)

        # Preprocessing the training dataset into a dataloader
        preprocessed = list(model.preprocess_many(train_docs, supervision=True))
        dataloader = DataLoader(
            preprocessed,
            batch_size=batch_size,
            collate_fn=model.collate,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
        )

        # We will loop over the dataloader
        iterator = itertools.chain.from_iterable(itertools.repeat(dataloader))

        for step in tqdm(range(max_steps), "Training model", leave=True):
            batch = next(iterator)
            optimizer.zero_grad()

            with model.cache():
                loss = torch.zeros((), device="cpu")
                for name, component in model.trainable_pipes():
                    output = component.module_forward(batch[component.name])
                    if "loss" in output:
                        loss += output["loss"]

                loss.backward()

                optimizer.step()

            if (step % 100) == 0:
                with model.select_pipes(enable=["classifier"]):
                    print(score(val_docs, model.pipe(deepcopy(val_docs))))
                model.save("model")

        return model


    if __name__ == "__main__":
        app()

    ```

```bash
python train.py --seed 42
```

At the end of the training, the pipeline is ready to use (with the `.pipe` method) since every trained component of the pipeline is self-sufficient, ie contains the preprocessing, inference and postprocessing code required to run it.

## Configuration

To decouple the configuration and the code of our training script, let's define a configuration file where we will describe **both** our training parameters and the pipeline. You can either write the config of the pipeline by hand, or generate it from an instantiated pipeline by running:

```python
print(pipeline.config.to_str())
```

=== "Custom architecture"

    ```toml title="config.cfg"
    # This is this equivalent of the API-based declaration at the beginning of the tutorial
    [pipeline]
    pipeline = ["extractor", "embedding", "classifier"]
    disabled = []
    components = ${components}

    [components]

    [components.extractor]
    @factory = "pdfminer-extractor"

    [components.embedding]
    @factory = "box-transformer"
    num_heads = 4
    dropout_p = 0.1
    activation = "gelu"
    init_resweight = 0.01
    head_size = 16
    attention_mode = ["c2c", "c2p", "p2c"]
    n_layers = 1
    n_relative_positions = 64

    [components.embedding.embedding]
    @factory = "embedding-combiner"
    dropout_p = 0.1

    [components.embedding.embedding.text_encoder]
    @factory = "sub-box-cnn-pooler"
    out_channels = 64
    kernel_sizes = (3, 4, 5)

    [components.embedding.embedding.text_encoder.embedding]
    @factory = "simple-text-embedding"
    size = 72

    [components.embedding.embedding.layout_encoder]
    @factory = "box-layout-embedding"
    n_positions = 64
    x_mode = "learned"
    y_mode = "learned"
    w_mode = "learned"
    h_mode = "learned"
    size = 72

    [components.classifier]
    @factory = "trainable-classifier"
    embedding = ${components.embedding}
    labels = []

    # This is were we define the training script parameters
    # the "train" section refers to the name of the command in the training script
    [train]
    model = ${pipeline}
    train_data = {"@adapter": "my-segmentation-adapter", "path": "data/train"}
    val_data = {"@adapter": "my-segmentation-adapter", "path": "data/val"}
    max_steps = 1000
    seed = 42
    lr = 3e-4
    batch_size = 4
    ```

=== "Pretrained Huggingface Transformer"

    ```toml title="config.cfg"
    [pipeline]
    pipeline = ["extractor", "embedding", "classifier"]
    disabled = []
    components = ${components}

    [components]

    [components.extractor]
    @factory = "mupdf-extractor"
    render_pages = true

    [components.embedding]
    @factory = "huggingface-embedding"
    model = "microsoft/layoutlmv3-base"
    use_image = false
    window = 128
    stride = 64
    line_pooling = "mean"

    [components.classifier]
    @factory = "trainable-classifier"
    embedding = ${components.embedding}
    labels = []

    [train]
    model = ${pipeline}
    max_steps = 1000
    lr = 5e-5
    seed = 42
    train_data = {"@adapter": "my-segmentation-adapter", "path": "data/train"}
    val_data = {"@adapter": "my-segmentation-adapter", "path": "data/val"}
    batch_size = 8
    ```

and update our training script to use the pipeline and the data adapters defined in the configuration file instead of the Python declaration :

```diff
@app.command(name="train")
def train_my_model(
+   model: Pipeline,
+   train_path: DirectoryPath = "data/train",
-   train_data: Callable = segmentation_adapter("data/train"),
+   val_path: DirectoryPath = "data/val",
-   val_data: Callable = segmentation_adapter("data/val"),
    seed: int = 42,
    max_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 3e-4,
):
    # Seed will be set by the CLI util, before `model` is instanciated
-   set_seed(seed)

    # Model will be defined from the config file using registries
-   model = Pipeline()
-   model.add_pipe("mupdf-extractor", name="extractor")
-   model.add_pipe(
-       "box-transformer",
-       name="embedding",
-       config={
-           "num_heads": 4,
-           "dropout_p": 0.1,
-           "activation": "gelu",
-           "init_resweight": 0.01,
-           "head_size": 16,
-           "attention_mode": ["c2c", "c2p", "p2c"],
-           "n_layers": 1,
-           "n_relative_positions": 64,
-           "embedding": {
-               "@factory": "embedding-combiner",
-               "dropout_p": 0.1,
-               "text_encoder": {
-                   "@factory": "sub-box-cnn-pooler",
-                   "out_channels": 64,
-                   "kernel_sizes": (3, 4, 5),
-                   "embedding": {
-                       "@factory": "simple-text-embedding",
-                       "size": 72,
-                   },
-               },
-               "layout_encoder": {
-                   "@factory": "box-layout-embedding",
-                   "n_positions": 64,
-                   "x_mode": "learned",
-                   "y_mode": "learned",
-                   "w_mode": "learned",
-                   "h_mode": "learned",
-                   "size": 72,
-               },
-           },
-       },
-   )
-   model.add_pipe(
-       "trainable-classifier",
-       name="classifier",
-       config={
-           "embedding": model.get_pipe("embedding"),
-           "labels": [],
-       },
-   )

    # Loading and adapting the training and validation data
-    train_docs = list(segmentation_adapter(train_path)(model))
+    train_docs = list(train_data(model))
-    val_docs = list(segmentation_adapter(val_path)(model))
+    val_docs = list(val_data(model))

    # Taking the first `initialization_subset` samples to initialize the model
    ...
```

That's it ! We can now call the training script with the configuration file as a parameter, and override some of its defaults values:

```bash
python train.py --config config.cfg --components.extractor.extract_styles=true --seed 43
```
