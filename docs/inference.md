# Inference

Once you have obtained a pipeline, either by composing rule-based components, training a model or loading a model from the disk, you can use it to make predictions on documents. This is referred to as inference.

## Inference on a single document

In EDS-PDF, computing the prediction on a single document is done by calling the pipeline on the document. The input can be either:

- a sequence of bytes
- or a [PDFDoc][edspdf.structures.PDFDoc] object

```python
from pathlib import Path

pipeline = ...
content = Path("path/to/.pdf").read_bytes()
doc = pipeline(content)
```

If you're lucky enough to have a GPU, you can use it to speed up inference by moving the model to the GPU before calling the pipeline. To leverage multiple GPUs, refer to the [multiprocessing accelerator][edspdf.accelerators.multiprocessing.MultiprocessingAccelerator] description below.

```python
pipeline.to("cuda")  # same semantics as pytorch
doc = pipeline(content)
```

## Inference on multiple documents

When processing multiple documents, it is usually more efficient to use the `pipeline.pipe(...)` method, especially when using deep learning components, since this allow matrix multiplications to be batched together. Depending on your computational resources and requirements, EDS-PDF comes with various "accelerators" to speed up inference (see the [Accelerators](#accelerators) section for more details). By default, the `.pipe()` method uses the [`simple` accelerator][edspdf.accelerators.simple.SimpleAccelerator] but you can switch to a different one by passing the `accelerator` argument.

```python
pipeline = ...
docs = pipeline.pipe(
    [content1, content2, ...],
    batch_size=16,  # optional, default to the one defined in the pipeline
    accelerator=my_accelerator,
)
```

The `pipe` method supports the following arguments :

::: edspdf.pipeline.Pipeline.pipe
    options:
        heading_level: 3
        only_parameters: true

## Accelerators

### Simple accelerator {: #edspdf.accelerators.simple.SimpleAccelerator }

::: edspdf.accelerators.simple.SimpleAccelerator
    options:
        heading_level: 3
        only_class_level: true

### Multiprocessing accelerator {: #edspdf.accelerators.multiprocessing.MultiprocessingAccelerator }

::: edspdf.accelerators.multiprocessing.MultiprocessingAccelerator
    options:
        heading_level: 3
        only_class_level: true
