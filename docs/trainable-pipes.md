# Trainable pipes {: #edspdf.trainable_pipe.TrainablePipe }

Trainable pipes allow for deep learning operations to be performed on the [PDFDoc][edspdf.structures.PDFDoc] object and must be trained to be used.
Such pipes can be used to train a model to predict the label of the lines extracted from a PDF document.

## Anatomy of a trainable pipe

Building and running deep learning models usually requires preprocessing the input sample into features, batching or "collating" these features together to process multiple samples at once, running deep learning operations over these features (in Pytorch, this step is done in the `forward` method) and postprocessing the outputs of these operation to complete the original sample.

In the trainable pipes of EDS-PDF, preprocessing and postprocessing are decoupled from the deep learning code but collocated with the forward method. This is achieved by splitting the class of a trainable component into four methods, which allows us to keep the development of new deep-learning components simple while ensuring efficient models both during training and inference.

### `preprocess` {: #edspdf.trainable_pipe.TrainablePipe.preprocess }

::: edspdf.trainable_pipe.TrainablePipe.preprocess
    options:
        heading_level: 3
        show_source: false

### `collate` {: #edspdf.trainable_pipe.TrainablePipe.collate }

::: edspdf.trainable_pipe.TrainablePipe.collate
    options:
        heading_level: 3
        show_source: false

### `forward` {: #edspdf.trainable_pipe.TrainablePipe.forward }

::: edspdf.trainable_pipe.TrainablePipe.forward
    options:
        heading_level: 3
        show_source: false

### `postprocess` {: #edspdf.trainable_pipe.TrainablePipe.postprocess }

::: edspdf.trainable_pipe.TrainablePipe.postprocess
    options:
        heading_level: 3
        show_source: false


Additionally, there is a fifth method:


### `post_init` {: #edspdf.trainable_pipe.TrainablePipe.post_init }

::: edspdf.trainable_pipe.TrainablePipe.post_init
    options:
        heading_level: 3
        show_source: false

## Implementing a trainable component

Here is an example of a trainable component:

```python
from typing import Any, Dict, Iterable, Sequence, List

import torch
from tqdm import tqdm

from edspdf import Pipeline, TrainablePipe, registry
from edspdf.structures import PDFDoc


@registry.factory.register("my-component")
class MyComponent(TrainablePipe):
    def __init__(
        self,
        # A subcomponent
        pipeline: Pipeline,
        name: str,
        embedding: TrainablePipe,
    ):
        super().__init__(pipeline=pipeline, name=name)
        self.embedding = embedding

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: set):
        # Initialize the component with the gold documents
        with self.label_vocabulary.initialization():
            for doc in tqdm(gold_data, desc="Initializing the component"):
                # Do something like learning a vocabulary over the initialization
                # documents
                ...

        # And post_init the subcomponent
        exclude.add(self.name)
        self.embedding.post_init(gold_data, exclude)

        # Initialize any layer that might be missing from the module
        self.classifier = torch.nn.Linear(...)

    def preprocess(self, doc: PDFDoc, supervision: bool = False) -> Dict[str, Any]:
        # Preprocess the doc to extract features required to run the embedding
        # subcomponent, and this component
        return {
            "embedding": self.embedding.preprocess_supervised(doc),
            "my-feature": ...(doc),
        }

    def collate(self, batch) -> Dict:
        # Collate the features of the "embedding" subcomponent
        # and the features of this component as well
        return {
            "embedding": self.embedding.collate(batch["embedding"]),
            "my-feature": torch.as_tensor(batch["my-feature"]),
        }

    def forward(self, batch: Dict, supervision=False) -> Dict:
        # Call the embedding subcomponent
        embeds = self.embedding(batch["embedding"])

        # Do something with the embedding tensors
        output = ...(embeds)

        return output

    def postprocess(
        self,
        docs: Sequence[PDFDoc],
        output: Dict,
        inputs: List[Dict[str, Any]],
    ) -> Sequence[PDFDoc]:
        # Annotate the docs with the outputs of the forward method
        ...
        return docs
```

## Nesting trainable pipes

Like pytorch modules, you can compose trainable pipes together to build complex architectures. For instance, a trainable classifier component may delegate some of its logic to an embedding component, which will only be responsible for converting PDF lines into multidimensional arrays of numbers.

Nesting pipes allows switching parts of the neural networks to test various architectures and keeping the modelling logic modular.

## Sharing subcomponents

Sharing parts of a neural network while training on different tasks can be an effective way to improve the network efficiency. For instance, it is common to share an embedding layer between multiple tasks that require embedding the same inputs.

In EDS-PDF, sharing a subcomponent is simply done by sharing the object between the multiple pipes. You can either refer to an existing subcomponent when configuring a new component in Python, or use the interpolation mechanism of our configuration system.

=== "API-based"

    ```python
    pipeline.add_pipe(
        "my-component-1",
        name="first",
        config={
            "embedding": {
                "@factory": "box-embedding",
                # ...
            }
        },
    )
    pipeline.add_pipe(
        "my-component-2",
        name="second",
        config={
            "embedding": pipeline.components.first.embedding,
        },
    )
    ```

=== "Configuration-based"

    ```toml
    [components.first]
    @factory = "my-component-1"

    [components.first.embedding]
    @factory = "box-embedding"
    ...

    [components.second]
    @factory = "my-component-2"
    embedding = ${components.first.embedding}
    ```

To avoid recomputing the `preprocess` / `forward` and `collate` in the multiple components that use it, we rely on a light cache system.

During the training loop, when computing the loss for each component, the forward calls must be wrapped by the [`pipeline.cache()`][edspdf.pipeline.Pipeline.cache] context to enable this caching mechanism between components.
