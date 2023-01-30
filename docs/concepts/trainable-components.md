
# Trainable components

Trainable components allow for deep learning operations to be performed on the PDFDoc object and must be trained to be used.
Such components can be used to train a model to predict the label of the lines extracted from a PDF document.

## Anatomy of a trainable component

Building and running deep learning models usually requires preprocessing the input sample into features, batching or "collating" these features together to process multiple samples at once, running deep learning operations over these features (in Pytorch, this step is done in the `forward` method) and postprocessing the outputs of these operation to complete the original sample.

In the trainable components of EDS-PDF, preprocessing and postprocessing are decoupled from the deep learning code but collocated with the forward method. This is achieved by splitting the class of a trainable component into four methods:

- `preprocess`: converts a doc into features that will be consumed by the forward method, e.g. building arrays of features, encoding words into indices, etc
- `collate`: concatenates the preprocessed features of multiple documents into pytorch tensors
- `forward`: applies transformations over the collated features to compute new embeddings, probabilities, etc
- `postprocess`: use these predictions to annotate the document, for instance converting label probabilities into label attributes on the document lines

This code organization allows us to keep the development of new deep-learning components simple while ensuring efficient models both during training and inference.

Additionally, there is a fifth method `initialize` that is only called before training a new model, to complete the attributes of a component by looking at some of documents. It is especially useful to build vocabularies or detect the labels of a classification task.

Here is an example of a trainable component:

```python
from typing import Any, Dict, Iterable, Sequence

import torch
from tqdm import tqdm

from edspdf import Module, TrainableComponent, registry
from edspdf.models import PDFDoc


@registry.factory.register("my-component")
class MyComponent(TrainableComponent):
    def __init__(
        self,
        # A subcomponent
        embedding: Module,
    ):
        super().__init__()
        self.embedding: Module = embedding

    def initialize(self, gold_data: Iterable[PDFDoc]):
        # Initialize the component with the gold documents
        with self.label_vocabulary.initialization():
            for doc in tqdm(gold_data, desc="Initializing the component"):
                # Do something like learning a vocabulary over the initialization
                # documents
                ...

        # And initialize the subcomponent
        self.embedding.initialize(gold_data)

        # Initialize any layer that might be missing from the module
        self.classifier = torch.nn.Linear(...)

    def preprocess(self, doc: PDFDoc, supervision: bool = False) -> Dict[str, Any]:
        # Preprocess the doc to extract features required to run the embedding
        # subcomponent, and this component
        return {
            "embedding": self.embedding.preprocess(doc, supervision=supervision),
            "my-feature": ...(doc),
        }

    def collate(self, batch, device: torch.device) -> Dict:
        # Collate the features of the "embedding" subcomponent
        # and the features of this component as well
        return {
            "embedding": self.embedding.collate(batch["embedding"], device),
            "my-feature": torch.as_tensor(batch["my-feature"], device=device),
        }

    def forward(self, batch: Dict, supervision=False) -> Dict:
        # Call the embedding subcomponent
        embeds = self.embedding(batch["embedding"])

        # Do something with the embedding tensors
        output = ...(embeds)

        return output

    def postprocess(self, docs: Sequence[PDFDoc], output: Dict) -> Sequence[PDFDoc]:
        # Annotate the docs with the outputs of the forward method
        ...
        return docs
```

## Nesting trainable components

Like pytorch modules, you can compose trainable components together to build complex architectures. For instance, a deep classifier component may delegate some of its logic to an embedding component, which will only be responsible for converting PDF lines into multidimensional arrays of numbers.

Nesting components allows switching parts of the neural networks to test various architectures and keeping the modelling logic modular.

## Sharing subcomponents

Sharing parts of a neural network while training components on different tasks can be an effective way to improve the network efficiency. For instance, it is common to share an embedding layer between multiple tasks that require embedding the same inputs.

In EDS-PDF, sharing a subcomponent is simply done by sharing the object between the multiple components. You can either refer to an existing subcomponent when configuring a new component in Python, or use the interpolation mechanism of our configuration system.

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

    ```ini
    [components.first]
    @factory = "my-component-1"

    [components.first.embedding]
    @factory = "box-embedding"
    ...

    [components.second]
    @factory = "my-component-2"
    embedding = ${components.first.embedding}
    ```

To avoid recomputing the `preprocess` / `forward` and `collate` in the multiple components that use it, we rely on a transparent cache system.

!!! warning "Cache"

    During the training loop, the cache must be emptied at every step to release CPU and GPU memory occupied by the cached methods outputs. This is done by calling `reset_cache` method on the pipeline.
