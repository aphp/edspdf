import typing
from abc import ABCMeta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Union

import torch

from edspdf.pipeline import Pipeline
from edspdf.structures import PDFDoc
from edspdf.utils.collections import batch_compress_dict, decompress_dict

NestedSequences = Dict[str, Union["NestedSequences", Sequence]]
NestedTensors = Dict[str, Union["NestedSequences", torch.Tensor]]
InputBatch = typing.TypeVar("InputBatch", bound=NestedTensors)
OutputBatch = typing.TypeVar("OutputBatch", bound=NestedTensors)
Scorer = Callable[[Sequence[Tuple[PDFDoc, PDFDoc]]], Union[float, Dict[str, Any]]]


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


def hash_batch(batch):
    if isinstance(batch, list):
        return hash(tuple(id(item) for item in batch))
    elif not isinstance(batch, dict):
        return id(batch)
    return hash((tuple(batch.keys()), tuple(map(hash_batch, batch.values()))))


def cached_preprocess(fn):
    @wraps(fn)
    def wrapped(self: "TrainablePipe", doc: PDFDoc):
        if self.pipeline is None or self.pipeline._cache is None:
            return fn(self, doc)
        cache_id = (id(self), "preprocess", id(doc))
        if cache_id in self.pipeline._cache:
            return self.pipeline._cache[cache_id]
        res = fn(self, doc)
        self.pipeline._cache[cache_id] = res
        return res

    return wrapped


def cached_preprocess_supervised(fn):
    @wraps(fn)
    def wrapped(self: "TrainablePipe", doc: PDFDoc):
        if self.pipeline is None or self.pipeline._cache is None:
            return fn(self, doc)
        cache_id = (id(self), "preprocess_supervised", id(doc))
        if cache_id in self.pipeline._cache:
            return self.pipeline._cache[cache_id]
        res = fn(self, doc)
        self.pipeline._cache[cache_id] = res
        return res

    return wrapped


def cached_collate(fn):
    import torch

    @wraps(fn)
    def wrapped(self: "TrainablePipe", batch: Dict, device: torch.device):
        if self.pipeline is None or self.pipeline._cache is None:
            return fn(self, batch, device)
        cache_id = (id(self), "collate", hash_batch(batch))
        if cache_id in self.pipeline._cache:
            return self.pipeline._cache[cache_id]
        res = fn(self, batch, device)
        self.pipeline._cache[cache_id] = res
        res["cache_id"] = cache_id
        return res

    return wrapped


def cached_forward(fn):
    @wraps(fn)
    def wrapped(self: "TrainablePipe", batch):
        if self.pipeline is None or self.pipeline._cache is None:
            return fn(self, batch)
        cache_id = (id(self), "collate", hash_batch(batch))
        if cache_id in self.pipeline._cache:
            return self.pipeline._cache[cache_id]
        res = fn(self, batch)
        self.pipeline._cache[cache_id] = res
        return res

    return wrapped


class TrainablePipeMeta(ABCMeta):
    def __new__(mcs, name, bases, class_dict):
        if "preprocess" in class_dict:
            class_dict["preprocess"] = cached_preprocess(class_dict["preprocess"])
        if "preprocess_supervised" in class_dict:
            class_dict["preprocess_supervised"] = cached_preprocess_supervised(
                class_dict["preprocess_supervised"]
            )
        if "collate" in class_dict:
            class_dict["collate"] = cached_collate(class_dict["collate"])
        if "forward" in class_dict:
            class_dict["forward"] = cached_forward(class_dict["forward"])

        return super().__new__(mcs, name, bases, class_dict)


class TrainablePipe(
    torch.nn.Module,
    typing.Generic[OutputBatch],
    metaclass=TrainablePipeMeta,
):
    """
    A TrainablePipe is a Component that can be trained and inherits `torch.nn.Module`.
    You can use it either as a torch module inside a more complex neural network, or as
    a standalone component in a [Pipeline][edspdf.pipeline.Pipeline].

    In addition to the methods of a torch module, a TrainablePipe adds a few methods to
    handle preprocessing and collating features, as well as caching intermediate results
    for components that share a common subcomponent.
    """

    def __init__(self, pipeline: Pipeline, name: str):
        super().__init__()
        self.pipeline = pipeline
        self.name = name
        self.cfg = {}
        self._preprocess_cache = {}
        self._preprocess_supervised_cache = {}
        self._collate_cache = {}
        self._forward_cache = {}

    @property
    def device(self):
        return next((p.device for p in self.parameters()), torch.device("cpu"))

    def named_component_children(self):
        for name, module in self.named_children():
            if isinstance(module, TrainablePipe):
                yield name, module

    def save_extra_data(self, path: Path, exclude: set):
        """
        Dumps vocabularies indices to json files

        Parameters
        ----------
        path: Path
            Path to the directory where the files will be saved
        exclude: Set
            The set of component names to exclude from saving
            This is useful when components are repeated in the pipeline.
        """
        if self.name in exclude:
            return
        exclude.add(self.name)
        for name, component in self.named_component_children():
            if hasattr(component, "save_extra_data"):
                component.save_extra_data(path / name, exclude)

    def load_extra_data(self, path: Path, exclude: set):
        """
        Loads vocabularies indices from json files

        Parameters
        ----------
        path: Path
            Path to the directory where the files will be loaded
        exclude: Set
            The set of component names to exclude from loading
            This is useful when components are repeated in the pipeline.
        """
        if self.name in exclude:
            return
        exclude.add(self.name)
        for name, component in self.named_component_children():
            if hasattr(component, "load_extra_data"):
                component.load_extra_data(path / name, exclude)

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: set):
        """
        This method completes the attributes of the component, by looking at some
        documents. It is especially useful to build vocabularies or detect the labels
        of a classification task.

        Parameters
        ----------
        gold_data: Iterable[PDFDoc]
            The documents to use for initialization.
        exclude: Optional[set]
            The names of components to exclude from initialization.
            This argument will be gradually updated  with the names of initialized
            components
        """
        if self.name in exclude:
            return
        exclude.add(self.name)
        for name, component in self.named_component_children():
            if hasattr(component, "post_init"):
                component.post_init(gold_data, exclude=exclude)

    def preprocess(self, doc: PDFDoc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its predictions.

        Parameters
        ----------
        doc: PDFDoc
            PDFDocument to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return {
            name: component.preprocess(doc)
            for name, component in self.named_component_children()
        }

    def collate(self, batch: NestedSequences, device: torch.device) -> InputBatch:
        """
        Collate the batch of features into a single batch of tensors that can be
        used by the forward method of the component.

        Parameters
        ----------
        batch: NestedSequences
            Batch of features
        device: torch.device
            Device on which the tensors should be moved

        Returns
        -------
        InputBatch
            Dictionary (optionally nested) containing the collated tensors
        """
        return {
            name: component.collate(batch[name], device)
            for name, component in self.named_component_children()
        }

    def forward(self, batch: InputBatch) -> OutputBatch:
        """
        Perform the forward pass of the neural network, i.e, apply transformations
        over the collated features to compute new embeddings, probabilities, losses, etc

        Parameters
        ----------
        batch: InputBatch
            Batch of tensors (nested dictionary) computed by the collate method

        Returns
        -------
        OutputBatch
        """
        raise NotImplementedError()

    def module_forward(self, batch: InputBatch) -> OutputBatch:
        """
        This is a wrapper around `torch.nn.Module.__call__` to avoid conflict
        with the
        [`TrainablePipe.__call__`][edspdf.trainable_pipe.TrainablePipe.__call__]
        method.
        """
        return torch.nn.Module.__call__(self, batch)

    def make_batch(
        self,
        docs: Sequence[PDFDoc],
        supervision: bool = False,
    ) -> Dict[str, Sequence[Any]]:
        """
        Convenience method to preprocess a batch of documents and collate them
        Features corresponding to the same path are grouped together in a list,
        under the same key.

        Parameters
        ----------
        docs: Sequence[PDFDoc]
            Batch of documents
        supervision: bool
            Whether to extract supervision features or not

        Returns
        -------
        Dict[str, Sequence[Any]]
        """
        batch = [
            (self.preprocess_supervised(doc) if supervision else self.preprocess(doc))
            for doc in docs
        ]
        return decompress_dict(list(batch_compress_dict(batch)))

    def batch_process(self, docs: Sequence[PDFDoc]) -> Sequence[PDFDoc]:
        """
        Process a batch of documents using the neural network.
        This differs from the `pipe` method in that it does not return an
        iterator, but executes the component on the whole batch at once.

        Parameters
        ----------
        docs: Sequence[PDFDoc]
            Batch of documents

        Returns
        -------
        Sequence[PDFDoc]
            Batch of updated documents
        """
        with torch.no_grad():
            batch = self.make_batch(docs)
            inputs = self.collate(batch, device=self.device)
            if hasattr(self, "compiled"):
                res = self.compiled(inputs)
            else:
                res = self.module_forward(inputs)
            docs = self.postprocess(docs, res)
            return docs

    def postprocess(
        self, docs: Sequence[PDFDoc], batch: OutputBatch
    ) -> Sequence[PDFDoc]:
        """
        Update the documents with the predictions of the neural network, for instance
        converting label probabilities into label attributes on the document lines.

        By default, this is a no-op.

        Parameters
        ----------
        docs: Sequence[PDFDoc]
            Batch of documents
        batch: OutputBatch
            Batch of predictions, as returned by the forward method

        Returns
        -------
        Sequence[PDFDoc]
        """
        return docs

    # Same as preprocess but with gold supervision data
    def preprocess_supervised(self, doc: PDFDoc) -> Dict[str, Any]:
        """
        Preprocess the document to extract features that will be used by the
        neural network to perform its training.
        By default, this returns the same features as the `preprocess` method.

        Parameters
        ----------
        doc: PDFDoc
            PDFDocument to preprocess

        Returns
        -------
        Dict[str, Any]
            Dictionary (optionally nested) containing the features extracted from
            the document.
        """
        return self.preprocess(doc)

    def clean_gold_for_evaluation(self, gold: PDFDoc) -> PDFDoc:
        """
        Clean the gold document before evaluation.
        Only the attributes that are predicted by the component should be removed.
        By default, this is a no-op.

        Parameters
        ----------
        gold: PDFDoc
            Gold document

        Returns
        -------
        PDFDoc
            The document without attributes that should be predicted
        """
        return gold

    def __call__(self, doc: PDFDoc) -> PDFDoc:
        """
        Applies the component on a single doc.
        For multiple documents, prefer batch processing via the
        [batch_process][edspdf.trainable_pipe.TrainablePipe.batch_process] method.
        In general, prefer the [Pipeline][edspdf.pipeline.Pipeline] methods

        Parameters
        ----------
        doc: PDFDoc

        Returns
        -------
        PDFDoc
        """
        return self.batch_process([doc])[0]
