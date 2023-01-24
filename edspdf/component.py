import contextlib
import inspect
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch

from .utils.collections import batch_compress_dict, batchify, decompress_dict

BatchT = TypeVar("BatchT")
InT = TypeVar("InT")
OutT = TypeVar("OutT")
Scorer = Callable[[Sequence[Tuple[OutT, OutT]]], Union[float, Dict[str, Any]]]


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


def base_scorer(pairs: Sequence[Tuple[OutT, OutT]]) -> Dict[str, Any]:
    return {}


def hash_batch(batch):
    if isinstance(batch, list):
        return hash(tuple(id(item) for item in batch))
    elif not isinstance(batch, dict):
        return id(batch)
    return hash((tuple(batch.keys()), tuple(map(hash_batch, batch.values()))))


def cached_preprocess(fn):
    @wraps(fn)
    def wrapped(self: "Module", doc: InT, supervision: bool = False):
        if not self._do_cache:
            return fn(self, doc, supervision=supervision)
        cache_id = hash((id(doc), supervision))
        if cache_id in self._preprocess_cache:
            return self._preprocess_cache[cache_id]
        res = fn(self, doc, supervision=supervision)
        self._preprocess_cache[cache_id] = res
        return res

    return wrapped


def cached_collate(fn):
    @wraps(fn)
    def wrapped(self: "Module", batch: Dict, device: torch.device):
        cache_id = hash_batch(batch)
        if not self._do_cache or cache_id is None:
            return fn(self, batch, device)
        if cache_id in self._collate_cache:
            return self._collate_cache[cache_id]
        res = fn(self, batch, device)
        self._collate_cache[cache_id] = res
        res["cache_id"] = cache_id
        return res

    return wrapped


def cached_forward(fn):
    @wraps(fn)
    def wrapped(self: "Module", *args, **kwargs):
        # Convert args and kwargs to a dictionary matching fn signature
        call_kwargs = inspect.getcallargs(fn, self, *args, **kwargs)
        call_kwargs.pop("self")
        cache_id = hash_batch(call_kwargs)
        if not self._do_cache or cache_id is None:
            return fn(self, *args, **kwargs)
        if cache_id in self._forward_cache:
            return self._forward_cache[cache_id]
        res = fn(self, *args, **kwargs)
        self._forward_cache[cache_id] = res
        return res

    return wrapped


class Component(Generic[InT, OutT]):
    def __init__(
        self,
        scorer: Optional[Scorer[OutT]] = None,
    ):
        self.name: Optional[str] = None
        self.scorer = scorer
        self.needs_training = False
        self.initialized = False

    def reset_cache(self, cache: Optional[CacheEnum] = None):
        pass

    def initialize(self, gold_data: Iterable[OutT]):
        """
        Initialize the missing properties of the component, such as its vocabulary,
        using the gold data.

        Parameters
        ----------
        gold_data: Iterable[OutT]
            Gold data to use for initialization
        """

    @abstractmethod
    def __call__(self, doc: InT) -> OutT:
        """
        Processes a single document

        Parameters
        ----------
        doc: InT
            Document to process

        Returns
        -------
        OutT
            Processed document
        """

    def batch_process(self, docs: Sequence[InT], refs=None) -> Sequence[OutT]:
        return [self(doc) for doc in docs]

    def score(self, pairs):
        if self.scorer is None:
            return {}
        return self.scorer(pairs)

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def pipe(self, docs: Iterable[InT], batch_size=1) -> Iterable[OutT]:
        """
        Applies the component on a collection of documents. It is recommended to use
        the [`Pipeline.pipe`][edspdf.pipeline.Pipeline.pipe] method instead of this one
        to apply a pipeline on a collection of documents, to benefit from the caching
        of intermediate results and avoiding loading too many documents in memory at
        once.

        Parameters
        ----------
        docs: Iterable[InT]
            Input docs
        batch_size: int
            Batch size to use when making batched to be process at once
        """
        for batch in batchify(docs, batch_size=batch_size):
            self.reset_cache()
            yield from self.batch_process(batch)


class ModuleMeta(ABCMeta):
    def __new__(meta, name, bases, class_dict):
        if "preprocess" in class_dict:
            class_dict["preprocess"] = cached_preprocess(class_dict["preprocess"])
        if "collate" in class_dict:
            class_dict["collate"] = cached_collate(class_dict["collate"])
        if "forward" in class_dict:
            class_dict["forward"] = cached_forward(class_dict["forward"])

        return super().__new__(meta, name, bases, class_dict)


class Module(torch.nn.Module, Generic[InT, BatchT], metaclass=ModuleMeta):
    """
    Base class for all EDS-PDF modules. This class is an extension of Pytorch's
    `torch.nn.Module` class. It adds a few methods to handle preprocessing and collating
    features, as well as caching intermediate results for components that share a common
    subcomponent.
    """

    IS_MODULE = True

    def __init__(self):
        super().__init__()
        self._preprocess_cache = {}
        self._collate_cache = {}
        self._forward_cache = {}
        self._do_cache = True

    @contextlib.contextmanager
    def no_cache(self):
        saved = self.enable_cache(False)
        yield
        self.enable_cache(saved)

    def initialize(self, gold_data: Iterable[InT], **kwargs):
        """
        Initialize the missing properties of the module, such as its vocabulary,
        using the gold data and the provided keyword arguments.

        Parameters
        ----------
        gold_data: Iterable[InT]
            Gold data to use for initialization
        kwargs: Any
            Additional keyword arguments to use for initialization
        """
        for name, value in kwargs.items():
            if value is None:
                continue
            current_value = getattr(self, name)
            if current_value is not None and current_value != value:
                raise ValueError(
                    "Cannot initialize module with different values for "
                    "attribute '{}': {} != {}".format(name, current_value, value)
                )
            setattr(self, name, value)

    def enable_cache(self, do_cache):
        saved = self._do_cache
        self._do_cache = do_cache

        for module in self.modules():
            if isinstance(module, Module) and module is not self:
                module.enable_cache(do_cache)

        return saved

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def reset_cache(self, cache: Optional[CacheEnum] = None):
        def clear(module):
            try:
                assert cache is None or cache in (
                    CacheEnum.preprocess,
                    CacheEnum.collate,
                    CacheEnum.forward,
                )
                if cache is None or cache == CacheEnum.preprocess:
                    module._preprocess_cache.clear()
                if cache is None or cache == CacheEnum.collate:
                    module._collate_cache.clear()
                if cache is None or cache == CacheEnum.forward:
                    module._forward_cache.clear()
            except AttributeError:
                pass

        self.apply(clear)

    def preprocess(self, doc: InT, supervision: bool = False) -> Dict[str, Any]:
        """
        Parameters
        ----------
        doc: InT
        supervision: bool

        Returns
        -------
        Dict[str, Any]
        """
        return {}

    def collate(self, batch: Dict[str, Sequence[Any]], device: torch.device) -> BatchT:
        """Collate operation : should return some tensors"""
        return {}

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """Forward pass of the torch module"""
        raise NotImplementedError()

    def module_forward(self, *args, **kwargs):
        """
        This is a wrapper around `torch.nn.Module.__call__` to avoid conflict
        with the [`Component.__call__`][edspdf.component.Component.__call__]
         method.
        """
        return torch.nn.Module.__call__(self, *args, **kwargs)


class TrainableComponent(Module[InT, BatchT], Component[InT, OutT]):
    """
    A TrainableComponent is a Component that can be trained and inherits from both
    `Module` and `Component`. You can therefore use it either as a torch module inside
    a more complex neural network, or as a standalone component in a
    [Pipeline][edspdf.pipeline.Pipeline].
    """

    def __init__(self):
        Module.__init__(self)
        Component.__init__(self)
        self.needs_training = True

    def __call__(self, doc: InT) -> OutT:
        return next(iter(self.batch_process([doc])))

    def make_batch(self, docs: Sequence[InT], supervision: bool = False):
        batch = decompress_dict(
            list(
                batch_compress_dict(
                    [{self.name: self.preprocess(doc, supervision)} for doc in docs]
                )
            )
        )
        return batch

    def batch_process(self, docs: Sequence[InT], refs=None) -> Sequence[OutT]:
        with torch.no_grad():
            batch = self.make_batch(docs)

            inputs = self.collate(batch[self.name], device=self.device)
            res = self.forward(inputs)
            docs = self.postprocess(docs, res)
            return docs

    def forward(self, batch: BatchT, supervision=False) -> Dict[str, Any]:
        return batch

    def postprocess(self, docs: Sequence[InT], batch: Dict[str, Any]) -> Sequence[OutT]:
        return docs
