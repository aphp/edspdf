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

from .registry import registry
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
    def wrapped(self: "Module", batch: Dict, *args, **kwargs):
        cache_id = hash_batch(batch)
        if not self._do_cache or cache_id is None:
            return fn(self, batch, *args, **kwargs)
        if cache_id in self._forward_cache:
            return self._forward_cache[cache_id]
        res = fn(self, batch, *args, **kwargs)
        self._forward_cache[cache_id] = res
        return res

    return wrapped


def save_init_kwargs(fn: Callable) -> Callable:
    @wraps(fn)
    def new_init(self, *args, **kwargs):
        sig = inspect.signature(fn)
        bound_args = dict(sig.bind(self, *args, **kwargs).arguments)
        bound_args.pop("self")
        if getattr(self, "_cfg", None) is None:
            self._cfg = bound_args
        return fn(self, *args, **kwargs)

    return new_init


class ComponentMeta(ABCMeta):
    def __new__(mcs, name, bases, class_dict):
        if "__init__" in class_dict:
            wrapped = save_init_kwargs(class_dict["__init__"])
            class_dict["__init__"] = wrapped
            # wrapped.vd.model.__name__ = name
        return super().__new__(mcs, name, bases, class_dict)

    @property
    def factory_name(cls):
        return next(k for k, v in registry.factory.get_all().items() if cls == v)


class Component(Generic[InT, OutT], metaclass=ComponentMeta):
    def __init__(
        self,
        scorer: Optional[Scorer[OutT]] = None,
    ):
        self.name: Optional[str] = None
        self.scorer = scorer
        self.needs_training = False
        self.initialized = False

    #    @classmethod
    #    def __get_validators__(cls):
    #        # one or more validators may be yielded which will be called in the
    #        # order to validate the input, each validator will receive as an input
    #        # the value returned from the previous validator
    #        yield cls.validate
    #
    #    @classmethod
    #    def validate(cls, v):
    #        if isinstance(v, dict):
    #            return cls(**v)
    #        if isinstance(v, cls):
    #            return v
    #        raise ValueError(f"Could not cast {cls.__name__} from {v}")

    @property
    def factory_name(self):
        return self.__class__.factory_name

    @property
    def cfg(self):
        return {"@factory": self.factory_name, **self._cfg}  # noqa

    def reset_cache(self, cache=None):
        pass

    def initialize(self, gold_data: Iterable[OutT]):
        pass

    @abstractmethod
    def __call__(self, doc: InT) -> OutT:
        """Transformers the document"""

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
        Apply the pipeline on a collection of documents

        Parameters
        ----------
        docs: Iterable[InT]
            Input docs
        batch_size: int
            Batch size to use when making batched to be process at once

        Returns
        -------
        """
        for batch in batchify(docs, batch_size=batch_size):
            self.reset_cache()
            yield from self.batch_process(batch)


class ModuleMeta(ABCMeta):
    def __new__(meta, name, bases, class_dict):
        if "__init__" in class_dict:
            wrapped = save_init_kwargs(class_dict["__init__"])
            class_dict["__init__"] = wrapped
        if "preprocess" in class_dict:
            class_dict["preprocess"] = cached_preprocess(class_dict["preprocess"])
        if "collate" in class_dict:
            class_dict["collate"] = cached_collate(class_dict["collate"])
        if "forward" in class_dict:
            class_dict["forward"] = cached_forward(class_dict["forward"])

        return super().__new__(meta, name, bases, class_dict)

    @property
    def factory_name(cls):
        return next(k for k, v in registry.factory.get_all().items() if cls == v)


class Module(torch.nn.Module, Generic[InT, BatchT], metaclass=ModuleMeta):
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

    @property
    def factory_name(self):
        return self.__class__.factory_name

    @property
    def cfg(self):
        return {"@factory": self.factory_name, **self._cfg}  # noqa

    def initialize(self, gold_data: Iterable[InT]):
        pass

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

    def reset_cache(self, cache=None):
        # print("Resetting", cache, "in", self.__class__.__name__)

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

    module_forward = torch.nn.Module.__call__

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

    def forward(self, batch: BatchT, supervision=False) -> Dict[str, Any]:
        """Forward pass of the torch module"""
        return batch


class TrainableComponentMeta(ComponentMeta, ModuleMeta):
    pass


class TrainableComponent(
    Module[InT, BatchT], Component[InT, OutT], metaclass=TrainableComponentMeta
):
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
        batch = self.make_batch(docs)

        inputs = self.collate(batch[self.name], device=self.device)
        res = self.forward(inputs)
        docs = self.postprocess(docs, res)
        return docs

    def postprocess(self, docs: Sequence[InT], batch: Dict[str, Any]) -> Sequence[OutT]:
        return docs
