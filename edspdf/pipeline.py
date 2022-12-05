import copy
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch
from tqdm import tqdm

from .component import Component, TrainableComponent
from .config import Config, validate_arguments
from .registry import registry
from .utils.collections import batch_compress_dict, batchify, decompress_dict

T = TypeVar("T")


class ComponentsMap(dict):
    def __getattr__(self, item: str) -> Component:
        if item in self:
            return self[item]
        raise AttributeError(item)

    def __dir__(self) -> Iterable[str]:
        return (
            *self.keys(),
            *super(ComponentsMap, self).__dir__(),
        )


@validate_arguments
class Pipeline:
    def __init__(
        self,
        components: Optional[List[str]] = None,
        components_config: Optional[Dict[str, Component]] = None,
        batch_size: int = 4,
    ):
        """
        Pipeline object

        Parameters
        ----------
        components
        components_config
        batch_size
        """
        super().__init__()
        if components is None:
            components = ComponentsMap({})
        if components_config is None:
            components_config = {}
        self.components = ComponentsMap({})
        for name in components:
            component = components_config[name]
            component.name = name
            self.components[name] = component

        self.batch_size = batch_size
        self.meta = {}

    @property
    def trainable_components(self) -> List[TrainableComponent]:
        return [
            c
            for c in self.components.values()
            if isinstance(c, TrainableComponent) and c.needs_training
        ]

    def __iter__(self):
        return iter(self.components.values())

    def __len__(self):
        return len(self.components)

    def add_pipe(self, factory_name: Union[str, Callable], name=None, config=None):
        """
        Add a component to the pipeline

        Parameters
        ----------
        factory_name
        name
        config

        Returns
        -------

        """
        if isinstance(factory_name, str):
            if config is None:
                config = {}
            config = Config(config).resolve()
            cls = registry.factory.get(factory_name)
            component = cls(**config)
        elif hasattr(factory_name, "__call__") or hasattr(factory_name, "process"):
            if config is not None:
                raise TypeError(
                    "Cannot provide both an instantiated component and its config"
                )
            component = factory_name
            factory_name = getattr(component, "factory_name", None)
        else:
            raise TypeError(
                "`add_pipe` first argument must either be a factory name "
                f"or an instantiated component. You passed a {type(factory_name)}"
            )
        if name is None:
            if factory_name is None:
                raise TypeError(
                    "Could not automatically assign a name for component {}: either"
                    "provide a name explicitly to the add_pipe function, or define a "
                    "factory_name field on this component.".format(component)
                )
            name = factory_name

        self.components[name] = component
        component.name = name

        if not (hasattr(component, "process") or hasattr(component, "__call__")):
            raise TypeError("Component must have a process method or be callable")

        return component

    def reset_cache(self, cache=None):
        """
        Reset the caches of the components in this pipeline

        Parameters
        ----------
        cache

        Returns
        -------

        """
        for component in self.components.values():
            try:
                component.reset_cache(cache)
            except AttributeError:
                pass

    def __call__(self, doc):
        """
        Applies the pipeline on a sample

        Parameters
        ----------
        doc

        Returns
        -------

        """
        self.reset_cache()
        for name, component in self.components.items():
            doc = component(doc)
        return doc

    def train(self, mode=True):
        """
        Enables training on pytorch modules

        Parameters
        ----------
        mode

        Returns
        -------

        """
        for component in self.components.values():
            if hasattr(component, "train"):
                component.train(mode)

    @property
    def cfg(self):
        return Config(
            components=list(self.components.keys()),
            components_config=Config(**self.components, __path__=("components",)),
        )

    @contextmanager
    def no_cache(self):
        """
        Disable caching

        Returns
        -------

        """
        saved = []
        for component in self.components.values():
            if isinstance(component, TrainableComponent):
                saved.append((component, component.enable_cache(False)))
        yield
        for component, do_cache in saved:
            component.enable_cache(do_cache)

    def pipe(self, docs: Iterable) -> Iterable:
        """
        Apply the pipeline on a collection of documents

        Parameters
        ----------
        docs

        Returns
        -------

        """
        # Pourquoi pre-batch et pas enchainement d'iterateurs ?
        # Car:
        # - si iterateurs, tailles de batch par comp différentes (comp1: 16, comp2: 32)
        # - or, pour cacher efficacement il faut égalité entre batch
        # - tbc

        for batch in batchify(docs, batch_size=self.batch_size):
            self.reset_cache()
            for component in self.components.values():
                batch = component.batch_process(batch)
            yield from batch

    def initialize(self, data: Iterable[T]):
        """
        Initialize the components of the pipeline
        Each component must be initialized before
        the next components are run. Since a component might need
        the full training data to be initialized, all data may be
        fed to the component, making it impossible to enable batch caching.

        Therefore, we disable cache during the entire operation, so
        heavy computation (such as embeddings) that is usually shared will
        be repeated for each initialized component.

        Parameters
        ----------
        data: SupervisedData

        """
        # Component initialization
        print("Initializing components")

        for name, component in self.components.items():
            if not component.initialized:
                component.initialize(data)
                print(f"Component {repr(name)} initialized")

    def score(self, data: Sequence[T]):
        """
        Score a pipeline against a supervised dataset
        Parameters
        ----------
        data

        Returns
        -------

        """
        self.train(False)
        inputs: Sequence[T] = copy.deepcopy(data)
        golds: Iterable[Dict[str, T]] = data

        scored_components = {}

        # Predicting intermediate steps
        preds = defaultdict(lambda: [])
        for batch in batchify(
            tqdm(inputs, "Scoring components"), batch_size=self.batch_size
        ):
            self.reset_cache()
            for name, component in self.components.items():
                if component.scorer is not None:
                    scored_components[name] = component
                    batch = component.batch_process(batch)
                    preds[name].extend(copy.deepcopy(batch))

        t0 = time.time()
        for _ in tqdm(self.pipe(inputs), "Scoring pipeline", total=len(inputs)):
            pass
        duration = time.time() - t0

        # Scoring
        metrics: Dict[str, Any] = {
            "speed": len(inputs) / duration,
        }
        for name, component in scored_components.items():
            metrics[name] = component.score(list(zip(preds[name], golds)))
        return metrics

    def preprocess(self, doc: T, supervision: bool = False):
        prep = {}
        for name, component in self.components.items():
            if isinstance(component, TrainableComponent):
                prep[name] = component.preprocess(doc, supervision=supervision)
        return prep

    def preprocess_many(self, docs: Iterable[T], compress=True, supervision=True):
        preprocessed = (self.preprocess(doc, supervision=supervision) for doc in docs)
        if compress:
            return batch_compress_dict(preprocessed)
        return preprocessed

    def parameters(self):
        for component in self.components.values():
            if isinstance(component, torch.nn.Module):
                yield from component.parameters()

    def collate(self, batch, device: Optional[torch.device] = None):
        batch = decompress_dict(batch)
        if device is None:
            device = next(p.device for p in self.parameters())
        for name, component in self.components.items():
            if name in batch:
                component: TrainableComponent
                component_inputs = batch[name]
                batch[name] = component.collate(component_inputs, device)
        return batch

    def __repr__(self):
        return "Pipeline({})".format(
            "\n{}\n".format(
                "\n".join(
                    indent(f"({name}): " + repr(component), prefix="  ")
                    for name, component in self.components.items()
                )
            )
            if len(self.components)
            else ""
        )


def load(config: Union[Path, str, Config]):
    if isinstance(config, (Path, str)):
        config = Config.from_disk(config)
    elif isinstance(config, dict):
        config = Config(config)
    elif not isinstance(config, Config):
        raise Exception("The load function expects a Config or a path to a config file")

    assert "pipeline" in config, "The config object is missing a `pipeline` section"

    return Pipeline(**config["pipeline"].resolve())
