import copy
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
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

from .component import CacheEnum, Component, TrainableComponent
from .config import Config, validate_arguments
from .registry import registry
from .utils.collections import batch_compress_dict, batchify, decompress_dict, multi_tee

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


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
    """
    The Pipeline is the core object of EDS-PDF. It is responsible for the
    orchestration of the components and processing PDF documents end-to-end.

    A pipeline is usually created empty and then populated with components via the
    `add_pipe` method. Here is an example :

    ```python
    pipeline = Pipeline()
    pipeline.add_pipe("pdfminer-extractor")
    pipeline.add_pipe("mask-classifier")
    pipeline.add_pipe("simple-aggregator")
    ```
    """

    def __init__(
        self,
        components: Optional[List[str]] = None,
        components_config: Optional[Dict[str, Component]] = None,
        batch_size: int = 4,
    ):
        """
        Initializes the pipeline. The pipeline is empty by default and can be
        populated with components via the `add_pipe` method.

        Parameters
        ----------
        components: Optional[List[str]]
            List of component names
        components_config: Optional[Dict[str, Component]]
            Dictionary of component configurations. The keys of the dictionary must
            match the component names. The values are the component configurations,
            which can contain unresolved configuration for nested components or
            instances of components.
        batch_size: int
            The default number of documents to process in parallel when running the
            pipeline
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

    def add_pipe(self, factory_name: Union[str, Callable], name=None, config=None):
        """
        Adds a component to the pipeline. The component can be either a factory name or
        an instantiated component. If a factory name is provided, the component will be
        instantiated with the class from the registry matching the factory name and
        using the provided config as arguments.

        Parameters
        ----------
        factory_name: Union[str, Callable]
            Either a factory name or an instantiated component
        name: str
            Name of the component
        config: Dict
            Configuration of the component. The configuration can contain unresolved
            configuration for nested components such as
            `{"@factory_name": "my-sub-component", ...}`

        Returns
        -------
        Component
            The added component
        """
        if isinstance(factory_name, str):
            if config is None:
                config = {}
            # config = Config(config).resolve()
            cls = registry.factory.get(factory_name)
            component = cls(**config)
        elif hasattr(factory_name, "__call__") or hasattr(factory_name, "process"):
            if config is not None:
                raise TypeError(
                    "Cannot provide both an instantiated component and its config"
                )
            component = factory_name
            factory_name = next(
                k
                for k, v in registry.factory.get_all().items()
                if component.__class__ == v
            )
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

    def reset_cache(self, cache: Optional[CacheEnum] = None):
        """
        Reset the caches of the components in this pipeline

        Parameters
        ----------
        cache: Optional[CacheEnum]
            The cache to reset (either `preprocess`, `collate` or `forward`)
            If None, all caches are reset
        """
        for component in self.components.values():
            try:
                component.reset_cache(cache)
            except AttributeError:
                pass

    def __call__(self, doc: InputT) -> OutputT:
        """
        Applies the pipeline on a sample

        Parameters
        ----------
        doc: InputT
            The document to process

        Returns
        -------
        OutputT
        """
        self.reset_cache()
        for name, component in self.components.items():
            doc = component(doc)
        return doc

    def pipe(self, docs: Iterable[InputT]) -> Iterable:
        """
        Apply the pipeline on a collection of documents

        Parameters
        ----------
        docs: Iterable[InputT]
            The documents to process

        Returns
        -------
        Iterable
            An iterable collection of processed documents
        """
        for batch in batchify(docs, batch_size=self.batch_size):
            self.reset_cache()
            for component in self.components.values():
                batch = component.batch_process(batch)
            yield from batch

    def initialize(self, data: Iterable[InputT]):
        """
        Initialize the components of the pipeline
        Each component must be initialized before the next components are run.
        Since a component might need the full training data to be initialized, all
        data may be fed to the component, making it impossible to enable batch caching.

        Therefore, we disable cache during the entire operation, so heavy computation
        (such as embeddings) that is usually shared will be repeated for each
        initialized component.

        Parameters
        ----------
        data: SupervisedData

        """
        # Component initialization
        print("Initializing components")
        data = multi_tee(data)

        with self.no_cache():
            for name, component in self.components.items():
                if not component.initialized:
                    component.initialize(data)
                    print(f"Component {repr(name)} initialized")

    def score(self, docs: Sequence[InputT]):
        """
        Scores a pipeline against a sequence of annotated documents

        Parameters
        ----------
        docs: Sequence[InputT]
            The documents to score

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the metrics of the pipeline, as well as the speed of
            the pipeline. Each component that has a scorer will also be scored and its
            metrics will be included in the returned dictionary under a key named after
            each component.
        """
        self.train(False)
        inputs: Sequence[InputT] = copy.deepcopy(docs)
        golds: Iterable[Dict[str, InputT]] = docs

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

    def preprocess(self, doc: InputT, supervision: bool = False):
        """
        Runs the preprocessing methods of each component in the pipeline
        on a document and returns a dictionary containing the results, with the
        component names as keys.

        Parameters
        ----------
        doc: InputT
            The document to preprocess
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Dict[str, Any]
        """
        prep = {}
        for name, component in self.components.items():
            if isinstance(component, TrainableComponent):
                prep[name] = component.preprocess(doc, supervision=supervision)
        return prep

    def preprocess_many(self, docs: Iterable[InputT], compress=True, supervision=True):
        """
        Runs the preprocessing methods of each component in the pipeline on
        a collection of documents and returns an iterable of dictionaries containing
        the results, with the component names as keys.

        Parameters
        ----------
        docs: Iterable[InputT]
        compress: bool
            Whether to deduplicate identical preprocessing outputs of the results
            if multiple documents share identical subcomponents. This step is required
            to enable the cache mechanism when training or running the pipeline over a
            tabular datasets such as pyarrow tables that do not store referential
            equality information.
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Iterable[OutputT]
        """
        preprocessed = map(partial(self.preprocess, supervision=supervision), docs)
        if compress:
            return batch_compress_dict(preprocessed)
        return preprocessed

    def collate(self, batch: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Collates a batch of preprocessed samples into a single (maybe nested)
        dictionary of tensors by calling the collate method of each component.

        Parameters
        ----------
        batch: Dict[str, Any]
            The batch of preprocessed samples
        device: Optional[torch.device]
            The device to move the tensors to before returning them

        Returns
        -------
        Dict[str, Any]
            The collated batch
        """
        batch = decompress_dict(batch)
        if device is None:
            device = next(p.device for p in self.parameters())
        for name, component in self.components.items():
            if name in batch:
                component: TrainableComponent
                component_inputs = batch[name]
                batch[name] = component.collate(component_inputs, device)
        return batch

    def train(self, mode=True):
        """
        Enables training mode on pytorch modules

        Parameters
        ----------
        mode: bool
            Whether to enable training or not
        """
        for component in self.components.values():
            if hasattr(component, "train"):
                component.train(mode)

    @property
    def cfg(self):
        """Returns the initial configuration of the pipeline"""
        return Config(
            components=list(self.components.keys()),
            components_config=Config(**self.components, __path__=("components",)),
        ).serialize()

    @contextmanager
    def no_cache(self):
        """Disable caching for all (trainable) components in the pipeline"""
        saved = []
        for component in self.components.values():
            if isinstance(component, TrainableComponent):
                saved.append((component, component.enable_cache(False)))
        yield
        for component, do_cache in saved:
            component.enable_cache(do_cache)

    def parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        seen = set()
        for component in self.components.values():
            if isinstance(component, torch.nn.Module):
                for param in component.parameters():
                    if param in seen:
                        continue
                    seen.add(param)
                    yield param

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

    @property
    def trainable_components(self) -> List[TrainableComponent]:
        """Returns the list of trainable components in the pipeline."""
        return [
            c
            for c in self.components.values()
            if isinstance(c, TrainableComponent) and c.needs_training
        ]

    def __iter__(self):
        """Returns an iterator over the components in the pipeline."""
        return iter(self.components.values())

    def __len__(self):
        """Returns the number of components in the pipeline."""
        return len(self.components)


def load(config: Union[Path, str, Config]):
    if isinstance(config, (Path, str)):
        config = Config.from_disk(config)
    elif isinstance(config, dict):
        config = Config(config)
    elif not isinstance(config, Config):
        raise Exception("The load function expects a Config or a path to a config file")

    assert "pipeline" in config, "The config object is missing a `pipeline` section"

    return Pipeline(**config["pipeline"].resolve())
