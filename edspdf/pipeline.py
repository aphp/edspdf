import functools
import importlib
import inspect
import json
import os
import shutil
import warnings
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from confit import Config
from confit.errors import ConfitValidationError, patch_errors
from confit.utils.xjson import Reference
from typing_extensions import Literal

import edspdf

from .data.converters import FILENAME
from .lazy_collection import LazyCollection
from .registry import CurriedFactory, registry
from .structures import PDFDoc
from .utils.collections import (
    FrozenList,
    batch_compress_dict,
    decompress_dict,
    multi_tee,
)

if TYPE_CHECKING:
    import torch

EMPTY_LIST = FrozenList()


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


Pipe = TypeVar("Pipe", bound=Callable[[PDFDoc], PDFDoc])


class Pipeline:
    """
    Pipeline to build hybrid and multitask PDF processing pipeline.
    It uses PyTorch as the deep-learning backend and allows components to share
    subcomponents.

    See the documentation for more details.
    """

    def __init__(
        self,
        batch_size: Optional[int] = 4,
        meta: Dict[str, Any] = None,
    ):
        """
        Parameters
        ----------
        batch_size: Optional[int]
            Batch size to use in the `.pipe()` method
        meta: Dict[str, Any]
            Meta information about the pipeline
        """
        self.batch_size = batch_size
        self.meta = dict(meta) if meta is not None else {}
        self._components: List[Tuple[str, Pipe]] = []
        self._disabled: List[str] = []
        self._path: Optional[Path] = None
        self._cache: Optional[Dict] = None

    @property
    def pipeline(self) -> List[Tuple[str, Pipe]]:
        return FrozenList(self._components)

    @property
    def pipe_names(self) -> List[str]:
        return FrozenList([name for name, _ in self._components])

    def get_pipe(self, name: str) -> Pipe:
        """
        Get a component by its name.

        Parameters
        ----------
        name: str
            The name of the component to get.

        Returns
        -------
        Pipe
        """
        for n, pipe in self.pipeline:
            if n == name:
                return pipe
        raise ValueError(f"Pipe {repr(name)} not found in pipeline.")

    def has_pipe(self, name: str) -> bool:
        """
        Check if a component exists in the pipeline.

        Parameters
        ----------
        name: str
            The name of the component to check.

        Returns
        -------
        bool
        """
        return any(n == name for n, _ in self.pipeline)

    def create_pipe(
        self,
        factory: str,
        name: str,
        config: Dict[str, Any] = None,
    ) -> Pipe:
        """
        Create a component from a factory name.

        Parameters
        ----------
        factory: str
            The name of the factory to use
        name: str
            The name of the component
        config: Dict[str, Any]
            The config to pass to the factory

        Returns
        -------
        Pipe
        """
        try:
            curried_factory: CurriedFactory = Config(
                {
                    "@factory": factory,
                    **(config if config is not None else {}),
                }
            ).resolve(registry=registry)
            pipe = curried_factory.instantiate(pipeline=self, path=(name,))
        except ConfitValidationError as e:
            raise e.with_traceback(None)
        return pipe

    def add_pipe(
        self,
        factory: Union[str, Pipe],
        first: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Pipe:
        """
        Add a component to the pipeline.

        Parameters
        ----------
        factory: Union[str, Pipe]
            The name of the component to add or the component itself
        name: Optional[str]
            The name of the component. If not provided, the name of the component
            will be used if it has one (.name), otherwise the factory name will be used.
        first: bool
            Whether to add the component to the beginning of the pipeline. This argument
            is mutually exclusive with `before` and `after`.
        before: Optional[str]
            The name of the component to add the new component before. This argument is
            mutually exclusive with `after` and `first`.
        after: Optional[str]
            The name of the component to add the new component after. This argument is
            mutually exclusive with `before` and `first`.
        config: Dict[str, Any]
            The arguments to pass to the component factory.

            Note that instead of replacing arguments with the same keys, the config
            will be merged with the default config of the component. This means that
            you can override specific nested arguments without having to specify the
            entire config.

        Returns
        -------
        Pipe
            The component that was added to the pipeline.
        """
        if isinstance(factory, str):
            if name is None:
                name = factory
            pipe = self.create_pipe(factory, name, config)
        else:
            if config is not None:
                raise ValueError(
                    "Can't pass config or name with an instantiated component",
                )
            pipe = factory
            if hasattr(pipe, "name"):
                if name is not None and name != pipe.name:
                    warnings.warn(
                        "The provided name does not match the name of the component."
                    )
                    pipe.name = name
                else:
                    name = pipe.name
            if name is None:
                raise ValueError(
                    "The component does not have a name, so you must provide one",
                )
        assert sum([before is not None, after is not None, first]) <= 1, (
            "You can only use one of before, after, or first",
        )
        insertion_idx = (
            0
            if first
            else self.pipe_names.index(before)
            if before is not None
            else self.pipe_names.index(after) + 1
            if after is not None
            else len(self._components)
        )
        self._components.insert(insertion_idx, (name, pipe))
        return pipe

    def ensure_doc(self, doc):
        return (
            doc
            if isinstance(doc, PDFDoc)
            else PDFDoc(content=doc)
            if isinstance(doc, bytes)
            else PDFDoc(
                content=doc["content"],
                id=doc.get("id") or doc.get(FILENAME),
            )
        )

    def __call__(self, doc: Any) -> PDFDoc:
        """
        Apply each component successively on a document.

        Parameters
        ----------
        doc: Union[str, PDFDoc]
            The doc to create the PDFDoc from, or a PDFDoc.

        Returns
        -------
        PDFDoc
        """
        with self.cache():
            for name, pipe in self.pipeline:
                if name in self._disabled:
                    continue
                # This is a hack to get around the ambiguity
                # between the __call__ method of Pytorch modules
                # and the __call__ methods of spacy components
                if hasattr(pipe, "batch_process"):
                    doc = next(iter(pipe.batch_process([doc])))
                else:
                    doc = pipe(doc)

        return doc

    def pipe(
        self,
        inputs: Union[LazyCollection, Iterable],
        batch_size: Optional[int] = None,
        *,
        accelerator: Any = None,
        to_doc: Any = None,
        from_doc: Any = None,
    ) -> LazyCollection:
        """
        Process a stream of documents by applying each component successively on
        batches of documents.

        Parameters
        ----------
        inputs: Union[LazyCollection, Iterable]
            The inputs to create the PDFDocs from, or the PDFDocs directly.
        batch_size: Optional[int]
            The batch size to use. If not provided, the batch size of the pipeline
            object will be used.
        accelerator: Optional[Union[str, Accelerator]]
            The accelerator to use for processing the documents. If not provided,
            the default accelerator will be used.
        to_doc: ToDoc
            The function to use to convert the inputs to PDFDoc objects. By default,
            the `content` field of the inputs will be used if dict-like objects are
            provided, otherwise the inputs will be passed directly to the pipeline.
        from_doc: FromDoc
            The function to use to convert the PDFDoc objects to outputs. By default,
            the PDFDoc objects will be returned directly.

        Returns
        -------
        LazyCollection
        """

        if batch_size is None:
            batch_size = self.batch_size

        lazy_collection = LazyCollection.ensure_lazy(inputs)

        if to_doc is not None:
            warnings.warn(
                "The `to_doc` argument is deprecated. "
                "Please use the returned value's `map` method or the read/from_{} "
                "method's converter argument instead.",
                DeprecationWarning,
            )
            if isinstance(to_doc, str):
                to_doc = {"content_field": to_doc}
            if isinstance(to_doc, dict):
                to_doc_dict = to_doc

                def to_doc(doc):
                    return PDFDoc(
                        content=doc[to_doc_dict["content_field"]],
                        id=doc[to_doc_dict["id_field"]]
                        if "id_field" in to_doc_dict
                        else None,
                    )

            if not callable(to_doc):
                raise ValueError(
                    "The `to_doc` argument must be a callable or a dictionary",
                )
            lazy_collection = lazy_collection.map(to_doc)

        lazy_collection = lazy_collection.map_pipeline(self).set_processing(
            batch_size=batch_size
        )

        if accelerator is not None:
            warnings.warn(
                "The `accelerator` argument is deprecated. "
                "Please use the returned value's `set_processing` method instead.",
                DeprecationWarning,
            )
            if isinstance(accelerator, str):
                kwargs = {}
                backend = accelerator
            elif isinstance(accelerator, dict):
                kwargs = dict(accelerator)
                backend = kwargs.pop("@accelerator", "simple")
            elif "Accelerator" in type(accelerator).__name__:
                backend = (
                    "multiprocessing"
                    if "Multiprocessing" in type(accelerator).__name__
                    else "simple"
                )
                kwargs = accelerator.__dict__
            lazy_collection.set_processing(
                backend=backend,
                **kwargs,
            )
        if from_doc is not None:
            warnings.warn(
                "The `from_doc` argument is deprecated. "
                "Please use the returned value's `map` method or the write/to_{} "
                "method's converter argument instead.",
                DeprecationWarning,
            )
            if isinstance(from_doc, dict):
                from_doc_dict = from_doc

                def from_doc(doc):
                    return {k: getattr(doc, v) for k, v in from_doc_dict.items()}

            if not callable(from_doc):
                raise ValueError(
                    "The `from_doc` argument must be a callable or a dictionary",
                )
            lazy_collection = lazy_collection.map(from_doc)

        return lazy_collection

    @contextmanager
    def cache(self):
        """
        Enable caching for all (trainable) components in the pipeline
        """
        to_disable = set()
        for name, pipe in self.trainable_pipes():
            if getattr(pipe, "_current_cache_id", None) is None:
                pipe.enable_cache()
                to_disable.add(name)
        yield
        for name, pipe in self.trainable_pipes():
            if name in to_disable:
                pipe.disable_cache()

    def trainable_pipes(
        self, disable: Sequence[str] = ()
    ) -> Iterable[Tuple[str, "edspdf.trainable_pipe.TrainablePipe"]]:
        """
        Yields components that are PyTorch modules.

        Parameters
        ----------
        disable: Sequence[str]
            The names of disabled components, which will be skipped.

        Returns
        -------
        Iterable[Tuple[str, 'edspdf.trainable_pipe.TrainablePipe']]
        """
        for name, pipe in self.pipeline:
            if name not in disable and hasattr(pipe, "batch_process"):
                yield name, pipe

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: Optional[Set] = None):
        """
        Completes the initialization of the pipeline by calling the post_init
        method of all components that have one.
        This is useful for components that need to see some data to build
        their vocabulary, for instance.

        Parameters
        ----------
        gold_data: Iterable[PDFDoc]
            The documents to use for initialization.
            Each component will not necessarily see all the data.
        exclude: Optional[Set]
            The names of components to exclude from initialization.
            This argument will be gradually updated  with the names of initialized
            components
        """
        gold_data = multi_tee(gold_data)
        exclude = set() if exclude is None else exclude
        for name, pipe in self._components:
            if hasattr(pipe, "post_init"):
                pipe.post_init(gold_data, exclude=exclude)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any] = {},
        *,
        disable: Optional[Set[str]] = None,
        enable: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a pipeline from a config object

        Parameters
        ----------
        config: Dict[str, Any]
            The config to use
        disable: Union[str, Iterable[str]]
            Components to disable
        enable: Union[str, Iterable[str]]
            Components to enable
        exclude: Union[str, Iterable[str]]
            Components to exclude
        meta: Dict[str, Any]
            Metadata to add to the pipeline

        Returns
        -------
        Pipeline
        """
        root_config = Config(config).copy()
        disable = disable if disable is not None else set()
        enable = enable if enable is not None else set()
        exclude = exclude if exclude is not None else set()
        meta = meta if meta is not None else {}

        loc_prefix = []
        if isinstance(config.get("pipeline"), dict):
            loc_prefix = ["pipeline"]
            if "components" in config and "components" not in config["pipeline"]:
                config["pipeline"]["components"] = Reference("components")
                loc_prefix = ["components"]
            config = config["pipeline"]
        loc_prefix.append("components")

        assert (
            "pipeline" in config and "components" in config
        ), "EDS-PDF config must contain a 'pipeline' and a 'components' key"

        config = Config(config).resolve(root=root_config, registry=registry)

        model = Pipeline(meta=meta)

        components = config.get("components", {})
        pipeline = config.get("pipeline", ())

        # Since components are actually resolved as curried factories,
        # we need to instantiate them here
        for name, component in components.items():
            if not isinstance(component, CurriedFactory):
                raise ValueError(
                    f"Component {repr(name)} is not instantiable. Please make sure "
                    "that you didn't forget to add a '@factory' key to the component "
                    "config."
                )

        try:
            components = CurriedFactory.instantiate(components, pipeline=model)
        except ConfitValidationError as e:
            e = ConfitValidationError(
                e.raw_errors,
                model=cls,
                name=cls.__module__ + "." + cls.__qualname__,
            )
            e.raw_errors = patch_errors(e.raw_errors, loc_prefix)
            raise e

        for name in pipeline:
            if name in exclude:
                continue
            if name not in components:
                raise ValueError(f"Component {repr(name)} not found in config")
            model.add_pipe(components[name], name=name)

        # Set of components name if it's in the disable list
        # or if it's not in the enable list and the enable list is not empty
        model._disabled = [
            name
            for name in model.pipe_names
            if name in disable or (enable and name not in enable)
        ]
        return model

    @property
    def disabled(self):
        """
        The names of the disabled components
        """
        return FrozenList(self._disabled)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(v)
                for v in cls.__get_validators__()
            ]
        )

    @classmethod
    def validate(cls, v, config=None):
        """
        Pydantic validator, used in the `validate_arguments` decorated functions
        """
        if isinstance(v, dict):
            return cls.from_config(v)
        if not isinstance(v, cls):
            raise ValueError("input is not a Pipeline or config dict")
        return v

    def preprocess(self, doc: PDFDoc, supervision: bool = False):
        """
        Run the preprocessing methods of each component in the pipeline
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
        with self.cache():
            for name, component in self.pipeline:
                prep_fn = getattr(
                    component,
                    "preprocess_supervised" if supervision else "preprocess",
                    None,
                )
                if prep_fn is not None:
                    prep[name] = prep_fn(doc)
        return prep

    def preprocess_many(self, docs: Iterable[PDFDoc], compress=True, supervision=True):
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
        preprocessed = map(
            functools.partial(self.preprocess, supervision=supervision), docs
        )
        if compress:
            preprocessed = batch_compress_dict(preprocessed)
        return preprocessed

    def collate(
        self,
        batch: List[Dict[str, Any]],
        device: Optional["torch.device"] = None,  # noqa F821
    ):
        """
        Collates a batch of preprocessed samples into a single (maybe nested)
        dictionary of tensors by calling the collate method of each component.

        Parameters
        ----------
        batch: List[Dict[str, Any]]
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
        with self.cache():
            for name, component in self.pipeline:
                if name in batch:
                    component_inputs = batch[name]
                    batch[name] = component.collate(component_inputs)
        return batch

    def parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        return (p for n, p in self.named_parameters())

    def named_parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        seen = set()
        for name, component in self.pipeline:
            if hasattr(component, "named_parameters"):
                for param_name, param in component.named_parameters():
                    if param in seen:
                        continue
                    seen.add(param)
                    yield f"{name}.{param_name}", param

    def to(self, device: Union[str, Optional["torch.device"]] = None):  # noqa F821
        """Moves the pipeline to a given device"""
        for name, component in self.trainable_pipes():
            component.to(device)
        return self

    def train(self, mode=True):
        """
        Enables training mode on pytorch modules

        Parameters
        ----------
        mode: bool
            Whether to enable training or not
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                for name, proc in self.trainable_pipes():
                    proc.train(was_training[name])

        was_training = {name: proc.training for name, proc in self.trainable_pipes()}
        for name, proc in self.trainable_pipes():
            proc.train(mode)

        return context()

    def to_disk(
        self, path: Union[str, Path], *, exclude: Optional[Set[str]] = None
    ) -> None:
        """
        Save the pipeline to a directory.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to save the pipeline to. Every component will be
            saved to separated subdirectories of this directory, except for tensors
            that will be saved to a shared files depending on the references between
            the components.
        exclude: Optional[Set[str]]
            The names of the components, or attributes to exclude from the saving
            process. This list will be gradually filled in place as components are
            saved
        """
        exclude = set() if exclude is None else exclude

        path = Path(path) if isinstance(path, str) else path

        if os.path.exists(path) and not os.path.exists(path / "config.cfg"):
            raise FileExistsError(
                "The directory already exists and doesn't appear to be a"
                "saved pipeline. Please erase it manually or choose a "
                "different directory."
            )
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        if "meta" not in exclude:
            (path / "meta.json").write_text(json.dumps(self.meta))
        if "config" not in exclude:
            self.config.to_disk(path / "config.cfg")

        pwd = os.getcwd()
        overrides = {"components": {}}
        try:
            os.chdir(path)
            for pipe_name, pipe in self._components:
                if hasattr(pipe, "to_disk") and pipe_name not in exclude:
                    pipe_overrides = pipe.to_disk(Path(pipe_name), exclude=exclude)
                    overrides["components"][pipe_name] = pipe_overrides
        finally:
            os.chdir(pwd)

        config = self.config.merge(overrides)

        if "config" not in exclude:
            config.to_disk(path / "config.cfg")

    save = to_disk

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Optional[Union[str, Sequence[str]]] = None,
        device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
    ) -> "Pipeline":
        """
        Load the pipeline from a directory. Components will be updated in-place.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to load the pipeline from
        exclude: Optional[Union[str, Sequence[str]]]
            The names of the components, or attributes to exclude from the loading
            process.
        device: Optional[Union[str, "torch.device"]]
            Device to use when loading the tensors
        """

        def deserialize_meta(path: Path) -> None:
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                self.meta.update(data)
                # self.meta always overrides meta["vectors"] with the metadata
                # from self.vocab.vectors, so set the name directly

        exclude = (
            set()
            if exclude is None
            else {exclude}
            if isinstance(exclude, str)
            else set(exclude)
        )

        path = (Path(path) if isinstance(path, str) else path).absolute()
        if "meta" not in exclude:
            deserialize_meta(path / "meta.json")

        pwd = os.getcwd()
        try:
            os.chdir(path)
            for name, proc in self._components:
                if hasattr(proc, "from_disk") and name not in exclude:
                    proc.from_disk(Path(name), exclude=exclude)
                # Convert to list here in case exclude is (default) tuple
                exclude.add(name)
        finally:
            os.chdir(pwd)

        self._path = path  # type: ignore[assignment]
        self.train(False)
        return self

    # override config property getter to remove "factory" key from components
    @property
    def cfg(self) -> Config:
        """
        Returns the config of the pipeline, including the config of all components.
        Updated from spacy to allow references between components.
        """
        return Config(
            {
                "pipeline": list(self.pipe_names),
                "components": {key: component for key, component in self._components},
                "disabled": list(self.disabled),
            }
        )

    @property
    def config(self) -> Config:
        config = Config({"pipeline": self.cfg.copy()})
        config["components"] = config["pipeline"].pop("components")
        config["pipeline"]["components"] = Reference("components")
        return config.serialize()

    def select_pipes(
        self,
        *,
        disable: Optional[Union[str, Iterable[str]]] = None,
        enable: Optional[Union[str, Iterable[str]]] = None,
    ):
        """
        Temporarily disable and enable components in the pipeline.

        Parameters
        ----------
        disable: Optional[Union[str, Iterable[str]]]
            The name of the component to disable, or a list of names.
        enable: Optional[Union[str, Iterable[str]]]
            The name of the component to enable, or a list of names.
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                self._disabled = disabled_before

        if enable is None and disable is None:
            raise ValueError("Expected either `enable` or `disable`")
        disable = [disable] if isinstance(disable, str) else disable
        pipe_names = set(self.pipe_names)
        if enable is not None:
            enable = [enable] if isinstance(enable, str) else enable
            if set(enable) - pipe_names:
                raise ValueError(
                    "Enabled pipes {} not found in pipeline.".format(
                        sorted(set(enable) - pipe_names)
                    )
                )
            to_disable = [pipe for pipe in self.pipe_names if pipe not in enable]
            # raise an error if the enable and disable keywords are not consistent
            if disable is not None and disable != to_disable:
                raise ValueError("Inconsistent values for `enable` and `disable`")
            disable = to_disable

        if set(disable) - pipe_names:
            raise ValueError(
                "Disabled pipes {} not found in pipeline.".format(
                    sorted(set(disable) - pipe_names)
                )
            )

        disabled_before = self._disabled
        self._disabled = disable
        return context()

    def package(
        self,
        name: Optional[str] = None,
        root_dir: Union[str, Path] = ".",
        build_dir: Union[str, Path] = "build",
        dist_dir: Union[str, Path] = "dist",
        artifacts_name: str = "artifacts",
        project_type: Optional[Literal["poetry", "setuptools"]] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = {},
        distributions: Optional[Sequence[Literal["wheel", "sdist"]]] = ["wheel"],
        config_settings: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        isolation: bool = True,
        skip_build_dependency_check: bool = False,
    ):
        from edspdf.utils.package import package

        return package(
            pipeline=self,
            name=name,
            root_dir=root_dir,
            build_dir=build_dir,
            dist_dir=dist_dir,
            artifacts_name=artifacts_name,
            project_type=project_type,
            version=version,
            metadata=metadata,
            distributions=distributions,
            config_settings=config_settings,
            isolation=isolation,
            skip_build_dependency_check=skip_build_dependency_check,
        )


def load(
    model: Union[Path, str, Config],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    exclude: Optional[Union[str, Iterable[str]]] = None,
    device: Optional[Union[str, "torch.device"]] = "cpu",
):
    """
    Load a pipeline from a config file or a directory.

    Examples
    --------

    ```{ .python .no-check }
    import edspdf

    nlp = edspdf.load(
        "path/to/config.cfg",
        overrides={"components": {"my_component": {"arg": "value"}}},
    )
    ```

    Parameters
    ----------
    model: Union[Path, str, Config]
        The config to use for the pipeline, or the path to a config file or a directory.
    overrides: Optional[Dict[str, Any]]
        Overrides to apply to the config when loading the pipeline. These are the
        same parameters as the ones used when initializing the pipeline.
    exclude: Optional[Union[str, Iterable[str]]]
        The names of the components, or attributes to exclude from the loading
        process. :warning: The `exclude` argument will be mutated in place.
    device: Optional[Union[str, "torch.device"]]
        Device to use when loading the tensors

    Returns
    -------
    Pipeline
    """
    error = (
        "The load function expects either :\n"
        "- a confit Config object\n"
        "- the path of a config file (.cfg file)\n"
        "- the path of a trained model\n"
        "- the name of an installed pipeline package\n"
        f"but got {model!r} which is neither"
    )
    if isinstance(model, (Path, str)):
        path = Path(model)
        is_dir = path.is_dir()
        is_config = path.is_file() and path.suffix == ".cfg"
        try:
            module = importlib.import_module(model)
            is_package = True
        except (ImportError, AttributeError, TypeError):
            module = None
            is_package = False
        if is_dir and is_package:
            warnings.warn(
                "The path provided is both a directory and a package : edspdf will "
                "load the package. To load from the directory instead, please pass the "
                f'path as "./{path}" instead.'
            )
        if is_dir:
            path = (Path(path) if isinstance(path, str) else path).absolute()
            config = Config.from_disk(path / "config.cfg")
            if overrides:
                config = config.merge(overrides)
            pwd = os.getcwd()
            try:
                os.chdir(path)
                nlp = Pipeline.from_config(config)
                nlp.from_disk(path, exclude=exclude, device=device)
            finally:
                os.chdir(pwd)
            return nlp
        elif is_config:
            model = Config.from_disk(path)
        elif is_package:
            # Load as package
            available_kwargs = {
                "overrides": overrides,
                "exclude": exclude,
                "device": device,
            }
            signature_kwargs = inspect.signature(module.load).parameters
            kwargs = {
                name: available_kwargs[name]
                for name in signature_kwargs
                if name in available_kwargs
            }
            return module.load(**kwargs)

    if not isinstance(model, Config):
        raise ValueError(error)

    return Pipeline.from_config(model)
