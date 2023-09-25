import functools
import json
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import (
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
from confit.utils.collections import join_path, split_path
from confit.utils.xjson import Reference
from pydantic import parse_obj_as
from typing_extensions import Literal

import edspdf

from .accelerators.base import Accelerator, FromDoc, ToDoc
from .registry import CurriedFactory, registry
from .structures import PDFDoc
from .utils.collections import (
    FrozenList,
    batch_compress_dict,
    decompress_dict,
    multi_tee,
)

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
            raise e.with_traceback(None) from None
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
                    raise ValueError(
                        "The provided name does not match the name of the component."
                    )
                else:
                    name = pipe.name
            else:
                if name is None:
                    raise ValueError(
                        "The component does not have a name, so you must provide one",
                    )
                pipe.name = name
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
        inputs: Any,
        batch_size: Optional[int] = None,
        *,
        accelerator: Optional[Union[str, Accelerator]] = None,
        to_doc: Optional[ToDoc] = None,
        from_doc: FromDoc = lambda doc: doc,
    ) -> Iterable[PDFDoc]:
        """
        Process a stream of documents by applying each component successively on
        batches of documents.

        Parameters
        ----------
        inputs: Iterable[Union[str, PDFDoc]]
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
        Iterable[PDFDoc]
        """

        if batch_size is None:
            batch_size = self.batch_size

        if accelerator is None:
            accelerator = "simple"
        if isinstance(accelerator, str):
            accelerator = {"@accelerator": accelerator, "batch_size": batch_size}
        if isinstance(accelerator, dict):
            accelerator = Config(accelerator).resolve(registry=registry)

        kwargs = {
            "inputs": inputs,
            "model": self,
            "to_doc": parse_obj_as(Optional[ToDoc], to_doc),
            "from_doc": parse_obj_as(Optional[FromDoc], from_doc),
        }
        for k, v in list(kwargs.items()):
            if v is None:
                del kwargs[k]

        with self.train(False):
            return accelerator(**kwargs)

    @contextmanager
    def cache(self):
        """
        Enable caching for all (trainable) components in the pipeline
        """
        was_not_cached = self._cache is None
        if was_not_cached:
            self._cache = {}
        yield
        if was_not_cached:
            self._cache = None

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

    def post_init(self, gold_data: Iterable[PDFDoc], exclude: Optional[set] = None):
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
        exclude: Optional[set]
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
            raise e from None

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
        """
        Pydantic validators generator
        """
        yield cls.validate

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
                    batch[name] = component.collate(component_inputs, device)
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

    def to(self, device: Optional["torch.device"] = None):  # noqa F821
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

    def save(
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

        def save_tensors(path: Path):
            import safetensors.torch

            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
            tensors = defaultdict(list)
            tensor_to_group = defaultdict(list)
            for pipe_name, pipe in self.trainable_pipes(disable=exclude):
                for key, tensor in pipe.state_dict(keep_vars=True).items():
                    full_key = join_path((pipe_name, *split_path(key)))
                    tensors[tensor].append(full_key)
                    tensor_to_group[tensor].append(pipe_name)
            group_to_tensors = defaultdict(set)
            for tensor, group in tensor_to_group.items():
                group_to_tensors["+".join(sorted(set(group)))].add(tensor)
            for group, group_tensors in group_to_tensors.items():
                sub_path = path / f"{group}.safetensors"
                tensor_dict = {
                    "+".join(tensors[p]): p
                    for p in {p.data_ptr(): p for p in group_tensors}.values()
                }
                safetensors.torch.save_file(tensor_dict, sub_path)

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

        extra_exclude = set(exclude)
        for pipe_name, pipe in self._components:
            if hasattr(pipe, "save_extra_data") and pipe_name not in extra_exclude:
                pipe.save_extra_data(path / pipe_name, exclude=extra_exclude)
        if "tensors" not in exclude:
            save_tensors(path / "tensors")

    def load_state_from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Set[str] = None,
        device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
    ) -> "Pipeline":
        """
        Load the pipeline from a directory. Components will be updated in-place.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to load the pipeline from
        exclude: Set[str]
            The names of the components, or attributes to exclude from the loading
            process. This list will be gradually filled in place as components are
            loaded
        """

        def deserialize_meta(path: Path) -> None:
            if path.exists():
                data = json.loads(path.read_text())
                self.meta.update(data)

        def deserialize_tensors(path: Path):
            import safetensors.torch

            trainable_components = dict(self.trainable_pipes())
            for file_name in path.iterdir():
                pipe_names = file_name.stem.split("+")
                if any(pipe_name in trainable_components for pipe_name in pipe_names):
                    # We only load tensors in one of the pipes since parameters
                    # are expected to be shared
                    pipe = trainable_components[pipe_names[0]]
                    tensor_dict = {}
                    for keys, tensor in safetensors.torch.load_file(
                        file_name, device=device
                    ).items():
                        split_keys = [split_path(key) for key in keys.split("+")]
                        key = next(key for key in split_keys if key[0] == pipe_names[0])
                        tensor_dict[join_path(key[1:])] = tensor
                    # Non-strict because tensors of a given pipeline can be shared
                    # between multiple files
                    print(f"Loading tensors of {pipe_names[0]} from {file_name}")
                    extra_tensors = set(tensor_dict) - set(
                        pipe.state_dict(keep_vars=True).keys()
                    )
                    if extra_tensors:
                        warnings.warn(
                            f"{file_name} contains tensors that are not in the state"
                            f"dict of {pipe_names[0]}: {sorted(extra_tensors)}"
                        )
                    pipe.load_state_dict(tensor_dict, strict=False)

        exclude = set() if exclude is None else exclude

        if "meta" not in exclude:
            deserialize_meta(path / "meta.json")

        extra_exclude = set(exclude)
        for name, proc in self._components:
            if hasattr(proc, "load_extra_data") and name not in extra_exclude:
                proc.load_extra_data(path / name, extra_exclude)

        if "tensors" not in exclude:
            deserialize_tensors(path / "tensors")

        self._path = path  # type: ignore[assignment]
        return self

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        exclude: Optional[Set[str]] = None,
        device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
    ):
        path = Path(path) if isinstance(path, str) else path
        config = Config.from_disk(path / "config.cfg")
        self = Pipeline.from_config(config)
        self.load_state_from_disk(path, exclude=exclude, device=device)
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
        if isinstance(disable, str):
            disable = [disable]
        pipe_names = set(self.pipe_names)
        if enable is not None:
            if isinstance(enable, str):
                enable = [enable]
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
        artifacts_name: str = "artifacts",
        check_dependencies: bool = False,
        project_type: Optional[Literal["poetry", "setuptools"]] = None,
        version: str = "0.1.0",
        metadata: Optional[Dict[str, Any]] = {},
        distributions: Optional[Sequence[Literal["wheel", "sdist"]]] = ["wheel"],
        config_settings: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        isolation: bool = True,
        skip_build_dependency_check: bool = False,
    ):
        from .utils.package import package

        return package(
            pipeline=self,
            name=name,
            root_dir=root_dir,
            artifacts_name=artifacts_name,
            check_dependencies=check_dependencies,
            project_type=project_type,
            version=version,
            metadata=metadata,
            distributions=distributions,
            config_settings=config_settings,
            isolation=isolation,
            skip_build_dependency_check=skip_build_dependency_check,
        )


def load(
    config: Union[Path, str, Config],
    device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
) -> Pipeline:
    error = "The load function expects a Config or a path to a config file"
    if isinstance(config, (Path, str)):
        path = Path(config)
        if path.is_dir():
            return Pipeline.load(path, device=device)
        elif path.is_file():
            config = Config.from_disk(path)
        else:
            raise ValueError(error)
    elif not isinstance(config, Config):
        raise ValueError(error)

    return Pipeline.from_config(config)
