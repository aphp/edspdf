from __future__ import annotations

import contextlib
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from typing_extensions import Literal

import edspdf.data

if TYPE_CHECKING:
    import torch

    from edspdf import Pipeline
    from edspdf.data.base import BaseReader, BaseWriter
    from edspdf.trainable_pipe import TrainablePipe

INFER = type("INFER", (), {"__repr__": lambda self: "INFER"})()


class MetaLazyCollection(type):
    def __getattr__(self, item):
        if item in edspdf.data.__all__:
            fn = getattr(edspdf.data, item)
            setattr(self, item, fn)
            return fn
        raise AttributeError(item)

    def __dir__(self):  # pragma: no cover
        return (*super().__dir__(), *edspdf.data.__all__)


class LazyCollection(metaclass=MetaLazyCollection):
    def __init__(
        self,
        reader: Optional[BaseReader] = None,
        writer: Optional[BaseWriter] = None,
        pipeline: List[Any] = [],
        config={},
    ):
        self.reader = reader
        self.writer = writer
        self.pipeline: List[Tuple[str, Callable, Dict]] = pipeline
        self.config = config

    @property
    def batch_size(self):
        return self.config.get("batch_size", 1)

    @property
    def batch_unit(self):
        return self.config.get("batch_unit", 1)

    @property
    def chunk_size(self):
        return self.config.get("chunk_size", self.config.get("batch_size", 128))

    @property
    def disable_implicit_parallelism(self):
        return self.config.get("disable_implicit_parallelism", True)

    @property
    def num_cpu_workers(self):
        return self.config.get("num_cpu_workers")

    @property
    def num_gpu_workers(self):
        return self.config.get("num_gpu_workers")

    @property
    def gpu_pipe_names(self):
        return self.config.get("gpu_pipe_names")

    @property
    def gpu_worker_devices(self):
        return self.config.get("gpu_worker_devices")

    @property
    def cpu_worker_devices(self):
        return self.config.get("cpu_worker_devices")

    @property
    def backend(self):
        return self.config.get("backend")

    @property
    def show_progress(self):
        return self.config.get("show_progress")

    @property
    def process_start_method(self):
        return self.config.get("process_start_method")

    def set_processing(
        self,
        batch_size: int = INFER,
        batch_unit: Literal["docs", "pages", "lines"] = INFER,
        chunk_size: int = INFER,
        num_cpu_workers: int = INFER,
        num_gpu_workers: int = INFER,
        disable_implicit_parallelism: bool = INFER,
        backend: Literal["simple", "multiprocessing"] = INFER,
        gpu_pipe_names: List[str] = INFER,
        show_progress: bool = INFER,
        process_start_method: bool = INFER,
        gpu_worker_devices: List[str] = INFER,
        cpu_worker_devices: List[str] = INFER,
    ) -> "LazyCollection":
        """
        Parameters
        ----------
        batch_size: int
            Number of documents to process at a time in a GPU worker (or in the
            main process if no workers are used).
        batch_unit: Literal["docs", "pages", "lines"]
            The unit of the batch size. Can be "docs" or "words". If "words", the
            batch size is total number of words in the documents.
        chunk_size: int
            Number of documents to build before splitting into batches. Only used
            with "simple" and "multiprocessing" backends. This is also the number of
            documents that will be passed through the first components of the pipeline
            until a GPU worker is used (then the will be split according to the
            `batch_size` and `batch_unit`).
        num_cpu_workers: int
            Number of CPU workers. A CPU worker handles the non deep-learning components
            and the preprocessing, collating and postprocessing of deep-learning
            components. If no GPU workers are used, the CPU workers also handle the
            forward call of the deep-learning components.
        num_gpu_workers: Optional[int]
            Number of GPU workers. A GPU worker handles the forward call of the
            deep-learning components. Only used with "multiprocessing" backend.
        disable_implicit_parallelism: bool
            Whether to disable OpenMP and Huggingface tokenizers implicit parallelism in
            multiprocessing mode. Defaults to True.
        gpu_pipe_names: Optional[List[str]]
            List of pipe names to accelerate on a GPUWorker, defaults to all pipes
            that inherit from TrainablePipe. Only used with "multiprocessing" backend.
        backend: Optional[Literal["simple", "multiprocessing", "spark"]]
            The backend to use for parallel processing. If not set, the backend is
            automatically selected based on the input data and the number of workers.

            - "simple" is the default backend and is used when `num_cpu_workers` is 1
                and `num_gpu_workers` is 0.
            - "multiprocessing" is used when `num_cpu_workers` is greater than 1 or
                `num_gpu_workers` is greater than 0.
            - "spark" is used when the input data is a Spark dataframe and the output
                writer is a Spark writer.
        show_progress: Optional[bool]
            Whether to show progress bars (only applicable with "simple" and
            "multiprocessing" backends).
        process_start_method: Optional[bool]
            Whether to use "fork" or "spawn" as the start method for the multiprocessing
            backend.

            - "fork" is the default start method on Unix systems and is the fastest
                start method, but it is not available on Windows, can cause issues
                with CUDA and is not safe when using multiple threads.
            - "spawn" is the default start method on Windows and is the safest start
                method, but it is not available on Unix systems and is slower than
                "fork".
        gpu_worker_devices: Optional[List[str]]
            List of GPU devices to use for the GPU workers. Defaults to all available
            devices, one worker per device. Only used with "multiprocessing" backend.
        cpu_worker_devices: Optional[List[str]]
            List of GPU devices to use for the CPU workers. Used for debugging purposes.

        Returns
        -------
        LazyCollection
        """
        kwargs = dict(locals())
        kwargs.pop("self")
        return LazyCollection(
            reader=self.reader,
            writer=self.writer,
            pipeline=self.pipeline,
            config={
                **self.config,
                **{k: v for k, v in kwargs.items() if v is not INFER},
            },
        )

    @classmethod
    def ensure_lazy(cls, data):
        from edspdf.data.base import IterableReader

        if isinstance(data, cls):
            return data
        return cls(reader=IterableReader(data))

    def map(self, pipe, name: Optional[str] = None, kwargs={}) -> "LazyCollection":
        return LazyCollection(
            reader=self.reader,
            writer=self.writer,
            pipeline=[*self.pipeline, (name, pipe, kwargs)],
            config=self.config,
        )

    def map_pipeline(self, model: Pipeline) -> "LazyCollection":
        new_steps = []
        for name, pipe, kwargs in self.pipeline:
            new_steps.append((name, pipe, kwargs))

        new_steps.append(("_ensure_doc", model.ensure_doc, {}))

        for name, pipe in model.pipeline:
            if name not in model._disabled:
                new_steps.append((name, pipe, {}))
        config = (
            {**self.config, "batch_size": model.batch_size}
            if self.batch_size is None
            else self.config
        )
        return LazyCollection(
            reader=self.reader,
            writer=self.writer,
            pipeline=new_steps,
            config=config,
        )

    def write(self, writer: BaseWriter, execute: bool = True) -> Any:
        lc = LazyCollection(
            reader=self.reader,
            writer=writer,
            pipeline=self.pipeline,
            config=self.config,
        )
        return lc.execute() if execute else lc

    def execute(self):
        import edspdf.processing

        backend = self.backend
        if backend is None:
            try:
                SparkReader = sys.modules.get("edspdf.data.spark").SparkReader
                SparkWriter = sys.modules.get("edspdf.data.spark").SparkWriter
            except (KeyError, AttributeError):  # pragma: no cover
                SparkReader = SparkWriter = None
            if (
                SparkReader
                and isinstance(self.reader, SparkReader)
                and SparkWriter
                and (self.writer is None or isinstance(self.writer, SparkWriter))
            ):
                backend = "spark"
            elif (
                self.num_cpu_workers is not None or self.num_gpu_workers is not None
            ) and (
                self.num_cpu_workers is not None
                and self.num_cpu_workers > 1
                or self.num_gpu_workers is not None
                and self.num_gpu_workers > 0
            ):
                backend = "multiprocessing"
            else:
                backend = "simple"
        execute = getattr(edspdf.processing, f"execute_{backend}_backend")
        return execute(self)

    def __iter__(self):
        return iter(self.execute())

    @contextlib.contextmanager
    def cache(self):
        for name, pipe, *_ in self.pipeline:
            if hasattr(pipe, "enable_cache"):
                pipe.enable_cache()
        yield
        for name, pipe, *_ in self.pipeline:
            if hasattr(pipe, "disable_cache"):
                pipe.disable_cache()

    def torch_components(
        self, disable: Container[str] = ()
    ) -> Iterable[Tuple[str, "TrainablePipe"]]:
        """
        Yields components that are PyTorch modules.

        Parameters
        ----------
        disable: Container[str]
            The names of disabled components, which will be skipped.

        Returns
        -------
        Iterable[Tuple[str, TrainablePipe]]
        """
        for name, pipe, *_ in self.pipeline:
            if name not in disable and hasattr(pipe, "forward"):
                yield name, pipe

    def to(self, device: Union[str, Optional["torch.device"]] = None):  # noqa F821
        """Moves the pipeline to a given device"""
        for name, pipe, *_ in self.torch_components():
            pipe.to(device)
        return self

    def worker_copy(self):
        return LazyCollection(
            reader=self.reader.worker_copy(),
            writer=self.writer,
            pipeline=self.pipeline,
            config=self.config,
        )

    def __dir__(self):  # pragma: no cover
        return (*super().__dir__(), *edspdf.data.__all__)

    def __getattr__(self, item):
        return getattr(LazyCollection, item).__get__(self)


if TYPE_CHECKING:
    # just to add read/from_* and write/to_* methods to the static type hints
    LazyCollection = edspdf.data  # noqa: F811
