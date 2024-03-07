# ruff: noqa: F401
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import pyarrow
import pyarrow.fs
from fsspec import AbstractFileSystem
from fsspec.implementations.arrow import ArrowFSWrapper
from loguru import logger

from edspdf import registry
from edspdf.data.base import BaseReader, BaseWriter
from edspdf.data.converters import (
    CONTENT,
    FILENAME,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edspdf.lazy_collection import LazyCollection
from edspdf.utils.collections import flatten


class FileReader(BaseReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        keep_ipynb_checkpoints: bool = False,
        load_annotations: bool = False,
        filesystem: Optional[Any] = None,
        recursive: bool = False,
    ):
        super().__init__()

        if filesystem is None or (isinstance(path, str) and "://" in path):
            path = (
                path
                if isinstance(path, Path) or "://" in path
                else f"file://{os.path.abspath(path)}"
            )
            inferred_fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
            filesystem = filesystem or inferred_fs
            assert inferred_fs.type_name == filesystem.type_name, (
                f"Protocol {inferred_fs.type_name} in path does not match "
                f"filesystem {filesystem.type_name}"
            )
            path = fs_path

        self.path = path
        self.filesystem = (
            ArrowFSWrapper(filesystem)
            if isinstance(filesystem, pyarrow.fs.FileSystem)
            else filesystem
        )
        self.load_annotations = load_annotations
        if not self.filesystem.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

        assert sys.version_info >= (3, 8) or not recursive, (
            "Recursive reading is only supported with Python 3.8 or higher. "
            "Please upgrade your Python version or set `recursive=False`."
        )
        glob_str = "**/*.pdf" if recursive else "*.pdf"
        self.files: List[str] = [
            file
            for file in self.filesystem.glob(os.path.join(str(self.path), glob_str))
            if (keep_ipynb_checkpoints or ".ipynb_checkpoints" not in str(file))
            and (
                not load_annotations
                or self.filesystem.exists(str(path).replace(".pdf", ".json"))
            )
        ]
        assert len(self.files), f"No .pdf files found in the directory {path}"
        logger.info(f"The directory contains {len(self.files)} .pdf files.")

    def read_main(self):
        return ((f, 1) for f in self.files)

    def read_worker(self, fragment):
        tasks = []
        for path in fragment:
            with self.filesystem.open(str(path), "rb") as f:
                content = f.read()

            json_path = str(path).replace(".pdf", ".json")

            record = {"content": content}
            if self.load_annotations and self.filesystem.exists(json_path):
                with self.filesystem.open(json_path) as f:
                    record["annotations"] = json.load(f)

            record[FILENAME] = str(os.path.relpath(path, self.path)).rsplit(".", 1)[0]
            record["id"] = record[FILENAME]
            tasks.append(record)
        return tasks


class FileWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        overwrite: bool = False,
        filesystem: Optional[AbstractFileSystem] = None,
    ):
        fs_path = path
        if filesystem is None or (isinstance(path, str) and "://" in path):
            path = (
                path
                if isinstance(path, Path) or "://" in path
                else f"file://{os.path.abspath(path)}"
            )
            inferred_fs, fs_path = pyarrow.fs.FileSystem.from_uri(path)
            filesystem = filesystem or inferred_fs
            assert inferred_fs.type_name == filesystem.type_name, (
                f"Protocol {inferred_fs.type_name} in path does not match "
                f"filesystem {filesystem.type_name}"
            )
            path = fs_path

        self.path = path
        self.filesystem = (
            ArrowFSWrapper(filesystem)
            if isinstance(filesystem, pyarrow.fs.FileSystem)
            else filesystem
        )
        self.filesystem.mkdirs(fs_path, exist_ok=True)

        if self.filesystem.exists(self.path):
            suffixes = Counter(f.suffix for f in self.filesystem.listdir(self.path))
            unsafe_suffixes = {
                s: v for s, v in suffixes.items() if s == ".pdf" or s == ".json"
            }
            if unsafe_suffixes and not overwrite:
                raise FileExistsError(
                    f"Directory {self.path} already exists and appear to contain "
                    "annotations:"
                    + "".join(f"\n -{s}: {v} files" for s, v in unsafe_suffixes.items())
                    + "\nUse overwrite=True to write files anyway."
                )

        self.filesystem.mkdirs(path, exist_ok=True)

        super().__init__()

    def write_worker(self, records):
        # If write as jsonl, we will perform the actual writing in the `write` method
        results = []
        for rec in flatten(records):
            filename = str(rec.pop(FILENAME))
            path = os.path.join(self.path, f"{filename}.pdf")
            parent_dir = filename.rsplit("/", 1)[0]
            if parent_dir and not self.filesystem.exists(parent_dir):
                self.filesystem.makedirs(parent_dir, exist_ok=True)
            if CONTENT in rec:
                content = rec.pop(CONTENT)
                with self.filesystem.open(path, "wb") as f:
                    f.write(content)
            ann_path = str(path).replace(".pdf", ".json")

            with self.filesystem.open(ann_path, "w") as f:
                json.dump(rec, f)

            results.append(path)
        return results, len(results)

    def write_main(self, fragments):
        return list(flatten(fragments))


# noinspection PyIncorrectDocstring
@registry.readers.register("files")
def read_files(
    path: Union[str, Path],
    *,
    keep_ipynb_checkpoints: bool = False,
    load_annotations: bool = False,
    converter: Optional[Union[str, Callable]] = None,
    filesystem: Optional[Any] = None,
    recursive: Optional[bool] = False,
    **kwargs,
) -> LazyCollection:
    """
    The BratReader (or `edspdf.data.read_files`) reads a directory of BRAT files and
    yields documents. At the moment, only entities and attributes are loaded. Relations
     and events are not supported.

    Example
    -------
    ```{ .python .no-check }

    import edspdf

    nlp = edspdf.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edspdf.data.read_files("path/to/brat/directory")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edspdf.data.read_files` returns a
        [LazyCollection][edspdf.core.lazy_collection.LazyCollection].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list :

        ```{ .python .no-check }
        docs = list(edspdf.data.read_files("path/to/brat/directory"))
        ```

    !!! warning "True/False attributes"

        Boolean values are not supported by the BRAT editor, and are stored as empty
        (key: empty value) if true, and not stored otherwise. This means that False
        values will not be assigned to attributes by default, which can be problematic
        when deciding if an entity is negated or not : is the entity not negated, or
        has the negation attribute not been annotated ?

        To avoid this issue, you can use the `bool_attributes` argument to specify
        which attributes should be considered as boolean when reading a BRAT dataset.
        These attributes will be assigned a value of `True` if they are present, and
        `False` otherwise.

        ```{ .python .no-check }
        doc_iterator = edspdf.data.read_files(
            "path/to/brat/directory",
            # Mapping from 'BRAT attribute name' to 'Doc attribute name'
            span_attributes={"Negation": "negated"},
            bool_attributes=["negated"],  # Missing values will be set to False
        )
        ```

    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory containing the BRAT files (will recursively look for
        files in subdirectories).
    nlp : Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    keep_ipynb_checkpoints : bool
        Whether to keep files in the `.ipynb_checkpoints` directories.
    load_annotations : bool
        Whether to load annotations from the `.json` files that share the same name as
        the `.pdf` files.
    converter : Optional[Union[str, Callable]]
        Converter to use to convert the dictionary objects to documents.
    filesystem: Optional[AbstractFileSystem]
        The filesystem to use to write the files. If not set, the local filesystem
        will be used.


    Returns
    -------
    LazyCollection
    """
    data = LazyCollection(
        reader=FileReader(
            path,
            keep_ipynb_checkpoints=keep_ipynb_checkpoints,
            load_annotations=load_annotations,
            filesystem=filesystem,
            recursive=recursive,
        )
    )
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


@registry.writers.register("files")
def write_files(
    data: Union[Any, LazyCollection],
    path: Union[str, Path],
    *,
    overwrite: bool = False,
    converter: Union[str, Callable],
    filesystem: Optional[AbstractFileSystem] = None,
    **kwargs,
) -> None:
    """
    `edspdf.data.write_files` writes a list of documents using the BRAT/File
    format in a directory. The BRAT files will be named after the `note_id` attribute of
    the documents, and subdirectories will be created if the name contains `/`
    characters.

    Example
    -------
    ```{ .python .no-check }

    import edspdf

    nlp = edspdf.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edspdf.data.write_files([doc], "path/to/brat/directory")
    ```

    !!! warning "Overwriting files"

        By default, `write_files` will raise an error if the directory already exists
        and contains files with `.json` or `.pdf` suffixes. This is to avoid overwriting
        existing annotations. To allow overwriting existing files, use `overwrite=True`.

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    path: Union[str, Path]
        Path to the directory containing the BRAT files (will recursively look for
        files in subdirectories).
    overwrite: bool
        Whether to overwrite existing directories.
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects.
    filesystem: Optional[AbstractFileSystem]
        The filesystem to use to write the files. If not set, the local filesystem
        will be used.
    """
    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(
        FileWriter(
            path=path,
            filesystem=filesystem,
            overwrite=overwrite,
        )
    )
