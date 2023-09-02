import io
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import build
import dill
import toml
from build.__main__ import build_package, build_package_via_sdist
from confit import Cli
from dill._dill import save_function as dill_save_function
from dill._dill import save_type as dill_save_type
from importlib_metadata import PackageNotFoundError
from importlib_metadata import version as get_version
from loguru import logger
from typing_extensions import Literal

import edspdf

py_version = f"{sys.version_info.major}.{sys.version_info.minor}"


def get_package(obj_type: Type):
    # Retrieve the __package__ attribute of the module of a type, if possible.
    # And returns the package version as well
    try:
        module_name = obj_type.__module__
        if module_name == "__main__":
            raise Exception(f"Could not find package of type {obj_type}")
        module = __import__(module_name, fromlist=["__package__"])
        package = module.__package__
        try:
            version = get_version(package)
        except (PackageNotFoundError, ValueError):
            return None
        return package, version
    except (ImportError, AttributeError):
        raise Exception(f"Cound not find package of type {obj_type}")


def save_type(pickler, obj, *args, **kwargs):
    package_name = get_package(obj)
    if package_name is not None:
        pickler.packages.add(package_name)
    dill_save_type(pickler, obj, *args, **kwargs)


def save_function(pickler, obj, *args, **kwargs):
    package_name = get_package(obj)
    if package_name is not None:
        pickler.packages.add(package_name)
    return dill_save_function(pickler, obj, *args, **kwargs)


class PackagingPickler(dill.Pickler):
    dispatch = dill.Pickler.dispatch.copy()

    dispatch[FunctionType] = save_function
    dispatch[type] = save_type

    def __init__(self, *args, **kwargs):
        self.file = io.BytesIO()
        super().__init__(self.file, *args, **kwargs)
        self.packages = set()


def get_deep_dependencies(obj):
    pickler = PackagingPickler()
    pickler.dump(obj)
    return sorted(pickler.packages)


app = Cli(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


def snake_case(s):
    # From https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-97.php  # noqa E501
    return "_".join(
        re.sub(
            "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()


class ModuleName(str):
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("ModuleName is only meant for typing.")

    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(cls, value, config=None):
        if not isinstance(value, str):
            raise TypeError("string required")

        if not re.match(
            r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", value, flags=re.IGNORECASE
        ):
            raise ValueError("invalid identifier")
        return value


if TYPE_CHECKING:
    ModuleName = str  # noqa F811

POETRY_SNIPPET = """\
from poetry.core.masonry.builders.sdist import SdistBuilder
from poetry.factory import Factory
from poetry.core.masonry.utils.module import ModuleOrPackageNotFound
import sys
# Initialize the Poetry object for the current project
poetry = Factory().create_poetry("__root_dir__")

# Initialize the builder
try:
    builder = SdistBuilder(poetry, None, None)
except ModuleOrPackageNotFound:
    if not poetry.package.packages:
        print([])
        sys.exit(0)

print([
    {k: v for k, v in {
    "include": include._include,
    "from": include.source,
    "formats": include.formats,
    }.items()}
    for include in builder._module.includes
])

# Get the list of files to include
files = builder.find_files_to_add()

# Print the list of files
for file in files:
    print(file.path)
"""

INIT_PY = """\
import edspdf
from pathlib import Path

def load(device: "torch.device" = "cpu") -> edspdf.Pipeline:
    artifacts_path = Path(__file__).parent / "{artifacts_path}"
    model = edspdf.load(artifacts_path, device=device)
    return model
"""


# def parse_authors_as_dicts(authors):
#     authors = [authors] if isinstance(authors, str) else authors
#     return [
#         dict(zip(("name", "email"), re.match(r"(.*) <(.*)>", author).groups()))
#         if isinstance(author, str)
#         else author
#         for author in authors
#     ]


def parse_authors_as_strings(authors):
    authors = [authors] if isinstance(authors, str) else authors
    return [
        author if isinstance(author, str) else f"{author['name']} <{author['email']}>"
        for author in authors
    ]


class PoetryPackager:
    def __init__(
        self,
        pyproject: Optional[Dict[str, Any]],
        pipeline: Union[Path, "edspdf.Pipeline"],
        version: str,
        name: ModuleName,
        root_dir: Path = ".",
        build_dir: Path = "build",
        out_dir: Path = "dist",
        artifacts_name: ModuleName = "artifacts",
        dependencies: Optional[Sequence[Tuple[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = {},
    ):
        self.poetry_bin_path = (
            subprocess.run(["which", "poetry"], stdout=subprocess.PIPE)
            .stdout.decode()
            .strip()
        )
        self.version = version
        self.pyproject = pyproject
        self.root_dir = root_dir.resolve()
        self.build_dir = build_dir
        self.out_dir = self.root_dir / out_dir
        self.artifacts_name = artifacts_name
        self.name = name
        self.pipeline = pipeline
        self.dependencies = dependencies

        with self.ensure_pyproject(metadata):
            logger.info(f"root_dir: {self.root_dir}")
            logger.info(f"build_dir: {self.build_dir}")
            logger.info(f"artifacts_name: {self.artifacts_name}")
            logger.info(f"name: {self.name}")

            python_executable = (
                Path(self.poetry_bin_path).read_text().split("\n")[0][2:]
            )
            result = subprocess.run(
                [
                    *python_executable.split(),
                    "-c",
                    POETRY_SNIPPET.replace("__root_dir__", str(self.root_dir)),
                ],
                stdout=subprocess.PIPE,
                cwd=self.root_dir,
            )
            if result.returncode != 0:
                raise Exception()
            out = result.stdout.decode().strip().split("\n")
        self.poetry_packages = eval(out[0])
        self.file_paths = [self.root_dir / file_path for file_path in out[1:]]

    @contextmanager
    def ensure_pyproject(self, metadata):
        """Generates a Poetry based pyproject.toml"""
        metadata = dict(metadata)
        new_pyproject = self.pyproject is None
        if "authors" in metadata:
            metadata["authors"] = parse_authors_as_strings(metadata["authors"])
        try:
            if new_pyproject:
                self.pyproject = {
                    "build-system": {
                        "requires": ["poetry-core>=1.0.0"],
                        "build-backend": "poetry.core.masonry.api",
                    },
                    "tool": {
                        "poetry": {
                            **metadata,
                            "name": self.name,
                            "version": self.version,
                            "dependencies": {
                                "python": f">={py_version},<4.0",
                                **{
                                    dep_name: f"^{dep_version}"
                                    for dep_name, dep_version in self.dependencies
                                },
                            },
                        },
                    },
                }
                (self.root_dir / "pyproject.toml").write_text(
                    toml.dumps(self.pyproject)
                )
            else:
                for key, value in metadata.items():
                    pyproject_value = self.pyproject["tool"]["poetry"].get(key)
                    if pyproject_value != metadata[key]:
                        raise ValueError(
                            f"Field {key} in pyproject.toml doesn't match the one "
                            f"passed as argument, you should remove it from the "
                            f"metadata parameter. Avoid using metadata if you already "
                            f"have a pyproject.toml file.\n"
                            f"pyproject.toml:\n {pyproject_value}\n"
                            f"metadata:\n {value}"
                        )
            yield
        except Exception:
            if new_pyproject:
                os.remove(self.root_dir / "pyproject.toml")
            raise

    def list_files_to_add(self):
        # Extract python from the shebang in the poetry executable
        return self.file_paths

    def build(
        self,
        distributions: Sequence[str] = (),
        config_settings: Optional[build.ConfigSettingsType] = None,
        isolation: bool = True,
        skip_dependency_check: bool = False,
    ):
        logger.info(f"Building package {self.name}")

        if distributions:
            build_call = build_package
        else:
            build_call = build_package_via_sdist
            distributions = ["wheel"]
        build_call(
            srcdir=self.build_dir,
            outdir=self.out_dir,
            distributions=distributions,
            config_settings=config_settings,
            isolation=isolation,
            skip_dependency_check=skip_dependency_check,
        )

    def update_pyproject(self):
        # Replacing project name
        old_name = self.pyproject["tool"]["poetry"]["name"]
        self.pyproject["tool"]["poetry"]["name"] = self.name
        logger.info(
            f"Replaced project name {old_name!r} with {self.name!r} in poetry based "
            f"project"
        )

        old_version = self.pyproject["tool"]["poetry"]["version"]
        self.pyproject["tool"]["poetry"]["version"] = self.version
        logger.info(
            f"Replaced project version {old_version!r} with {self.version!r} in poetry "
            f"based project"
        )

        # Adding artifacts to include in pyproject.toml
        snake_name = snake_case(self.name.lower())
        included = self.pyproject["tool"]["poetry"].setdefault("include", [])
        included.append(f"{snake_name}/{self.artifacts_name}/**")

        packages = list(self.poetry_packages)
        packages.append({"include": snake_name})
        self.pyproject["tool"]["poetry"]["packages"] = packages

    def make_src_dir(self):
        snake_name = snake_case(self.name.lower())
        package_dir = self.build_dir / snake_name
        build_artifacts_dir = package_dir / self.artifacts_name
        for file_path in self.list_files_to_add():
            new_file_path = self.build_dir / Path(file_path).relative_to(self.root_dir)
            if isinstance(self.pipeline, Path) and self.pipeline in file_path.parents:
                raise Exception(
                    f"Pipeline ({self.artifacts_name}) is already "
                    "included in the package's data, you should "
                    "remove it from the pyproject.toml metadata."
                )
            os.makedirs(new_file_path.parent, exist_ok=True)
            logger.info(f"COPY {file_path} TO {new_file_path}")
            shutil.copy(file_path, new_file_path)

        self.update_pyproject()

        # Write pyproject.toml
        (self.build_dir / "pyproject.toml").write_text(toml.dumps(self.pyproject))

        if isinstance(self.pipeline, Path):
            # self.pipeline = edspdf.load(self.pipeline)
            shutil.copytree(
                self.pipeline,
                build_artifacts_dir,
            )
        else:
            self.pipeline.save(build_artifacts_dir)
        os.makedirs(package_dir, exist_ok=True)
        (package_dir / "__init__.py").write_text(
            INIT_PY.format(
                artifacts_path=os.path.relpath(build_artifacts_dir, package_dir)
            )
        )


@app.command(name="package")
def package(
    pipeline: Union[Path, "edspdf.Pipeline"],
    name: ModuleName,
    root_dir: Path = ".",
    artifacts_name: ModuleName = "artifacts",
    check_dependencies: bool = False,
    project_type: Optional[Literal["poetry", "setuptools"]] = None,
    version: str = "0.1.0",
    metadata: Optional[Dict[str, Any]] = {},
    distributions: Optional[Sequence[Literal["wheel", "sdist"]]] = ["wheel"],
    config_settings: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    isolation: bool = True,
    skip_build_dependency_check: bool = False,
):
    # root_dir = Path(".").resolve()
    pyproject_path = root_dir / "pyproject.toml"

    if not pyproject_path.exists():
        check_dependencies = True

    dependencies = None
    if check_dependencies:
        if isinstance(pipeline, Path):
            pipeline = edspdf.load(pipeline)
        dependencies = get_deep_dependencies(pipeline)
        for dep in dependencies:
            print("DEPENDENCY", dep[0].ljust(30), dep[1])

    root_dir = root_dir.resolve()
    build_dir = root_dir / "build" / name
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir)

    pyproject = None
    if pyproject_path.exists():
        pyproject = toml.loads((root_dir / "pyproject.toml").read_text())

        if "tool" in pyproject and "poetry" in pyproject["tool"]:
            project_type = "poetry"

    if project_type == "poetry":
        packager = PoetryPackager(
            pyproject=pyproject,
            pipeline=pipeline,
            name=name,
            version=version,
            root_dir=root_dir,
            build_dir=build_dir,
            artifacts_name=artifacts_name,
            dependencies=dependencies,
            metadata=metadata,
        )
    else:
        raise Exception(
            "Could not infer project type, only poetry based projects are "
            "supported for now"
        )

    packager.make_src_dir()
    packager.build(
        distributions=distributions,
        config_settings=config_settings,
        isolation=isolation,
        skip_dependency_check=skip_build_dependency_check,
    )
