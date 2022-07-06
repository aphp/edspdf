from importlib import resources
from pathlib import Path
from typing import Optional

from edspdf.reg import registry


@registry.misc("package-resource.v1")
def package_resource(path: Path, package: Optional[Path] = None) -> Path:

    with resources.path(package=package, resource="__init__.py") as p:
        path = p.parent / path

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    return path
