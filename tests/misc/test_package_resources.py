from pathlib import Path

import pytest

from edspdf.misc import package_resource

BASE_DIR = Path(__file__).parent.parent.parent


def test_package_resource():
    path = package_resource("reg.py", "edspdf")
    assert path == BASE_DIR / "edspdf" / "reg.py"

    with pytest.raises(FileNotFoundError):
        package_resource("dummy.py", "edspdf")
