import re
from pathlib import Path

from edspdf import __version__ as init_version


def test_versions():
    poetry = (Path(__file__).parent.parent / "pyproject.toml").read_text()

    poetry_version = re.search(r'version = "(?P<version>.+)"', poetry).groupdict()[
        "version"
    ]

    assert init_version == poetry_version
