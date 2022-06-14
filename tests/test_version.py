import re
from pathlib import Path


def test_versions():
    init = Path("edspdf/__init__.py").read_text()
    poetry = Path("pyproject.toml").read_text()

    init_version = re.search(r'__version__ = "(?P<version>.+)"', init).groupdict()[
        "version"
    ]
    poetry_version = re.search(r'version = "(?P<version>.+)"', poetry).groupdict()[
        "version"
    ]

    assert init_version == poetry_version
