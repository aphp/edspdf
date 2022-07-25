from pathlib import Path

from thinc.api import Config

from edspdf import registry
from edspdf.readers.reader import PdfReader


def load(path: Path) -> PdfReader:
    """
    Load a complete pipeline.

    TODO: implement other ways to load a pipeline.

    Parameters
    ----------
    path : Path
        Path to the pipeline.

    Returns
    -------
    PdfReader
        A PdfReader object.
    """
    conf = Config().from_disk(path)
    return registry.resolve(conf)["reader"]


def from_str(config: str) -> PdfReader:
    """
    Load a complete pipeline from a string config.

    Parameters
    ----------
    config : str
        Configuration.

    Returns
    -------
    PdfReader
        A PdfReader object.
    """
    conf = Config().from_str(config)
    return registry.resolve(conf)["reader"]
