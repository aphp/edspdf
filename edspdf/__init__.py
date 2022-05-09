import importlib.metadata

import edspdf.classification.factory
import edspdf.extraction.factory
import edspdf.reading.factory
import edspdf.transforms.factory
from edspdf.reg import registry

__version__ = importlib.metadata.version("edspdf")
