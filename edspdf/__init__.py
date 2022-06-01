import importlib.metadata

from edspdf import aggregation, classification, extraction, reading, transforms
from edspdf.reg import registry

__version__ = importlib.metadata.version("edspdf")
