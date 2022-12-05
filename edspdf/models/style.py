import re
from typing import Optional

import attrs

from edspdf.model import BaseModel

SEP_PATTERN = re.compile(r"[,-]")
SPACE_PATTERN = re.compile(r"\s")


class BaseStyle(BaseModel):
    """
    Model acting as an abstraction for a style.
    """

    # font: str
    # style: str
    # size: float
    italic: bool
    bold: bool

    fontname: Optional[str] = None

    dict = attrs.asdict


class Style(BaseStyle):
    """
    Model acting as an abstraction for a style.
    """


class SpannedStyle(BaseStyle):

    begin: int
    end: int


class StyledText(BaseModel):
    """
    Abstraction of a word, containing the style and the text.
    """

    text: str
    style: Style

    dict = attrs.asdict
