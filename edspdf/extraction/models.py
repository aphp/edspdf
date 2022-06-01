from typing import List

from pydantic import BaseModel

from .style.models import SpannedStyle


class Line(BaseModel):

    page: int
    bloc: int

    x0: float
    x1: float
    y0: float
    y1: float

    page_width: float
    page_height: float

    text: str
    styles: List[SpannedStyle]
