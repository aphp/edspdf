from pdfminer.layout import LTTextLineHorizontal
from pydantic import BaseModel


class Line(BaseModel):

    page: int
    bloc: int

    x0: float
    x1: float
    y0: float
    y1: float

    page_width: float
    page_height: float

    line: LTTextLineHorizontal

    class Config:
        arbitrary_types_allowed = True
