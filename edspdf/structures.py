from typing import Dict, List, Optional, Union

import attrs
import numpy as np
from confit import Config

from .registry import registry


class BaseModel:
    # @classmethod
    # def add_extension(cls, name, **kwargs):
    #     def getter(self):
    #         if name in self.__dict__:
    #             return self.__dict__[name]
    #         return kwargs["default"]
    #
    #     def setter(self, value):
    #         self.__dict__[name] = value
    #
    #     setattr(cls, name, property(
    #         fget=getter,
    #         fset=setter,
    #     ))

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(v)
                for v in cls.__get_validators__()
            ]
        )

    @classmethod
    def validate(cls, v, config=None):
        if isinstance(v, dict):
            v = Config(v).resolve(deep=False, registry=registry)
        if isinstance(v, dict):
            v = cls(**v)
        return v

    dict = attrs.asdict

    copy = attrs.evolve

    evolve = attrs.evolve


@attrs.define(kw_only=True, hash=True)
class PDFDoc(BaseModel):
    """
    This is the main data structure of the library to hold PDFs.
    It contains the content of the PDF, as well as box annotations and text outputs.

    Attributes
    ----------
    content : bytes
        The content of the PDF document.
    id : str, optional
        The ID of the PDF document.
    pages : List[Page]
        The pages of the PDF document.
    error : bool, optional
        Whether there was an error when processing this PDF document.
    content_boxes : List[Union[TextBox, ImageBox]]
        The content boxes/annotations of the PDF document.
    aggregated_texts : Dict[str, Text]
        The aggregated text outputs of the PDF document.
    text_boxes : List[TextBox]
        The text boxes of the PDF document.
    """

    content: bytes = attrs.field(repr=lambda c: f"{len(c)} bytes")
    id: str = None
    num_pages: int = 0
    pages: List["Page"] = attrs.field(factory=list)
    error: bool = False
    content_boxes: List[Union["TextBox"]] = attrs.field(factory=list)
    aggregated_texts: Dict[str, "Text"] = attrs.field(factory=dict)

    @property
    def text_boxes(self) -> List["TextBox"]:
        return [box for box in self.content_boxes if isinstance(box, TextBox)]

    lines = text_boxes


@attrs.define(kw_only=True, hash=True)
class Page(BaseModel):
    """
    The `Page` class represents a page of a PDF document.

    Attributes
    ----------
    page_num : int
        The page number of the page.
    width : float
        The width of the page.
    height : float
        The height of the page.
    doc : PDFDoc
        The PDF document that this page belongs to.
    image : Optional[np.ndarray]
        The rendered image of the page, stored as a NumPy array.
    text_boxes : List[TextBox]
        The text boxes of the page.
    """

    page_num: int
    width: Optional[float] = None
    height: Optional[float] = None
    doc: PDFDoc = attrs.field(repr=False, default=None, eq=False)
    image: Optional[np.ndarray] = None

    @property
    def text_boxes(self):
        return [
            box
            for box in self.doc.content_boxes
            if isinstance(box, TextBox) and box.page_num == self.page_num
        ]


@attrs.define(kw_only=True, hash=True)
class TextProperties(BaseModel):
    """
    The `TextProperties` class represents the style properties of a span of text in a
    [TextBox][edspdf.structures.TextBox].

    Attributes
    ----------
    italic : bool
        Whether the text is italic.
    bold : bool
        Whether the text is bold.
    begin : int
        The beginning index of the span of text.
    end : int
        The ending index of the span of text.
    fontname : Optional[str]
        The font name of the span of text.
    """

    italic: bool
    bold: bool
    begin: int
    end: int
    fontname: Optional[str] = None


@attrs.define(kw_only=True, hash=True)
class Box(BaseModel):
    """
    The `Box` class represents a box annotation in a PDF document. It is the base class
    of [TextBox][edspdf.structures.TextBox].

    Attributes
    ----------
    doc : PDFDoc
        The PDF document that this box belongs to.
    page_num : Optional[int]
        The page number of the box.
    x0 : float
        The left x-coordinate of the box.
    x1 : float
        The right x-coordinate of the box.
    y0 : float
        The top y-coordinate of the box.
    y1 : float
        The bottom y-coordinate of the box.
    label : Optional[str]
        The label of the box.
    page : Page
        The page object that this box belongs to.
    """

    x0: float
    x1: float
    y0: float
    y1: float

    label: Optional[str] = None
    doc: "PDFDoc" = attrs.field(repr=False, default=None, eq=False)
    page_num: Optional[int] = attrs.field(default=None)

    @property
    def page(self):
        return next(p for p in self.doc.pages if p.page_num == self.page_num)

    def __lt__(self, other):
        self_page_num = self.page_num or 0
        other_page_num = other.page_num or 0
        if self_page_num < other_page_num:
            return True
        if self_page_num > other_page_num:
            return False

        alpha = 0.2
        beta = 1 - alpha
        self_x0 = self.x0 * beta + self.x1 * alpha
        self_x1 = self.x0 * alpha + self.x1 * beta
        self_y0 = self.y0 * beta + self.y1 * alpha
        self_y1 = self.y0 * alpha + self.y1 * beta

        other_x0 = other.x0 * beta + other.x1 * alpha
        other_x1 = other.x0 * alpha + other.x1 * beta
        other_y0 = other.y0 * beta + other.y1 * alpha
        other_y1 = other.y0 * alpha + other.y1 * beta

        dy0 = other_y1 - self_y0
        dy1 = other_y0 - self_y1
        if dy0 > 0 and dy1 > 0:
            return True
        if dy0 < 0 and dy1 < 0:
            return False
        dx0 = other_x1 - self_x0
        dx1 = other_x0 - self_x1

        if dx0 > 0 and dx1 > 0:
            return True
        if dx0 < 0 and dx1 < 0:
            return False

        return ((self.y0 + self.y1) / 2, (self.x0 + self.x1) / 2) < (
            (other.y0 + other.y1) / 2,
            (other.x0 + other.x1) / 2,
        )


@attrs.define(kw_only=True, hash=True)
class Text(BaseModel):
    """
    The `TextBox` class represents text object, not bound to any box.

    It can be used to store aggregated text from multiple boxes for example.

    Attributes
    ----------
    text : str
        The text content.
    properties : List[TextProperties]
        The style properties of the text.
    """

    text: str
    properties: List[TextProperties] = attrs.field(factory=list)

    def __repr__(self):
        return f"Text(text={self.text!r}, properties={self.properties!r})"

    def __str__(self):
        return self.text


@attrs.define(kw_only=True, hash=True)
class TextBox(Box):
    """
    The `TextBox` class represents a text box annotation in a PDF document.

    Attributes
    ----------
    text : str
        The text content of the text box.
    props : List[TextProperties]
        The style properties of the text box.
    """

    text: str
    props: List[TextProperties] = attrs.field(factory=list)
