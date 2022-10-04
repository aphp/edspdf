import re
from typing import Optional

from pdfminer.layout import LTChar
from pydantic import BaseModel

SEP_PATTERN = re.compile(r",|-")
SPACE_PATTERN = re.compile(r"\s")


class BaseStyle(BaseModel):
    """
    Model acting as an abstraction for a style.
    """

    fontname: Optional[str] = None

    font: str
    style: str
    size: float
    upright: bool

    x0: float
    x1: float
    y0: float
    y1: float


class Style(BaseStyle):
    """
    Model acting as an abstraction for a style.
    """

    @classmethod
    def from_fontname(
        cls,
        fontname: str,
        size: float,
        upright: bool,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
    ) -> "Style":
        """
        Constructor using the compound `fontname` representation.

        Parameters
        ----------
        fontname : str
            Compound description of the font. Often `Arial`,
            `Arial,Bold` or `Arial-Bold`
        size : float
            Character size.
        upright : bool
            Whether the character is upright.

        Returns
        -------
        Style
            Style representation.
        """
        # Round the size to avoid floating point aberrations.
        size = round(size, 2)

        s = SEP_PATTERN.split(fontname)

        font = s.pop(0)

        if s:
            style = s[-1]
        else:
            style = "Normal"

        s = Style(
            fontname=fontname,
            font=font,
            style=style,
            size=size,
            upright=upright,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )

        return s

    @classmethod
    def from_char(
        cls,
        char: LTChar,
        width: float,
        height: float,
    ):
        return cls.from_fontname(
            fontname=char.fontname,
            size=char.size,
            upright=char.upright,
            x0=char.x0 / width,
            x1=char.x1 / width,
            y0=1 - char.y1 / height,
            y1=1 - char.y0 / height,
        )

    def __eq__(self, other: "Style") -> bool:
        """
        Computes equality between two styles.

        Parameters
        ----------
        other : Style
            Style object to compare.

        Returns
        -------
        bool
            Whether the two styles are equal.
        """

        s = (self.font, self.style, round(self.size, 2), self.upright)
        o = (other.font, other.style, round(other.size, 2), other.upright)

        return s == o

    def __add__(self, other: "Style") -> "Style":

        if self != other:
            raise ValueError("You cannot add two different styles")

        st = self.copy()

        st.x0 = min(self.x0, other.x0)
        st.x1 = max(self.x1, other.x1)
        st.y0 = min(self.y0, other.y0)
        st.y1 = max(self.y1, other.y1)

        return st


class SpannedStyle(BaseStyle):

    start: int
    end: int


class StyledText(BaseModel):
    """
    Abstraction of a word, containing the style and the text.
    """

    text: str
    style: Style

    @classmethod
    def from_char(
        cls,
        char: LTChar,
        width: float,
        height: float,
    ):
        return StyledText(
            text=SPACE_PATTERN.sub(" ", char._text),
            style=Style.from_char(char, width=width, height=height),
        )

    def add_space(self) -> None:
        self.text = f"{self.text.rstrip()} "

    def rstrip(self) -> None:
        self.text = self.text.rstrip()

    def __add__(self, other: "StyledText") -> "StyledText":

        st = StyledText(
            text=self.text + other.text,
            style=self.style + other.style,
        )

        return st

    def __iadd__(self, other: "StyledText") -> "StyledText":
        return self + other
