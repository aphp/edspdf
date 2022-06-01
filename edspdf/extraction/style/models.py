import re

from pdfminer.layout import LTChar
from pydantic import BaseModel

SEP_PATTERN = re.compile(r",|-")
SPACE_PATTERN = re.compile(r"\s")


class BaseStyle(BaseModel):
    """
    Model acting as an abstraction for a style.
    """

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
    def from_char(cls, char: LTChar):
        return cls.from_fontname(
            fontname=char.fontname,
            size=char.size,
            upright=char.upright,
            x0=char.x0,
            x1=char.x1,
            y0=char.y0,
            y1=char.y1,
        )

    def __str__(self) -> str:
        """Representation for the style"""
        s = f"font={self.font} size={round(self.size, 2)} style={self.style}"
        if not self.upright:
            s += " italics"
        return s

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

        st = Style(
            font=self.font,
            style=self.style,
            size=self.size,
            upright=self.upright,
            x0=min(self.x0, other.x0),
            x1=max(self.x1, other.x1),
            y0=min(self.y0, other.y0),
            y1=max(self.y1, other.y1),
        )

        return st

    def decorate(self, text: str) -> str:
        """
        Decorates a string of text with the given style, in the form:
        `<s font=Font size=10.0 style=Normal>text</s>`

        Parameters
        ----------
        text : str
            Text to decorate.

        Returns
        -------
        str
            Decorated string.
        """
        return f"<s {self}>{text}</s>"

    def __call__(self, text: str) -> str:
        """
        "Alias" for the `decorate` method.

        Parameters
        ----------
        text : str
            Text to decorate.

        Returns
        -------
        str
            Decorated string.
        """
        return self.decorate(text)


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
    def from_char(cls, char: LTChar):
        return StyledText(
            text=SPACE_PATTERN.sub(" ", char._text),
            style=Style.from_char(char),
        )

    def add_space(self) -> None:
        self.text = f"{self.text.rstrip()} "

    def strip(self) -> None:
        self.text = self.text.rstrip()

    def __len__(self) -> int:
        """Return length of the text"""
        return len(self.text)

    def __add__(self, other: "StyledText") -> "StyledText":

        if self.style != other.style:
            raise ValueError("You cannot add two different styles")

        st = StyledText(
            text=self.text + other.text,
            style=self.style + other.style,
        )

        return st

    def __iadd__(self, other: "StyledText") -> "StyledText":
        return self + other

    def __truediv__(self, other: "StyledText") -> "StyledText":

        if self.style != other.style:
            raise ValueError("You cannot add two different styles")

        st = StyledText(
            text=f"{self.text} {other.text}",
            style=self.style + other.style,
        )

        return st

    def __itruediv__(self, other: "StyledText") -> "StyledText":
        return self / other
