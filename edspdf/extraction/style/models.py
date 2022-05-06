import re

from pydantic import BaseModel

SEP_PATTERN = re.compile(r",|-")


class Style(BaseModel):
    """
    Model acting as an abstraction for a style.
    """

    font: str
    style: str
    size: float
    upright: bool

    @classmethod
    def from_fontname(cls, fontname: str, size: float, upright: bool) -> "Style":
        """
        Constructor using the compound ``fontname`` representation.

        Parameters
        ----------
        fontname : str
            Compound description of the font. Often ``Arial``,
            ``Arial,Bold`` or ``Arial-Bold``
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

        return cls(font=font, style=style, size=size, upright=upright)

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

    def decorate(self, text: str) -> str:
        """
        Decorates a string of text with the given style, in the form:
        ``<s font=Font size=10.0 style=Normal>text</s>``

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
        "Alias" for the ``decorate`` method.

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


class Word(BaseModel):
    """
    Abstraction of a word, containing the style and the text.
    """

    text: str
    style: Style
