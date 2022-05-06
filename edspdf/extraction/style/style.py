import re
from typing import List

from pdfminer.layout import LTAnno, LTTextLineHorizontal

from .models import Style, Word

SPACE_PATTERN = re.compile(r"\s")


def create_words(line: LTTextLineHorizontal) -> List[Word]:
    """
    Generate a list of words from a PDFMiner line object.

    Parameters
    ----------
    line : LTTextLineHorizontal
        Line abstraction in PDFMiner

    Returns
    -------
    List[Word]
        List of ``Word`` objects.
    """
    words = []

    word = ""
    first_char = None
    for char in line:
        if not word:
            first_char = char
        if isinstance(char, LTAnno) or SPACE_PATTERN.match(char._text):
            if word:
                words.append(
                    Word(
                        text=word,
                        style=Style.from_fontname(
                            fontname=first_char.fontname,
                            size=first_char.size,
                            upright=first_char.upright,
                        ),
                    )
                )
                word = ""
        else:
            word += char._text
    return words


def words2style(words: List[Word]) -> str:
    """
    Transforms a list of ``Word``s into a textual representation.

    Parameters
    ----------
    words : List[Word]
        List of ``Word``s, containing word-by-word text and style.

    Returns
    -------
    str
        Styled textual representation, in the form
        ``<s font=Font size=10.0 style=Normal>text</s>``
    """

    if not words:
        return ""

    sublist = []
    style = words[0].style

    styled = []

    for word in words:
        if word.style == style:
            sublist.append(word)
        else:
            # Adding the previous substyle
            text = " ".join([w.text for w in sublist])
            styled.append(style(text))

            # Re-creating the list
            sublist = [word]
            style = word.style

    text = " ".join([w.text for w in sublist])
    styled.append(style(text))

    return " ".join(styled)


def line2style(line: LTTextLineHorizontal) -> str:
    """
    Transform a PDFMiner line (``LTTextLineHorizontal``) into a style-formatted string.

    Parameters
    ----------
    line : LTTextLineHorizontal
        Line abstraction in PDFMiner

    Returns
    -------
    str
        Styled extraction, ie formatted representation for the line.
    """
    words = create_words(line)
    return words2style(words)


def lines2style(lines: List[LTTextLineHorizontal]) -> str:
    """
    Transform a PDFMiner line (``LTTextLineHorizontal``) into a style-formatted string.
    Useful for text lines spanning multiple blocs (eg tables).

    Parameters
    ----------
    lines : List[LTTextLineHorizontal]
        List of lines (abstraction in PDFMiner).

    Returns
    -------
    str
        Formatted "sentence"
    """
    words = []
    for line in lines:
        words.extend(create_words(line))
    return words2style(words)
