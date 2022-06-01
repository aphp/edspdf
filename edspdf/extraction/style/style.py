import re
from typing import Iterable, List, Tuple

from pdfminer.layout import LTAnno, LTTextLineHorizontal

from .models import SpannedStyle, Style, StyledText

SPACE_PATTERN = re.compile(r"\s")


def extract_text(line: LTTextLineHorizontal) -> Tuple[str, List[SpannedStyle]]:

    styles = []
    texts = []

    start = 0

    for styled in generate_styles(line):
        texts.append(styled.text)

        end = start + len(styled.text)

        style = SpannedStyle(
            **styled.style.dict(),
            start=start,
            end=end,
        )

        start = end

        styles.append(style)

    return "".join(texts), styles


def generate_styles(line: LTTextLineHorizontal) -> Iterable[StyledText]:

    styled_text = None

    for char in line:

        if isinstance(char, LTAnno) or SPACE_PATTERN.match(char._text):
            if styled_text is not None:
                styled_text.add_space()
            continue

        styled_char = StyledText.from_char(char)

        if styled_text is None:
            styled_text = styled_char

        elif styled_text.style == styled_char.style:
            styled_text += styled_char

        else:
            yield styled_text
            styled_text = styled_char

    styled_text.strip()
    yield styled_text


def create_words(line: LTTextLineHorizontal) -> List[StyledText]:
    """
    Generate a list of words from a PDFMiner line object.

    Parameters
    ----------
    line : LTTextLineHorizontal
        Line abstraction in PDFMiner

    Returns
    -------
    List[Word]
        List of `Word` objects.
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
                    StyledText(
                        text=word,
                        style=Style.from_char(first_char),
                        x0=min(first_char.x0, char.x0),
                        y0=min(first_char.y0, char.y0),
                        x1=max(first_char.y0, char.x1),
                        y1=max(first_char.y1, char.y1),
                    )
                )
                word = ""
        else:
            word += char._text

    return words


def group_words(words: List[StyledText]) -> Iterable[StyledText]:

    if words:
        styled = words[0]

        for word in words[1:]:
            if styled.style == word.style:
                styled /= word

            else:
                yield styled
                styled = word

        yield styled


def words2style(words: List[StyledText]) -> str:
    """
    Transforms a list of `Word`s into a textual representation.

    Parameters
    ----------
    words : List[StyledText]
        List of `StyledText`s, containing word-by-word text and style.

    Returns
    -------
    str
        Styled textual representation, in the form
        `<s font=Font size=10.0 style=Normal>text</s>`
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
    Transform a PDFMiner line (`LTTextLineHorizontal`) into a style-formatted string.

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
    Transform a PDFMiner line (`LTTextLineHorizontal`) into a style-formatted string.
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
