import re
from typing import Iterable, List, Tuple

from pdfminer.layout import LTTextLineHorizontal

from .models import SpannedStyle, StyledText

SPACE_PATTERN = re.compile(r"\s")


def extract_style(
    line: LTTextLineHorizontal,
    width: float,
    height: float,
) -> Tuple[str, List[SpannedStyle]]:

    styles = []
    texts = []

    start = 0

    for styled in generate_styles(line, width=width, height=height):
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


def generate_styles(
    line: LTTextLineHorizontal,
    width: float,
    height: float,
) -> Iterable[StyledText]:

    styled_text = None

    for char in line:

        if SPACE_PATTERN.match(char._text):
            if styled_text is not None:
                styled_text.add_space()
            continue

        styled_char = StyledText.from_char(char, width=width, height=height)

        if styled_text is None:
            styled_text = styled_char

        elif styled_text.style == styled_char.style:
            styled_text += styled_char

        else:
            yield styled_text
            styled_text = styled_char

    styled_text.rstrip()
    yield styled_text
