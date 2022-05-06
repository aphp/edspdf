import re

from pytest import fixture


@fixture
def lines(pdf, extractor):
    return extractor(pdf)


def test_layout(lines):

    assert lines.bloc.nunique() >= 13


def test_italics(lines):

    line = lines.loc[0]
    text = line.styled_text

    style = re.search(r"<s.+? style=(\w+)[\s>]", text).group(1)
    assert style == "Italic"


def test_bold(lines):

    line = lines.loc[11]
    text = line.styled_text

    style = re.search(r"<s.+? style=(\w+)[\s>]", text).group(1)
    assert style == "Bold"


def test_sizes(lines):

    t1, t2 = lines.loc[0].styled_text, lines.loc[17].styled_text

    s1 = re.search(r"<s.+? size=(.+?)[\s>]", t1).group(1)
    s2 = re.search(r"<s.+? size=(.+?)[\s>]", t2).group(1)

    assert float(s1) < float(s2)


def test_equality(lines):

    text = lines.styled_text.str.replace(r"<s.+?>(.+?)</s>", r"\1", regex=True)
    assert (text == lines.text).all()
