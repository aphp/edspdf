from edspdf.utils.collections import (
    flatten_dict,
    get_deep_attr,
    nest_dict,
    set_deep_attr,
)


def test_nest_dict():
    assert nest_dict({"a/b/c": 4, "a/b/d": "ok", "a/x": {"key": "value"}}) == {
        "a": {
            "b": {"c": 4, "d": "ok"},
            "x": {"key": "value"},
        }
    }
    assert nest_dict({}) == {}


def test_flatten_dict():
    assert flatten_dict(
        {
            "a": {
                "b": {"c": 4, "d": "ok"},
                "x": {"key": "value"},
            },
            "empty": {},
        }
    ) == (
        {
            "a/b/c": 4,
            "a/b/d": "ok",
            "a/x/key": "value",
        }
    )


class Point:
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.data = data


def test_deep_attr():
    obj = [Point(2, 3, {"my": ({"attr": 4},)})]
    assert get_deep_attr(obj, "0.data.my.0.attr") == 4

    set_deep_attr(obj, "0.data.my.0.attr", 5)

    assert get_deep_attr(obj, "0.data.my.0.attr") == 5

    set_deep_attr(obj, "0.data.my.0", 5)

    assert get_deep_attr(obj, "0.data.my") == (5,)

    set_deep_attr(obj, "0.new_attr", "ok")

    assert get_deep_attr(obj, "0.new_attr") == "ok"
