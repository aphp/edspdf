import copy
import itertools
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, TypeVar

It = TypeVar("It", bound=Iterable)
T = TypeVar("T")


def flatten_dict(root: Dict[str, Any], depth=-1) -> Dict[str, Any]:
    res = {}

    def rec(d, path, current_depth):
        for k, v in d.items():
            if isinstance(v, dict) and current_depth != depth:
                rec(v, path + "/" + k if path is not None else k, current_depth + 1)
            else:
                res[path + "/" + k if path is not None else k] = v

    rec(root, None, 0)
    return res


def nest_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    res = {}

    for key, values in flat.items():
        for path in key.split("|"):
            current = res
            parts = path.split("/")
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = values

    return res


def ld_to_dl(ld: Iterable[Mapping[str, T]]) -> Dict[str, List[T]]:
    ld = list(ld)
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def dl_to_ld(dl: Mapping[str, Sequence[T]]) -> List[Dict[str, T]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


def flatten(seq: Sequence[Sequence["T"]]) -> List["T"]:
    return list(itertools.chain.from_iterable(seq))


FLATTEN_TEMPLATE = """\
def flatten(root):
    res={}
    return res
"""


def discover_scheme(obj):
    keys = defaultdict(lambda: [])

    def rec(current, path):
        if not isinstance(current, dict):
            keys[id(current)].append(path)
            return
        for key, value in current.items():
            rec(value, (*path, key))

    rec(obj, ())

    code = FLATTEN_TEMPLATE.format(
        "{"
        + "\n".join(
            "'{}': root{},".format(
                "|".join(map("/".join, key_list)),
                "".join(f"['{k}']" for k in key_list[0]),
            )
            for key_list in keys.values()
        )
        + "}"
    )
    return code


class batch_compress_dict:
    __slots__ = ("flatten", "seq")

    def __init__(self, seq: Iterable[Dict[str, Any]]):
        self.seq = seq
        self.flatten = None

    def __iter__(self):
        return batch_compress_dict(iter(self.seq))

    def __getstate__(self):
        return {"seq": self.seq}

    def __setstate__(self, state):
        self.seq = state["seq"]
        self.flatten = None

    def __next__(self) -> Dict[str, List]:
        exec_result = {}

        item = next(self.seq)
        if self.flatten is None:
            exec(discover_scheme(item), {}, exec_result)
            self.flatten = exec_result["flatten"]
        return self.flatten(item)


def decompress_dict(seq):
    obj = ld_to_dl(seq) if isinstance(seq, Sequence) else seq
    res = {}
    for key, value in obj.items():
        for path in key.split("|"):
            current = res
            parts = path.split("/")
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
    return res


class batchify(Iterable[List[T]]):
    def __init__(self, iterable: Iterable[T], batch_size: int):
        self.iterable = iter(iterable)
        self.batch_size = batch_size
        try:
            self.length = math.ceil(len(iterable) / batch_size)
        except (AttributeError, TypeError):
            pass

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        batch = list(itertools.islice(self.iterable, self.batch_size))
        if len(batch) == 0:
            raise StopIteration()
        return batch


def get_attr_item(base, attr):
    try:
        return base[attr]
    except (KeyError, TypeError):
        return getattr(base, attr)


def split_names(names):
    _names = []
    for part in names.split("."):
        try:
            _names.append(int(part))
        except ValueError:
            _names.append(part)
    return _names


def get_deep_attr(base, names):
    if isinstance(names, str):
        names = split_names(names)
    if len(names) == 0:
        return base
    [current, *remaining] = names
    return get_deep_attr(get_attr_item(base, current), remaining)


def set_attr_item(base, attr, val):
    try:
        base[attr] = val
    except (KeyError, TypeError):
        setattr(base, attr, val)
    return base


def set_deep_attr(base, names, val):
    if isinstance(names, str):
        names = split_names(names)
    if len(names) == 0:
        return val
    if len(names) == 1:
        if isinstance(base, (dict, list)):
            base[names[0]] = val
        else:
            setattr(base, names[0], val)
    [current, *remaining] = names
    attr = base[current] if isinstance(base, (dict, list)) else getattr(base, current)
    try:
        set_deep_attr(attr, remaining, val)
    except TypeError:
        new_attr = list(attr)
        set_deep_attr(new_attr, remaining, val)
        return set_attr_item(base, current, tuple(new_attr))
    return base


def list_factorize(values, reference_values=None, freeze_reference=None):
    if freeze_reference is None:
        freeze_reference = reference_values is not None
    reference_values = (
        dict(zip(list(reference_values), range(len(reference_values))))
        if reference_values is not None
        else {}
    )

    def rec(obj):
        if hasattr(obj, "__len__") and not isinstance(obj, str):
            return list(item for item in (rec(item) for item in obj) if item != -1)
        return (
            reference_values.get(obj, -1)
            if freeze_reference
            else reference_values.setdefault(obj, len(reference_values))
        )

    return rec(values), list(reference_values.keys())


class multi_tee:
    """
    Makes copies of an iterable such that every iteration over it
    starts from 0. If the iterable is a sequence (list, tuple), just returns
    it since every iter() over the object restart from the beginning
    """

    def __new__(cls, iterable):
        if isinstance(iterable, Sequence):
            return iterable
        return super().__new__(cls)

    def __init__(self, iterable):
        self.main, self.copy = itertools.tee(iterable)

    def __iter__(self):
        if self.copy is not None:
            it = self.copy
            self.copy = None
            return it
        return copy.copy(self.main)


class FrozenDict(dict):
    """
    Copied from `spacy.util.SimpleFrozenDict` to ensure compatibility.


    """

    def __init__(self, *args, error: str = None, **kwargs) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """

        if error is None:
            error = "Cannot mutate this frozen dict"
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):  # pragma: no cover
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):  # pragma: no cover
        raise NotImplementedError(self.error)

    def update(self, other):  # pragma: no cover
        raise NotImplementedError(self.error)


class FrozenList(list):
    """
    Copied from `spacy.util.SimpleFrozenDict` to ensure compatibility
    """

    def __init__(self, *args, error: str = None) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """

        if error is None:
            error = "Cannot mutate this frozen list"
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(self.error)
