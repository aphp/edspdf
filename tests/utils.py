import attr
from pytest import approx as pytest_approx

from edspdf.structures import BaseModel


def is_primitive(x):
    return x is None or type(x) in (int, float, str, bool)


pytest_plugins = ["helpers_namespace"]


def nested_approx(A, B, abs=1e-6, rel=1e-6, enforce_same_type=False):
    if enforce_same_type and type(A) != type(B) and not is_primitive(A):
        # I use `not is_primitive(A)` to enforce the same type only for data structures
        return False
    if isinstance(A, BaseModel):
        names = [field.name for field in attr.fields(type(A)) if field.eq]
        return type(A) == type(B) and nested_approx(
            A.dict(filter=lambda a, v: a.name in names),
            B.dict(filter=lambda a, v: a.name in names),
            abs=abs,
            rel=rel,
        )

    elif isinstance(A, set) or isinstance(B, set):
        # if any of the data structures is a set, convert both of them to a sorted
        # list, but return False if the length has changed
        len_A, len_B = len(A), len(B)
        A, B = sorted(A), sorted(B)
        if len_A != len(A) or len_B != len(B):
            return False

        for i in range(len(A)):
            if not nested_approx(A[i], B[i], abs, rel):
                return False

        return True
    elif isinstance(A, dict) and isinstance(B, dict):
        for k in A.keys():
            if not nested_approx(A[k], B[k], abs, rel):
                return False

        return True
    elif (isinstance(A, list) or isinstance(A, tuple)) and (
        isinstance(B, list) or isinstance(B, tuple)
    ):
        for i in range(len(A)):
            if not nested_approx(A[i], B[i], abs, rel):
                return False

        return True
    else:
        try:
            assert A == pytest_approx(B, rel=rel, abs=abs)
            is_approx_equal = A == pytest_approx(B, rel=rel, abs=abs)
        except (AssertionError, TypeError):
            is_approx_equal = False

        return is_approx_equal
