import collections
import re
from ast import literal_eval
from configparser import ConfigParser
from copy import deepcopy
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from weakref import WeakKeyDictionary

import pydantic
import srsly
import typer
from pydantic import ValidationError
from pydantic.decorator import (
    ALT_V_ARGS,
    ALT_V_KWARGS,
    V_DUPLICATE_KWARGS,
    V_POSITIONAL_ONLY_NAME,
)
from pydantic.error_wrappers import ErrorWrapper

from edspdf.utils.collections import dedup

RESOLVED = WeakKeyDictionary()


def resolve_non_dict(model: pydantic.BaseModel, values: Dict[str, Any]):
    """
    Iterates over the model fields and try to resolve the matching values
    if they are not type hinted as dictionaries.
    """
    values = dict(values)
    for field in model.__fields__.values():
        if field.name not in values:
            continue
        if field.shape not in pydantic.fields.MAPPING_LIKE_SHAPES and isinstance(
            values[field.name], dict
        ):
            values[field.name] = Config(values[field.name]).resolve(deep=False)
    return values


def validate_arguments(
    func: Optional[Callable] = None,
    *,
    config: Dict = None,
    save_params: Optional[Dict] = None,
) -> Any:
    """
    Decorator to validate the arguments passed to a function.

    Parameters
    ----------
    func: Callable
        The function or class to call
    config: Dict
        The validation configuration object
    save_params: bool
        Should we save the function parameters

    Returns
    -------
    Any
    """
    if config is None:
        config = {}
    config = {**config, "arbitrary_types_allowed": True}

    def validate(_func: Callable) -> Callable:

        if isinstance(_func, type):

            if hasattr(_func, "raw_function"):
                vd = pydantic.decorator.ValidatedFunction(_func.raw_function, config)
            else:
                vd = pydantic.decorator.ValidatedFunction(_func.__init__, config)
            vd.model.__name__ = _func.__name__
            vd.model.__fields__["self"].required = False

            def __get_validators__():
                def validate(value):
                    params = value
                    if isinstance(value, dict):
                        value = Config(value).resolve(deep=False)

                    if not isinstance(value, dict):
                        return value

                    m = vd.init_model_instance(**value)
                    d = {
                        k: v
                        for k, v in m._iter()
                        if k in m.__fields_set__ or m.__fields__[k].default_factory
                    }
                    var_kwargs = d.pop(vd.v_kwargs_name, {})
                    resolved = _func(**d, **var_kwargs)

                    if save_params is not None:
                        RESOLVED[resolved] = {**save_params, **params}

                    return resolved

                yield validate

            @wraps(vd.raw_function)
            def wrapper_function(*args: Any, **kwargs: Any) -> Any:
                values = vd.build_values(args, kwargs)
                if save_params is not None:
                    if set(values.keys()) & {
                        ALT_V_ARGS,
                        ALT_V_KWARGS,
                        V_POSITIONAL_ONLY_NAME,
                        V_DUPLICATE_KWARGS,
                        "args",
                        "kwargs",
                    }:
                        print("VALUES", values.keys(), values["kwargs"])
                        raise Exception(
                            f"{func} must not have positional only args, "
                            f"kwargs or duplicated kwargs"
                        )
                    params = dict(values)
                    resolved = params.pop("self")
                    RESOLVED[resolved] = {**save_params, **params}
                return vd.execute(vd.model(**resolve_non_dict(vd.model, values)))

            _func.vd = vd  # type: ignore
            # _func.validate = vd.init_model_instance  # type: ignore
            _func.__get_validators__ = __get_validators__  # type: ignore
            _func.raw_function = vd.raw_function  # type: ignore
            _func.model = vd.model  # type: ignore
            _func.__init__ = wrapper_function
            return _func

        else:
            vd = pydantic.decorator.ValidatedFunction(_func, config)

            @wraps(_func)
            def wrapper_function(*args: Any, **kwargs: Any) -> Any:
                values = vd.build_values(args, kwargs)
                resolved = vd.execute(vd.model(**resolve_non_dict(vd.model, values)))
                if save_params is not None:
                    if set(values.keys()) & {
                        ALT_V_ARGS,
                        ALT_V_KWARGS,
                        V_POSITIONAL_ONLY_NAME,
                        V_DUPLICATE_KWARGS,
                        "args",
                        "kwargs",
                    }:
                        raise Exception(
                            f"{func} must not have positional only args, "
                            f"kwargs or duplicated kwargs"
                        )
                    RESOLVED[resolved] = {**save_params, **values}
                return resolved

            wrapper_function.vd = vd  # type: ignore
            wrapper_function.validate = vd.init_model_instance  # type: ignore
            wrapper_function.raw_function = vd.raw_function  # type: ignore
            wrapper_function.model = vd.model  # type: ignore
            return wrapper_function

    if func:
        return validate(func)
    else:
        return validate


def parse_overrides(args: List[str]) -> Dict[str, Any]:
    result = {}
    while args:
        opt = args.pop(0)
        err = f"Invalid config override '{opt}'"
        if opt.startswith("--"):  # new argument
            opt = opt.replace("--", "")
            # if "." not in opt:
            #     typer.secho(
            #         f"{err}: can't override top-level sections", fg=typer.colors.RED
            #     )
            #     exit(1)
            if "=" in opt:  # we have --opt=value
                opt, value = opt.split("=", 1)
            else:
                if not args or args[0].startswith("--"):  # flag with no value
                    value = "true"
                else:
                    value = args.pop(0)
            opt = opt.replace("-", "_")
            result[opt] = parse_override(value)
        else:
            typer.secho(
                f"{err}: can't override top-level sections", fg=typer.colors.RED
            )
            exit(1)
    return result


def parse_override(value: Any) -> Any:
    # Just like we do in the config, we're calling json.loads on the
    # values. But since they come from the CLI, it'd be unintuitive to
    # explicitly mark strings with escaped quotes. So we're working
    # around that here by falling back to a string if parsing fails.
    try:
        return srsly.json_loads(value)
    except ValueError:
        return str(value)


def config_literal_eval(s):
    if s.startswith("${") and s.endswith("}"):
        return Reference(s[2:-1])
    try:
        return literal_eval(s)
    except ValueError:
        try:
            return srsly.json_loads(s)
        except ValueError:
            return s


def config_literal_dump(v: Any):
    if isinstance(v, Reference):
        return "${" + v.value + "}"
    if isinstance(v, str):
        if config_literal_eval(str(v)) == v:
            return str(v)
        return srsly.json_dumps(v)
    return srsly.json_dumps(v)


def flatten_sections(root: Dict[str, Any]) -> Dict[str, Any]:
    res = collections.defaultdict(lambda: {})

    def rec(d, path):
        res.setdefault(join_path(path), {})
        section = {}
        for k, v in d.items():
            if isinstance(v, dict):
                rec(v, (*path, k))
            else:
                section[k] = v
        res[join_path(path)].update(section)

    rec(root, ())
    res.pop("", None)
    return dict(res)


def join_path(path):
    return ".".join(repr(x) if "." in x else x for x in path)


def split_path(path):
    offset = 0
    result = []
    for match in re.finditer(r"(?:'([^']*)'|\"([^\"]*)\"|([^.]*))(?:[.]|$)", str(path)):
        assert match.start() == offset, f"Malformed path: {path!r} in config"
        offset = match.end()
        result.append(next((g for g in match.groups() if g is not None)))
        if offset == len(path):
            break
    return result


class Reference:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Reference) and self.value == other.value

    def __str__(self):
        return self.value

    def __len__(self):
        return len(self.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return "${{{}}}".format(self.value)


class MissingReference(Exception):
    def __init__(self, references: List[Reference]):
        self.references = references
        super().__init__()

    def __str__(self):
        return "Could not interpolate the following references: {}".format(
            ", ".join("${{{}}}".format(r) for r in self.references)
        )


class Config(dict):
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            assert len(kwargs) == 0
            kwargs = args[0]
        path = kwargs.pop("__path__", None)
        kwargs = {
            key: Config(value)
            if isinstance(value, dict) and not isinstance(value, Config)
            else value
            for key, value in kwargs.items()
        }
        super().__init__(**kwargs)
        self.__path__ = path

    @classmethod
    def from_str(cls, s: str, resolve: bool = False) -> "Config":
        parser = ConfigParser()
        parser.optionxform = str
        parser.read_string(s)

        config = Config()

        for section in parser.sections():
            parts = split_path(section)
            current = config
            for part in parts:
                if part not in current:
                    current[part] = current = Config()
                else:
                    current = current[part]

            current.clear()
            current.update(
                {k: config_literal_eval(v) for k, v in parser.items(section)}
            )

        if resolve:
            return config.resolve()

        return config

    @classmethod
    def from_disk(cls, path: Union[str, Path], resolve: bool = False) -> "Config":
        s = Path(path).read_text()
        return cls.from_str(s, resolve=resolve)

    def to_disk(self, path: Union[str, Path]):
        s = self.to_str()
        Path(path).write_text(s)

    def serialize(self):
        """
        Try to convert non-serializable objects using the RESOLVED object
        back to their original catalogue + params form

        Returns
        -------
        Config
        """
        refs = {}

        def rec(o, path=()):
            if o is None or isinstance(
                o, (str, int, float, bool, tuple, list, Reference)
            ):
                return o
            if isinstance(o, collections.Mapping):
                serialized = {k: rec(v, (*path, k)) for k, v in o.items()}
                if isinstance(o, Config):
                    serialized = Config(serialized)
                    serialized.__path__ = o.__path__
                return serialized
            cfg = None
            try:
                cfg = o.cfg
            except AttributeError:
                try:
                    cfg = RESOLVED[o]
                except KeyError:
                    pass
            if cfg is not None:
                if id(o) in refs:
                    return refs[id(o)]
                else:
                    refs[id(o)] = Reference(join_path(path))
                return rec(cfg, path)
            raise TypeError(f"Cannot dump {o!r}")

        result = rec(self)
        return result

    def to_str(self):
        additional_sections = {}

        def rec(o, path=()):
            if isinstance(o, collections.Mapping):
                if isinstance(o, Config) and o.__path__ is not None:
                    res = {k: rec(v, (*o.__path__, k)) for k, v in o.items()}
                    current = additional_sections
                    for part in o.__path__[:-1]:
                        current = current.setdefault(part, Config())
                    current[o.__path__[-1]] = res
                    return Reference(join_path(o.__path__))
                else:
                    return {k: rec(v, (*path, k)) for k, v in o.items()}
            return o

        prepared = flatten_sections(rec(self.serialize()))
        prepared.update(flatten_sections(additional_sections))

        parser = ConfigParser()
        parser.optionxform = str
        for section_name, section in prepared.items():
            parser.add_section(section_name)
            parser[section_name].update(
                {k: config_literal_dump(v) for k, v in section.items()}
            )
        s = StringIO()
        parser.write(s)
        return s.getvalue()

    def resolve(self, _path=(), leaves=None, deep=True):
        from .registry import registry  # local import because circular deps

        copy = Config(**self)
        if leaves is None:
            leaves = {}
        missing = []
        items = [(k, v) for k, v in copy.items()] if deep else []
        last_count = len(leaves)
        while len(items):
            traced_missing_values = []
            for key, value in items:
                try:
                    if isinstance(value, Config):
                        if (*_path, key) not in leaves:
                            leaves[(*_path, key)] = value.resolve((*_path, key), leaves)
                        copy[key] = leaves[(*_path, key)]
                    elif isinstance(value, Reference):
                        try:
                            leaves[(*_path, key)] = leaves[tuple(split_path(value))]
                        except KeyError:
                            raise MissingReference([value])
                        else:
                            copy[key] = leaves[(*_path, key)]

                except MissingReference as e:
                    traced_missing_values.extend(e.references)
                    missing.append((key, value))
            if len(missing) > 0 and len(leaves) <= last_count:
                raise MissingReference(dedup(traced_missing_values))
            items = list(missing)
            last_count = len(leaves)
            missing = []

        registries = [
            (key, value, registry._catalogue[key[1:]])
            for key, value in copy.items()
            if key.startswith("@")
        ]
        assert len(registries) <= 1, (
            f"Cannot resolve using multiple " f"registries at {'.'.join(_path)}"
        )

        def patch_errors(errors: Union[Sequence[ErrorWrapper], ErrorWrapper]):
            if isinstance(errors, list):
                res = []
                for error in errors:
                    res.append(patch_errors(error))
                return res
            return ErrorWrapper(errors.exc, (*_path, *errors.loc_tuple()))

        if len(registries) == 1:
            params = dict(copy)
            params.pop(registries[0][0])
            fn = registries[0][2].get(registries[0][1])
            try:
                resolved = fn(**params)
                try:
                    resolved.cfg
                except Exception:
                    try:
                        RESOLVED[resolved] = self
                    except Exception:
                        print(f"Could not store original config for {resolved}")
                        pass

                return resolved
            except ValidationError as e:
                raise ValidationError(patch_errors(e.raw_errors), e.model)

        return copy

    def merge(
        self,
        *updates: Union[Dict[str, Any], "Config"],
        remove_extra: bool = False,
    ) -> "Config":
        """
        Deep merge two configs. Largely inspired from spaCy config merge function.

        Parameters
        ----------
        updates: Union[Config, Dict]
            Configs to update the original config
        remove_extra:
            If true, restricts update to keys that existed in the original config

        Returns
        -------
        The new config
        """

        def deep_set(current, path, val):
            try:
                path = split_path(path)
                for part in path[:-1]:
                    current = (
                        current[part] if remove_extra else current.setdefault(part, {})
                    )
            except KeyError:
                return
            if path[-1] not in current and remove_extra:
                return
            current[path[-1]] = val

        def rec(old, new):
            for key, new_val in list(new.items()):
                if "." in key:
                    deep_set(old, key, new_val)
                    continue

                if key not in old:
                    if remove_extra:
                        continue
                    else:
                        old[key] = new_val
                        continue

                old_val = old[key]
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    old_resolver = next((k for k in old_val if k.startswith("@")), None)
                    new_resolver = next((k for k in new_val if k.startswith("@")), None)
                    if (
                        new_resolver is not None
                        and old_resolver is not None
                        and (
                            old_resolver != new_resolver
                            or old_val.get(old_resolver) != new_val.get(new_resolver)
                        )
                    ):
                        old[key] = new_val
                    else:
                        rec(old[key], new_val)
                else:
                    old[key] = new_val
            return old

        config = deepcopy(self)
        for u in updates:
            u = deepcopy(u)
            rec(config, u)
        return Config(**config)
