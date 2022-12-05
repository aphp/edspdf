import collections
import re
from ast import literal_eval
from configparser import ConfigParser
from copy import deepcopy
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pydantic
import srsly
import typer
from pydantic import ValidationError
from pydantic.error_wrappers import ErrorWrapper

from edspdf.utils.collections import dedup


def validate_arguments(func: Optional[Callable] = None, *, config: Dict = None) -> Any:
    """
    Decorator to validate the arguments passed to a function.
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
                    return _func(**d, **var_kwargs)

                yield validate

            @wraps(vd.raw_function)
            def wrapper_function(*args: Any, **kwargs: Any) -> Any:
                return vd.call(*args, **kwargs)

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
                return vd.call(*args, **kwargs)

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
    def from_str(cls, s: str, resolve: bool = True) -> "Config":
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
    def from_disk(cls, path: Union[str, Path], resolve: bool = True) -> "Config":
        s = Path(path).read_text(path)
        return cls.from_str(s, resolve=resolve)

    def to_disk(self, path: Union[str, Path]):
        s = self.to_str()
        Path(path).write_text(s)

    def to_str(self):
        refs = {}

        additional_sections = {}

        def prepare(o, path=()):
            if o is None or isinstance(o, (str, int, float, bool, tuple, list)):
                return srsly.json_dumps(o)
            if isinstance(o, Reference):
                return repr(o)
            if isinstance(o, collections.Mapping):

                if isinstance(o, Config) and o.__path__ is not None:
                    res = {k: prepare(v, (*o.__path__, k)) for k, v in o.items()}
                    current = additional_sections
                    for part in o.__path__[:-1]:
                        current = current.setdefault(part, Config())
                    current[o.__path__[-1]] = res
                    return "${" + join_path(o.__path__) + "}"
                else:
                    return {k: prepare(v, (*path, k)) for k, v in o.items()}
            try:
                cfg = o.cfg
            except AttributeError:
                pass
            else:
                if id(o) in refs:
                    return refs[id(o)]
                else:
                    refs[id(o)] = "${" + join_path(path) + "}"
                return prepare(cfg, path)
            raise TypeError(f"Cannot dump {o!r}")

        prepared = flatten_sections(prepare(self))
        prepared.update(flatten_sections(additional_sections))

        parser = ConfigParser()
        parser.optionxform = str
        for section_name, section in prepared.items():
            parser.add_section(section_name)
            parser[section_name].update(section)
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
                return fn(**params)
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
