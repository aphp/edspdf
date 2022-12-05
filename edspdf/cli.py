import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from pydantic import ValidationError
from typer import Context, Typer
from typer.core import TyperCommand
from typer.models import CommandFunctionType, Default

from .config import Config, parse_overrides, validate_arguments
from .utils.random import set_seed


class Cli(Typer):
    """
    Custom Typer object that:
    - validates a command parameters before executing it
    - accepts a configuration file describing the parameters
    - automatically instantiates parameters given a dictionary when type hinted
    """

    def command(  # noqa
        self,
        name,
        *,
        cls: Optional[Type[TyperCommand]] = None,
        context_settings: Optional[Dict[Any, Any]] = None,
        help: Optional[str] = None,
        epilog: Optional[str] = None,
        short_help: Optional[str] = None,
        options_metavar: str = "[OPTIONS]",
        add_help_option: bool = True,
        no_args_is_help: bool = False,
        hidden: bool = False,
        deprecated: bool = False,
        # Rich settings
        rich_help_panel: Union[str, None] = Default(None),
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        typer_command = super().command(
            name=name,
            cls=cls,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            no_args_is_help=no_args_is_help,
            hidden=hidden,
            deprecated=deprecated,
            rich_help_panel=rich_help_panel,
            context_settings={
                **(context_settings or {}),
                "ignore_unknown_options": True,
                "allow_extra_args": True,
            },
        )

        def wrapper(fn):
            validated = validate_arguments(fn)

            @typer_command
            def command(ctx: Context, config: Optional[Path] = None):
                if config is not None:
                    config = Config.from_str(Path(config).read_text(), resolve=False)
                else:
                    config = Config({name: {}})
                for k, v in parse_overrides(ctx.args).items():
                    if "." not in k:
                        parts = (name, k)
                    else:
                        parts = k.split(".")
                        if (
                            parts[0] in validated.model.__fields__
                            and not parts[0] in config
                        ):
                            parts = (name, *parts)
                    current = config
                    if parts[0] not in current:
                        raise Exception(
                            f"{k} does not match any existing section in config"
                        )
                    for part in parts[:-1]:
                        current = current.setdefault(part, Config())
                    current[parts[-1]] = v
                try:
                    default_seed = validated.model.__fields__.get("seed")
                    seed = config.get(name, {}).get("seed", default_seed)
                    if seed is not None:
                        set_seed(seed)

                    config = config.resolve()

                    return validated(**config.get(name, {}))
                except ValidationError as e:
                    print("\x1b[{}m{}\x1b[0m".format("38;5;1", "Validation error"))
                    print(str(e))
                    sys.exit(1)

            return validated

        return wrapper
