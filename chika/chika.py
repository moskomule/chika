from __future__ import annotations

import argparse
import dataclasses
import json
import types
import typing
from collections import defaultdict
from enum import Enum
from functools import wraps
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import yaml

__all__ = ["ChikaArgumentParser",
           # space functions
           "with_help", "choices", "sequence", "required",
           # config
           "ChikaConfig",
           # decorators
           "config",
           "main"]

# fileio
JSON_SUFFIXES = {".json"}
YAML_SUFFIXES = {".yaml", ".yml"}


def is_supported_filetype(file: str
                          ) -> bool:
    return Path(file).suffix in (JSON_SUFFIXES | YAML_SUFFIXES)


def load_from_file(file: str
                   ) -> Dict[str, Any]:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"{file} not found")
    if not is_supported_filetype(file):
        raise RuntimeError(f"Unsupported file type with a suffix of {file.suffix}")

    with file.open() as f:
        if file.suffix in JSON_SUFFIXES:
            return json.load(f)
        elif file.suffix in YAML_SUFFIXES:
            return yaml.safe_load(f)


def save_as_file(file: str,
                 state_dict: Dict[str, Any]
                 ) -> None:
    file = Path(file)
    if not is_supported_filetype(file):
        raise RuntimeError(f"Unsupported file type with a suffix of {file.suffix}")
    file.parent.mkdir(exist_ok=True, parents=True)
    with file.open("w") as f:
        if file.suffix in JSON_SUFFIXES:
            json.dump(state_dict, f)
        elif file.suffix in YAML_SUFFIXES:
            yaml.safe_dump(state_dict, f)


# mark default values
class _DefaultUntouched(object):
    # mark as untouched default
    def __init__(self,
                 value: Any):
        self.value = value

    def __cal__(self):
        return self.value

    def __repr__(self):
        return self.value.__repr__()


# argument parser
class ChikaArgumentParser(argparse.ArgumentParser):
    """ This subclass of argparser that generates arguments from dataclasses.
     Inspired from huggingface transformers.hf_argparser.

     Args:
         dataclass_type: The top-level config class
         kwargs: kwargs for ArgumentParser
    """

    def __init__(self,
                 dataclass_type: Type[ChikaConfig],
                 **kwargs
                 ) -> None:
        if kwargs.get("formatter_class") is None:
            # help will show default values
            kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter
        if kwargs.get("allow_abbrev") is None:
            kwargs["allow_abbrev"] = False
        super().__init__(**kwargs)
        self.dataclass_type = dataclass_type
        self.add_dataclass_arguments(self.dataclass_type)

    def add_dataclass_arguments(self,
                                dtype: Type[ChikaConfig] or ChikaConfig,
                                prefix: Optional[str] = None,
                                nest_level: int = 0
                                ) -> None:
        name_to_type = typing.get_type_hints(dtype)
        for field in dataclasses.fields(dtype):
            # field is dataclass field and has following properties
            # name:
            # type:
            # default: default value (right hans side)
            # metadata: dict. this is used by chika's functions
            # default_factory, init, repr, compare, hash

            # __annotation__ changes type hint to str, so field.type might be str
            # field_type is actual type hint
            field_type = name_to_type[field.name]
            # if dtype is parent, --config
            # if dtype is child, --main.subconfig
            field_name = f"--{field.name}" if prefix is None else f"--{prefix}.{field.name}"
            kwargs: Dict[str, Any] = field.metadata.copy()
            if kwargs.get("help") is None:
                # to show default values
                kwargs["help"] = " "

            # remove Optional. Optional is Union[..., NoneType]
            # typing.get_args, typing.get_origin may make it easy
            type_string = str(field.type)
            for prim_type in (int, float, str):
                # Optional[List[int]] -> List[int]
                if type_string == f"typing.Union[List[{prim_type}], NoneType]":
                    field_type = List[prim_type]
                # Optional[int] -> int
                if type_string == f"typing.Union[{prim_type.__name__}, NoneType]":
                    field_type = prim_type

            if isinstance(field_type, type) and issubclass(field_type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field_type
                if kwargs.get("required") is None:
                    kwargs["default"] = field.default

            # foo: bool = True -> --foo makes foo False
            elif field_type is bool or field_type is Optional[bool]:
                kwargs["action"] = "store_false" if field.default is True else "store_true"

            elif self._is_type_list_or_tuple(field_type):
                if kwargs.get("nargs") is None:
                    kwargs["nargs"] = "+"
                kwargs["type"] = field_type

            # for ChikaConfig
            elif dataclasses.is_dataclass(field_type):
                if nest_level > 0:
                    raise NotImplementedError("The depth of config is expected to be at most 2")

                kwargs["help"] = f"load {{yaml,yml,json}} file for {field.name} if necessary"
                # argparse.SUPPRESS causes no attribute to be added if the command-line argument was not present
                if field.default is dataclasses.MISSING:
                    kwargs["default"] = argparse.SUPPRESS
                else:
                    kwargs["default"] = field.default
                self.add_argument(field_name, **kwargs)

                if field.default is dataclasses.MISSING:
                    self.add_dataclass_arguments(field_type, prefix=field.name, nest_level=nest_level + 1)
                else:
                    raise NotImplementedError("dataclass as default value is not supported")
                continue

            # for int, float, str
            else:
                kwargs["type"] = field_type
                # value is not missing
                # if nest_level > 0:
                #     kwargs["default"] = argparse.SUPPRESS
                if field.default is dataclasses.MISSING and kwargs.get("default") is None:
                    kwargs["required"] = True
                    kwargs["help"] += " (required)"
                elif field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default

            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclass(self,
                                  args: Optional[List[str]] = None,
                                  ) -> [ChikaConfig, Any]:
        print(self.print_help())
        namespace, remaining_args = self.parse_known_args(args=args)
        name_to_type = typing.get_type_hints(self.dataclass_type)
        unflatten_dict = defaultdict(dict)
        for k, v in vars(namespace).items():
            print(k, v)
            # ChikaConfig
            if dataclasses.is_dataclass(name_to_type.get(k)):
                if isinstance(v, str) and is_supported_filetype(v):
                    unflatten_dict[k] = load_from_file(v)
                else:
                    raise TypeError(f"{k}={v}")
            elif "." in k:
                # ChikaConfig's child
                cls_name, cls_field = k.split(".", 1)
                unflatten_dict[cls_name][cls_field] = v
            else:
                unflatten_dict[k] = v

        dclass = self.dataclass_type.from_dict(unflatten_dict)
        return dclass, remaining_args

    @staticmethod
    def _is_type_list_or_tuple(type: Type):
        origin = typing.get_origin(type)
        return origin is not None and issubclass(origin, (List, Tuple))


# configs

def with_help(default: Any,
              help: Optional[str] = None
              ) -> dataclasses.Field:
    """ Add help to ChikaConfig, which is used in ArgumentParser.

    Args:
        default: default value
        help: help message

    Returns: default value with a help message

    """

    meta = {'default': default}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def choices(*values: Any,
            help: Optional[str] = None
            ) -> dataclasses.Field:
    """ Add choices to ChikaConfig, which is used in ArgumentParser. The first value is used as the default value.

    Args:
        *values: candidate values to be chosen
        help: help message

    Returns: default value with candidates and a help message

    """
    values = list(values)
    meta = {'choices': values, 'default': values[0]}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def sequence(*values: Any,
             size: Optional[int] = None,
             help: Optional[str] = None
             ) -> dataclasses.Field:
    """ Add a default value of list, which is invalid in dataclass.

    Args:
        *values:
        size: size of sequence. If specified, the length is fixed, and if violated, ValueError will be raised.
        help: help message

    Returns: sequence with a help message

    """
    meta = {'default': list(values)}
    if size is not None:
        meta['nargs'] = size
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def required(*, help: Optional[str] = None
             ) -> dataclasses.Field:
    """ Add a missing value with a help message. This value must be specified later. ::

    Args:
        help: help message

    Returns: help message.

    """

    meta = {'required': True}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def bounded(default: Any,
            _from: Number,
            _to: Number,
            *,
            help: Optional[str] = None
            ) -> dataclasses.Field:
    # use metaclass[type]
    pass


@dataclasses.dataclass
class ChikaConfig:
    # mixin

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls,
                  state_dict: Dict[str, Any]
                  ) -> ChikaConfig:
        _state_dict = {}
        field_type = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            name = field.name
            if dataclasses.is_dataclass(field_type[name]):
                # ChikaConfig
                _state_dict[name] = field_type[name].from_dict(state_dict[name])
            else:
                _state_dict[name] = state_dict[name]
        return cls(**_state_dict)

    def __repr__(self):
        # todo: better looking
        return super().__repr__()


# config decorator
def config(cls=None
           ) -> ChikaConfig:
    """ A wrapper to make ChikaConfig ::

    @config
    class DataConfig:
        name: cifar10

    Args:
        cls: wrapped class. Class name is expected to be FooConfig, and foo is used as key if this class is used as a child

    Returns: config in dataclass and ChikaConfig

    """

    def wrap(cls):
        # create cls whose baseclass is ChikaConfig
        cls = types.new_class(cls.__name__, (ChikaConfig,), {}, lambda ns: ns.update(cls.__dict__))
        # make cls to dataclass
        return dataclasses.dataclass(cls)

    return wrap if cls is None else wrap(cls)


# entry point
def main(cfg_cls: Type[ChikaConfig],
         strict: bool = False
         ) -> Callable:
    def _decorator(func: Callable):
        @wraps(func)
        def _wrapper():
            _config, remaining_args = ChikaArgumentParser(cfg_cls).parse_args_into_dataclass()
            if strict and len(remaining_args) > 0:
                raise ValueError(f"Some specified arguments are not used by ChikaArgumentParser: {remaining_args}")
            return func(_config)

        return _wrapper

    return _decorator
