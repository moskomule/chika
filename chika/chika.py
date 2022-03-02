from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import math
import os
import types
import typing
import uuid
import warnings
from collections import defaultdict
from datetime import datetime
from enum import Enum
from functools import wraps
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .utils import (DefaultUntouched, SUPPORTED_SUFFIXES, _container_to_type, _container_types, _get_git_hash,
                    _is_container_type, _enum_to_value,
                    _primitive_types,
                    _unpack_optional, is_supported_filetype, load_from_file, save_as_file)

__all__ = ["ChikaArgumentParser",
           # functions for ChikaConfig
           "with_help", "choices", "sequence", "required", "bounded",
           # config
           "ChikaConfig",
           # path
           "original_path", "resolve_original_path",
           # decorators
           "config",
           "main"]


# argument parser
class ChikaArgumentParser(argparse.ArgumentParser):
    """ This subclass of argparser that generates arguments from dataclasses.
     Inspired from huggingface transformers.hf_argparser.

     Args:
         dataclass_type: The top-level config class
         kwargs: kwargs for ArgumentParser
    """

    def __init__(self,
                 dataclass_type: Type[ChikaConfig] | ChikaConfig,
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
                                dtype: Type[ChikaConfig] | ChikaConfig,
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
            # remove Optional. Optional is Union[..., NoneType]
            field_type = _unpack_optional(field_type)

            # if dtype is parent, --config
            # if dtype is child, --main.subconfig
            field_name = f"--{field.name}" if prefix is None else f"--{prefix}.{field.name}"
            kwargs: Dict[str, Any] = field.metadata.copy()
            if kwargs.get("help") is None:
                # to show default values
                kwargs["help"] = " "

            if isinstance(field_type, type) and issubclass(field_type, Enum):
                kwargs["choices"] = [en.value for en in field_type]
                kwargs["type"] = field_type
                if kwargs.get("required") is None:
                    kwargs["default"] = field.default

            elif field_type is bool or field_type is Optional[bool]:
                # foo: bool = True -> --foo makes foo False
                kwargs["action"] = "store_false" if field.default is True else "store_true"

            elif _is_container_type(field_type):
                if kwargs.get("nargs") is None:
                    kwargs["nargs"] = "+"
                kwargs["type"] = typing.get_args(field_type)[0]

            elif dataclasses.is_dataclass(field_type):
                # for ChikaConfig
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

            else:
                # for int, float, str
                kwargs["type"] = field_type
                if field.default is dataclasses.MISSING and kwargs.get("default") is None:
                    if nest_level == 0:
                        kwargs["required"] = True
                    else:
                        # when nested, mark the value
                        kwargs["default"] = DefaultUntouched(None)
                    kwargs["help"] += " (required)"
                elif field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default if nest_level == 0 else DefaultUntouched(field.default)
                elif nest_level > 0 and kwargs.get("default") is not None:
                    kwargs["default"] = DefaultUntouched(kwargs["default"])

            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclass(self,
                                  args: Optional[List[str]] = None,
                                  ) -> [ChikaConfig, Any]:
        namespace, remaining_args = self.parse_known_args(args=args)
        name_to_type = typing.get_type_hints(self.dataclass_type)
        unflatten_dict = defaultdict(dict)
        # todo: more nested configs
        for k, v in vars(namespace).items():
            if dataclasses.is_dataclass(name_to_type.get(k)):
                # ChikaConfig
                if isinstance(v, str) and is_supported_filetype(v):
                    loaded = load_from_file(v)
                    for _k, _v in loaded.items():
                        if unflatten_dict[k].get(_k) is not None:
                            old = unflatten_dict[k][_k]
                            # if user changes value, use that value, otherwise load from file
                            unflatten_dict[k][_k] = _v if isinstance(old, DefaultUntouched) else old
                else:
                    raise RuntimeError(f"Unsupported filetype, config file must be one of {SUPPORTED_SUFFIXES}")
            elif "." in k:
                # ChikaConfig's child
                cls_name, cls_field = k.split(".", 1)
                unflatten_dict[cls_name][cls_field] = v
            else:
                unflatten_dict[k] = v

        def remove_default_untouched(d):
            _d = {}
            for k, v in d.items():
                if isinstance(v, DefaultUntouched):
                    _d[k] = v.value
                elif isinstance(v, dict):
                    _d[k] = remove_default_untouched(v)
                else:
                    _d[k] = v
            return _d

        dclass = self.dataclass_type.from_dict(remove_default_untouched(unflatten_dict))
        return dclass, remaining_args

    @staticmethod
    def _is_type_list_or_tuple(type: Type):
        origin = typing.get_origin(type)
        return origin is not None and issubclass(origin, (List, Tuple))


# configs

def with_help(default: Any, *,
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
    return dataclasses.field(default=default, metadata=meta)


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
    return dataclasses.field(default=values[0], metadata=meta)


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
    meta = {'default': list(values), 'nargs': "+" if size is None else size}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(default=tuple(values), metadata=meta)


def required(*, help: Optional[str] = None
             ) -> dataclasses.Field:
    """ Add a missing value. This value must be specified later. ::

    Args:
        help: help message

    Returns: help message.

    """

    meta = {'required': True}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(default=None, metadata=meta)


def bounded(default: Optional[Number] = None,
            _from: Optional[Number] = None,
            _to: Optional[Number] = None,
            *,
            help: Optional[str] = None
            ) -> dataclasses.Field:
    """ Bound an argument value in (_from, _to).

    Args:
        default: Default value
        _from: Lower bound value
        _to: Upper bound value
        help: help message

    Returns:

    """
    _from = -math.inf if _from is None else _from
    _to = math.inf if _to is None else _to
    if isinstance(default, Number) and not (_from <= default <= _to):
        raise ValueError("default value is out of range")

    def _impl(val: Number):
        if not isinstance(val, Number):
            raise ValueError(f"value needs to be number, but got {type(val)}")
        if not (_from <= val <= _to):
            raise ValueError(f"value is expected to be from {_from} to {_to}, but got {val} instead")
        return val

    meta = {"type": _impl}
    if default is not None:
        meta["default"] = default
    if help is not None:
        meta["help"] = help
    return dataclasses.field(default=default, metadata=meta)


@dataclasses.dataclass
class ChikaConfig:
    # mixin

    def to_dict(self):
        return _enum_to_value(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls,
                  state_dict: Dict[str, Any],
                  allow_missing: bool = False
                  ) -> ChikaConfig:
        _state_dict = {}
        field_type_hints = typing.get_type_hints(cls)
        for field in dataclasses.fields(cls):
            name = field.name
            if dataclasses.is_dataclass(field_type_hints[name]):
                # ChikaConfig
                _state_dict[name] = field_type_hints[name].from_dict(state_dict[name])
            elif isinstance(field_type_hints[name], enum.Enum):
                _state_dict[name] = field_type_hints[name](state_dict[name])
            elif name in state_dict.keys():
                value = state_dict[name]
                ft = _unpack_optional(field.type)
                if ft in _primitive_types:
                    if value is not None:
                        _state_dict[name] = ft(value)
                elif typing.get_origin(ft) in _container_types:
                    # list[str] -> list
                    origin = typing.get_origin(ft)
                    if origin in _container_to_type.keys():
                        origin = _container_to_type[origin]
                    if value is not None:
                        _state_dict[name] = origin(value)
                else:
                    _state_dict[name] = state_dict[name]
            else:
                if allow_missing:
                    _state_dict[name] = None
                else:
                    raise ValueError(f"key={name} is expected, but could not be found")

        return cls(**_state_dict)

    def __repr__(self):
        # todo: better looking
        return super().__repr__()


# config decorator
def config(cls=None,
           is_root: bool = False
           ) -> Type[ChikaConfig]:
    """ A wrapper to make ChikaConfig ::

    @config
    class DataConfig:
        name: cifar10

    Args:
        cls: wrapped class. Class name is expected to be FooConfig, and foo is used as key if this class is used as a child
        is_root: If True, some `job_id`, `output_dir` will be set

    Returns: config in dataclass and ChikaConfig

    """

    def wrap(cls):
        # mro[0] is cls and mro[-1] is object
        other_bases = tuple(inspect.getmro(cls)[1:-1])
        # order matters in inheritance
        bases = other_bases if ChikaConfig in other_bases else (ChikaConfig,) + other_bases
        # create cls whose baseclass is ChikaConfig
        cls = types.new_class(cls.__name__, bases, {}, lambda ns: ns.update(cls.__dict__))
        if is_root:
            cls._job_id = JOB_ID
            cls._job_dir = ORIGINAL_PATH
            git_hash = _get_git_hash()

            if git_hash is not None:
                cls._job_git_hash = git_hash
        # make cls to dataclass
        return dataclasses.dataclass(cls)

    return wrap if cls is None else wrap(cls)


# JOB_ID is expected to be a unique such as
# 0901_122412_558f5a
JOB_ID = datetime.now().strftime('%Y_%m%d_%H%M%S_') + uuid.uuid4().hex[-6:]
ORIGINAL_PATH = Path(".").resolve()
original_path = ORIGINAL_PATH


def resolve_original_path(path: str or Path
                          ) -> Path:
    return ORIGINAL_PATH / path


# entry point
def main(cfg_cls: Type[ChikaConfig] | ChikaConfig,
         strict: bool = False,
         change_job_dir: bool = False,
         job_dir_name: str = None
         ) -> Callable:
    """ Wrapper of the main function

    Args:
        cfg_cls: ChikaConfig
        strict: check if unspecified arguments exist. If True and unspecified arguments exist, raise ValueError
        change_job_dir: specify if a job specific current directory is used
        job_dir_name: Name of the job directory. 'outputs' by default

    Returns: returned value of the wrapped function

    """

    def _decorator(func: Callable):
        @wraps(func)
        def _wrapper():
            _config, remaining_args = ChikaArgumentParser(cfg_cls).parse_args_into_dataclass()
            if len(remaining_args) > 0:
                message = f"Some arguments are unknown to ChikaArgumentParser: {remaining_args}"
                if strict:
                    raise ValueError(message)
                else:
                    warnings.warn(message)

            if change_job_dir or job_dir_name is not None:
                job_dir = job_dir_name or "outputs"
                job_dir = Path(job_dir) / JOB_ID
                job_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(_config, "_job_dir"):
                    _config._job_dir = job_dir
                os.chdir(job_dir)
                save_as_file("run.yaml", _config.to_dict())
            try:
                return func(_config)
            finally:
                # after finishing the job...
                os.chdir(original_path)

        return _wrapper

    return _decorator
