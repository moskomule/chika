from __future__ import annotations

import dataclasses
import types
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Type

__all__ = ["ChikaArgumentParser",
           # space functions
           "with_help", "choices", "sequence", "required",
           # config
           "ChikaConfig",
           # decorators
           "config",
           "main"]

# types
DataClass = NewType("DataClass", Any)


# argument parser
class ChikaArgumentParser(ArgumentParser):
    """ This subclass of argparser that generates arguments from dataclasses.
     Inspired from huggingface transformers.hf_argparser.

     Args:
         dataclass_type: The top-level config class
         kwargs: kwargs for ArgumentParser
    """

    def __init__(self,
                 dataclass_type: Type[DataClass],
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.dataclass_type = dataclass_type
        self.add_dataclass_arguments(self.dataclass_type)

    def add_dataclass_arguments(self,
                                dtype: DataClass,
                                prefix: Optional[str] = None
                                ) -> None:
        for field in dataclasses.fields(dtype):
            # if dtype is base, --config
            # if dtype is subconfig, --main.subconfig
            field_name = f"--{field.name}" if prefix is None else f"--{prefix}.{field.name}"
            kwargs: Dict[str, Any] = field.metadata.copy()

            # Optional is difficult to handle...
            type_string = str(field.type)
            for prim_type in (int, float, str):
                # Optional[List[int]] -> List[int]
                if type_string == f"typing.Union[List[{prim_type}], NoneType]":
                    field.type = List[prim_type]
                # Optional[int] -> int
                if type_string == f"typing.Union[{prim_type.__name__}, NoneType]":
                    field.type = prim_type

            if isinstance(field.type, type) and issubclass(field.type, Enum):
                kwargs["choices"] = list(field.type)
                kwargs["type"] = field.type
                if kwargs.get("required") is None:
                    kwargs["default"] = field.default

            elif field.type is bool or field.type is Optional[bool]:
                kwargs["action"] = "store_false" if field.default is True else "store_true"

            elif _is_type_list_or_tuple(field.type):
                if kwargs.get("nargs") is None:
                    kwargs["nargs"] = "+"
                kwargs["type"] = field.type.__args__[0]

            # for ChikaConfig
            elif dataclasses.is_dataclass(field.type):
                self.add_dataclass_arguments(field.type, prefix=field.name)
                continue

            else:
                kwargs["type"] = field.type
                # value is not missing
                if field.default is not dataclasses.MISSING:
                    kwargs["default"] = field.default

            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclass(self,
                                  args: Optional[List[str]] = None,
                                  ) -> [DataClass, Any]:
        namespace, remaining_args = self.parse_known_args(args=args)

        def unflatten(flatten_map: Dict[str, Any]) -> Dict[str, Any]:
            unflatten_dict = defaultdict(dict)
            for k, v in flatten_map.items():
                # dataclass
                # what if nested?
                if "." in k:
                    cls_name, cls_field = k.split(".", 1)
                    if "." in cls_field:
                        raise RuntimeError("Double nested config is not supported yet")
                    unflatten_dict[cls_name][cls_field] = v
                else:
                    unflatten_dict[k] = v

            return unflatten_dict

        unflattened = unflatten(vars(namespace))

        for field in dataclasses.fields(self.dataclass_type):
            if dataclasses.is_dataclass(field.type):
                args = unflattened[field.name]
                unflattened[field.name] = field.type(**args)

        dclass = self.dataclass_type(**unflattened)
        return dclass, remaining_args


# typing helpers
def _is_type_list_or_tuple(type: Type):
    return hasattr(type, "__origin__") and issubclass(type.__origin__, (List, Tuple))


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


def required(*, help: Optional[str] = None) -> dataclasses.Field:
    """ Add a missing value with a help message. This value must be specified later. ::

    Args:
        help: help message

    Returns: help message.

    """

    meta = {'required': True}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


@dataclasses.dataclass
class ChikaConfig:
    # mixin

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls,
                  state_dict: Dict[str, Any]
                  ) -> ChikaConfig:
        return cls(**state_dict)


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
