from __future__ import annotations

import dataclasses
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
from pathlib import Path
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
    """ This subclass of argparser generates argiuments from dataclasses inspired from huggingface transformers.hf_argparser.
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
                                  return_remaining_strings=False,
                                  look_for_args_file=True,
                                  args_filename=None
                                  ) -> DataClass:
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
        if return_remaining_strings:
            return dclass, remaining_args

        else:
            if len(remaining_args) > 0:
                raise ValueError(f"Some specified arguments are not used by _ChikaArgumentParser: {remaining_args}")
            return dclass


# typing helpers
def _is_type_list_or_tuple(type: Type):
    return hasattr(type, "__origin__") and issubclass(type.__origin__, (List, Tuple))


# configs

def with_help(default: Any,
              help: Optional[str] = None
              ) -> dataclasses.Field:
    """ Add help to ChikaConfig, which is used in ArgumentParser

    Args:
        default: default value
        help: help message

    Returns: default value with help message

    """

    meta = {'default': default}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def choices(*values: Any,
            help: Optional[str] = None
            ) -> dataclasses.Field:
    """ Add choices to ChikaConfig, which is used in ArgumentParser. The first value is used as the default value

    Args:
        *values:
        help:

    Returns:

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
    meta = {'default': list(values)}
    if size is not None:
        meta['nargs'] = size
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)


def required(*, help: Optional[str] = None) -> dataclasses.Field:
    meta = {'required': True}
    if help is not None:
        meta['help'] = help
    return dataclasses.field(metadata=meta)
