from __future__ import annotations

import enum
import json
import subprocess
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Set, Tuple, Union

import yaml

# fileio
JSON_SUFFIXES = {".json"}
YAML_SUFFIXES = {".yaml", ".yml"}
SUPPORTED_SUFFIXES = (JSON_SUFFIXES | YAML_SUFFIXES)


def is_supported_filetype(file: str
                          ) -> bool:
    return Path(file).suffix in SUPPORTED_SUFFIXES


def load_from_file(file: str | Path
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
class DefaultUntouched(object):
    # mark as untouched default
    def __init__(self,
                 value: Any):
        self.value = value

    def __repr__(self):
        return self.value.__repr__()


# handle types
_container_types = [list, tuple, set, frozenset, List, Tuple, Set, FrozenSet]
_container_to_type = {List: list,
                      Tuple: tuple,
                      Set: set,
                      FrozenSet: frozenset}
_primitive_types = [bool, int, float, str]


def _is_optional(_type) -> bool:
    # Optional is Union and its final argument is NoneType
    if typing.get_origin(_type) is Union:
        last_arg = typing.get_args(_type)[-1]
        # check if NoneType
        return last_arg is type(None)
    return False


def _unpack_optional(_type):
    # unpack Optional
    if not _is_optional(_type):
        return _type

    args = typing.get_args(_type)[0]
    if len(args) > 1:
        warnings.warn(f"Got complex type: {_type}.")

    first_arg = args[0]
    if first_arg in _primitive_types:
        #  Optional[int] -> int
        return first_arg

    origin = typing.get_origin(first_arg)
    if origin in _container_types:
        return first_arg

    raise ValueError(f"Got unsupported type: {_type}")


def _is_container_type(_type) -> bool:
    origin = typing.get_origin(_type)
    return _type in _container_types or origin in _container_types


# handle enums
def _enum_to_value(_dict: dict):
    out = {}
    for k, v in _dict.items():
        if isinstance(v, dict):
            v = _enum_to_value(v)
        elif isinstance(v, enum.Enum):
            v = v.value
        out[k] = v
    return out


# get git info

def _get_git_hash() -> typing.Optional[str]:
    def _decode_bytes(b: bytes) -> str:
        return b.decode("ascii")[:-1]

    try:
        is_git_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        if _decode_bytes(is_git_repo) == "true":
            git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE).stdout
            return _decode_bytes(git_hash)
    except FileNotFoundError:
        pass
    return None
