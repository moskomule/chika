import enum
import json
from typing import List

import pytest

from chika.chika import ChikaArgumentParser, bounded, choices, config, required, sequence, with_help


def test_simple_case():
    @config
    class A:
        a: int
        b: int = 2
        c: bool = False
        d: bool = True
        e: str = "test"

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])
    assert isinstance(r, A)
    assert r.a == 1
    assert r.b == 2
    assert not r.c
    assert r.d
    assert r.e == "test"

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1", "--c", "--d"])
    assert r.c
    assert not r.d

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1", "--e", "train"])
    assert r.e == "train"

    with pytest.raises(SystemExit):
        # https://stackoverflow.com/questions/39028204/using-unittest-to-test-argparse-exit-errors
        ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1", "--b", "1.0"])


def test_nested_case():
    @config
    class A:
        a: int = 1

    @config
    class B:
        a: float
        b: A

    r, _ = ChikaArgumentParser(B).parse_args_into_dataclass(["--a", "3.2"])
    assert r.a == 3.2
    assert r.b.a == 1

    @config
    class C:
        c: B

    # this is not supported yet
    with pytest.raises(Exception):
        ChikaArgumentParser(C).parse_args_into_dataclass([])


def test_from_file(tmp_path):
    json_file = tmp_path / "test.json"
    with json_file.open("w") as f:
        json.dump({"a": 1, "b": 0.2}, f)

    @config
    class A:
        a: int
        b: float

    @config
    class B:
        a: float
        b: A

    r, _ = ChikaArgumentParser(B).parse_args_into_dataclass(f"--a 0.1 --b {json_file}".split())
    assert r.a == 0.1
    assert r.b.a == 1
    assert r.b.b == 0.2

    r, _ = ChikaArgumentParser(B).parse_args_into_dataclass(f"--a 0.1 --b {json_file} --b.b 0.3".split())
    assert r.a == 0.1
    assert r.b.a == 1
    assert r.b.b == 0.3


def test_choices():
    @config
    class A:
        a: int = choices(1, 2, 3)
        b: int = 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "2"])
    assert r.a == 2

    with pytest.raises(SystemExit):
        ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "4"])

    @config
    class A:
        b: int = 1
        a: int = choices(1, 2, 3)

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1


def test_sequence():
    @config
    class A:
        a: List[int] = sequence(1, 2, 3, size=3)
        b: int = 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == [1, 2, 3]

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1", "2", "4"])
    assert r.a == [1, 2, 4]

    with pytest.raises(SystemExit):
        ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])

    @config
    class A:
        b: int = 1
        a: List[int] = sequence(1, 2, 3, size=3)

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == [1, 2, 3]


def test_required():
    @config
    class A:
        a: int = required()
        b: int = 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])
    assert r.a == 1

    with pytest.raises(SystemExit):
        # a is required
        ChikaArgumentParser(A).parse_args_into_dataclass([])

    @config
    class A:
        b: int = 1
        a: int = required()

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])
    assert r.a == 1


def test_with_help():
    @config
    class A:
        a: int = with_help(1, help="this is help")
        b: int = 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1

    @config
    class A:
        b: int = 1
        a: int = with_help(1, help="this is help")

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1


def test_bounded():
    with pytest.raises(ValueError):
        bounded(1, -1, 0.9)

    with pytest.raises(ValueError):
        bounded(1, 1.1, 2)

    with pytest.raises(ValueError):
        bounded(1, 1, -1)

    @config
    class A:
        a: int = bounded(1, -1, 2)
        b: int = 1

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1

    with pytest.raises(SystemExit):
        r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "-1.1"])

    with pytest.raises(SystemExit):
        r, _ = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "2.1"])

    @config
    class A:
        b: int = 1
        a: int = bounded(1, -1, 2)

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1


def test_enum():
    class E(str, enum.Enum):
        a = 'a'
        b = 'b'

    @config
    class A:
        a: E = E.a

    @config
    class B:
        a: A

    r, _ = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == E.a

    r, _ = ChikaArgumentParser(B).parse_args_into_dataclass([])
    assert r.a.a == E.a
