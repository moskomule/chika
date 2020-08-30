from dataclasses import dataclass
from typing import List

import pytest

from chika.chika import ChikaArgumentParser, choices, required, sequence, with_help


def test_simple_case():
    @dataclass
    class A:
        a: int
        b: int = 2
        c: bool = False
        d: bool = True
        e: str = "test"

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])
    assert isinstance(r, A)
    assert r.a == 1
    assert r.b == 2
    assert not r.c
    assert r.d
    assert r.e == "test"

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--c", "--d"])
    assert r.c
    assert not r.d

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--e", "train"])
    assert r.e == "train"

    with pytest.raises(SystemExit):
        # https://stackoverflow.com/questions/39028204/using-unittest-to-test-argparse-exit-errors
        ChikaArgumentParser(A).parse_args_into_dataclass(["--b", "1.0"])


def test_nested_case():
    @dataclass
    class A:
        a: int

    @dataclass
    class B:
        a: float
        b: A = A(1)

    r = ChikaArgumentParser(B).parse_args_into_dataclass(["--a", "3.2", "--b.a", "2"])
    assert r.a == 3.2
    assert r.b.a == 2

    @dataclass
    class C:
        c: B

    # this is not supported yet
    with pytest.raises(Exception):
        ChikaArgumentParser(C).parse_args_into_dataclass([])


def test_choices():
    @dataclass
    class A:
        a: int = choices(1, 2, 3)

    r = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "2"])
    assert r.a == 2

    with pytest.raises(SystemExit):
        ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "4"])


def test_sequence():
    @dataclass
    class A:
        a: List[int] = sequence(1, 2, 3, size=3)

    r = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == [1, 2, 3]

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1", "2", "4"])
    assert r.a == [1, 2, 4]

    with pytest.raises(SystemExit):
        ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])


def test_required():
    @dataclass
    class A:
        a: int = required()

    r = ChikaArgumentParser(A).parse_args_into_dataclass(["--a", "1"])
    assert r.a == 1

    with pytest.raises(SystemExit):
        # a is required
        ChikaArgumentParser(A).parse_args_into_dataclass([])


def test_with_help():
    @dataclass
    class A:
        a: int = with_help(1, "this is help")

    r = ChikaArgumentParser(A).parse_args_into_dataclass([])
    assert r.a == 1
