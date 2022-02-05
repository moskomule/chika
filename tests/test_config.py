import enum

from chika.chika import config


def test_simple_config():
    @config
    class C:
        a: int = 1
        b: str = "test"

    c = C()
    assert c.to_dict() == {"a": 1, "b": "test"}

    d = C.from_dict({"a": 2, "b": "train"})
    assert d == C(2, "train")


def test_nested_config():
    @config
    class C:
        c: int

    @config
    class B:
        a: C
        b: int

    b = B(C(1), 3)
    assert b.to_dict() == {"a": {"c": 1}, "b": 3}
    assert B.from_dict({"a": {"c": 1}, "b": 3}) == b
    assert B.from_dict({"a": {"c": 1}}, allow_missing=True) == B(C(1), None)


def test_inherited_config():
    @config
    class C:
        a: int = 1
        b: int = 2

    @config
    class B(C):
        a: int = 2
        c: int = 3

    b = B()
    assert b.to_dict() == {"a": 2, "b": 2, "c": 3}


def test_enum():
    class A(enum.Enum):
        a = "a"
        b = "b"

    @config
    class B:
        a: A = A.a

    b = B()
    assert b.to_dict()['a'] == A.a
