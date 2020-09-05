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

    b = B(C(1))
    assert b.to_dict() == {"a": {"c": 1}}
    assert B.from_dict({"a": {"c": 1}}) == b
