import sys

from chika import config, main


def test_main():
    @config
    class C:
        a: int = 1

    @main(C)
    def f(cfg):
        return cfg.a + 1

    assert f() == 2

    sys.argv += ["--a", "2"]

    assert f() == 3
