import contextlib
import enum
import sys
from pathlib import Path

from chika import config, main, original_path, resolve_original_path
from chika.utils import load_from_file


@contextlib.contextmanager
def _clean_argv(new):
    original = sys.argv.copy()
    sys.argv += new
    yield
    sys.argv = original


def test_main():
    @config
    class C:
        a: int = 1

    @main(C)
    def f(cfg):
        return cfg.a + 1

    assert f() == 2

    with _clean_argv(['--a', '2']):
        assert f() == 3

    @config
    class D:
        c: C
        d: int = 1

    @main(D)
    def f(cfg):
        return cfg.c.a + cfg.d

    assert f() == 2

    with _clean_argv(['--c.a', '2']):
        assert f() == 3

    with _clean_argv(['--d', '2']):
        assert f() == 3


def test_main_enum():
    class A(str, enum.Enum):
        a = "a"
        b = "b"

    @config
    class C:
        a: A = A.a

    @main(C)
    def f(cfg):
        return cfg.a

    assert f() == A.a

    with _clean_argv(['--a', 'b']):
        assert f() == A.b


def test_main_cd():
    @config
    class C:
        a: int

    @main(C, change_job_dir=True)
    def f(cfg):
        return resolve_original_path(f"{cfg.a}.pt"), Path(".").resolve()

    with _clean_argv(['--a', '2']):
        resolved_path, working_dir = f()
        assert resolved_path == (original_path / "2.pt")
        assert load_from_file(working_dir / "run.yaml") == {"a": 2}


def test_main_job_id():
    @config(is_root=True)
    class C:
        a: int = 1

    @main(C)
    def f(cfg):
        assert hasattr(cfg, "_job_dir")

    f()
