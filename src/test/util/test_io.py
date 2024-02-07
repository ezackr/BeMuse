import os

from src.main.util.io import get_parent_dir

current_dir: str = os.getcwd()


def test_get_parent_dir():
    parent_dir_name = get_parent_dir(current_dir, 1)
    assert parent_dir_name.endswith("test")


def test_get_parent_dir_zero():
    assert current_dir == get_parent_dir(current_dir, 0)


def test_get_parent_dir_negative():
    try:
        get_parent_dir(current_dir, -1)
        assert False
    except ValueError:
        pass
