import os

from src.main.util.io import get_parent_dir

current_path: str = os.path.abspath(__file__)


def test_get_parent_dir():
    parent_dir_name = get_parent_dir(current_path, 2)
    print(parent_dir_name)
    assert parent_dir_name.endswith("test")


def test_get_parent_dir_zero():
    assert current_path == get_parent_dir(current_path, 0)


def test_get_parent_dir_negative():
    try:
        get_parent_dir(current_path, -1)
        assert False
    except ValueError:
        pass
