import numpy as np

from src.main.data.augment import get_random_transposition

seed = 24
np.random.seed(seed)


def test_get_random_transposition():
    sequence = np.array([[1, 0, 4, 5], [1, 0.5, 7, 5]])
    expected = np.array([[1, 0, 7, 5], [1, 0.5, 10, 5]])
    assert np.array_equal(expected, get_random_transposition(sequence))


def test_get_random_transposition_out_of_bounds():
    sequence = np.array([[1, 0, 127, 5], [1, 0.5, 7, 5]])
    expected = np.array([[1, 0, 127, 5], [1, 0.5, 11, 5]])
    assert np.array_equal(expected, get_random_transposition(sequence))
