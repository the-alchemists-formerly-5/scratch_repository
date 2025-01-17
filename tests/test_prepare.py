import pytest

from src.team5.data.prepare import vectorize


# Mock pl.Struct for testing
def mock_struct(mzs, intensities):
    return {"mzs": mzs, "intensities": intensities}


def test_vectorize_single_element():
    lookup = {"a": 0, "b": 1, "c": 2}
    result = vectorize(lookup, "b")
    assert result == [0, 1, 0]


def test_vectorize_first_element():
    lookup = {"x": 0, "y": 1, "z": 2}
    result = vectorize(lookup, "x")
    assert result == [1, 0, 0]


def test_vectorize_last_element():
    lookup = {"p": 0, "q": 1, "r": 2}
    result = vectorize(lookup, "r")
    assert result == [0, 0, 1]


def test_vectorize_single_item_lookup():
    lookup = {"single": 0}
    result = vectorize(lookup, "single")
    assert result == [1]


def test_vectorize_element_not_in_lookup():
    lookup = {"a": 0, "b": 1, "c": 2}
    with pytest.raises(KeyError):
        vectorize(lookup, "d")


def test_vectorize_empty_lookup():
    with pytest.raises(KeyError):
        vectorize({}, "a")


def test_vectorize_non_string_element():
    lookup = {"1": 0, "2": 1, "3": 2}
    with pytest.raises(KeyError):
        vectorize(lookup, 2)


def test_vectorize_non_int_values_in_lookup():
    lookup = {"a": "0", "b": "1", "c": "2"}
    with pytest.raises(TypeError):
        vectorize(lookup, "b")


if __name__ == "__main__":
    pytest.main(["-s"])
