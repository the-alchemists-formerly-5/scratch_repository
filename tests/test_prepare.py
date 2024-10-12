from itertools import chain

import polars as pl
import pytest

from src.team5.data.prepare import MAX_MZS, interleave, vectorize


# Mock pl.Struct for testing
def mock_struct(mzs, intensities):
    return {"mzs": mzs, "intensities": intensities}


def test_interleave_normal_case():
    input_struct = mock_struct([1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    result = interleave(input_struct)
    expected = [1.0, 100.0, 2.0, 200.0, 3.0, 300.0] + [0.0] * (MAX_MZS * 2 - 6)
    assert result == expected


def test_interleave_empty_input():
    input_struct = mock_struct([], [])
    result = interleave(input_struct)
    expected = [0.0] * (MAX_MZS * 2)
    assert result == expected


def test_interleave_single_element():
    input_struct = mock_struct([1.0], [100.0])
    result = interleave(input_struct)
    expected = [1.0, 100.0] + [0.0] * (MAX_MZS * 2 - 2)
    assert result == expected


def test_interleave_max_elements():
    mzs = [float(i) for i in range(MAX_MZS)]
    intensities = [float(i * 100) for i in range(MAX_MZS)]
    input_struct = mock_struct(mzs, intensities)
    result = interleave(input_struct)
    expected = list(chain.from_iterable(zip(mzs, intensities)))
    assert result == expected
    assert len(result) == MAX_MZS * 2


def test_interleave_more_than_max_elements():
    mzs = [float(i) for i in range(MAX_MZS + 1)]
    intensities = [float(i * 100) for i in range(MAX_MZS + 1)]
    input_struct = mock_struct(mzs, intensities)
    result = interleave(input_struct)
    expected = list(chain.from_iterable(zip(mzs[:MAX_MZS], intensities[:MAX_MZS])))
    assert result == expected
    assert len(result) == MAX_MZS * 2


def test_interleave_uneven_inputs():
    input_struct = mock_struct([1.0, 2.0, 3.0], [100.0, 200.0])

    result = interleave(input_struct)
    expected = [1.0, 100.0, 2.0, 200.0] + [0.0] * (MAX_MZS * 2 - 4)
    assert result == expected
    assert len(result) == MAX_MZS * 2


def test_interleave_result_length():
    input_struct = mock_struct([1.0, 2.0], [100.0, 200.0])
    result = interleave(input_struct)
    assert len(result) == MAX_MZS * 2


def test_interleave_padding():
    input_struct = mock_struct([1.0, 2.0], [100.0, 200.0])
    result = interleave(input_struct)
    assert result[4:] == [0.0] * (MAX_MZS * 2 - 4)


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
    pytest.main()
