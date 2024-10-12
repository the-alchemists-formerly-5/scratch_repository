import pandas as pd
import pytest

from team5.data.data_split import sort_dataframe_by_scaffold, split_dataframe


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        "scaffold_smiles": ["CCCC", "AAAA", "BBBB", "DDDD", "CCCC"],
        "other_column": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


def test_sort_dataframe_by_scaffold(sample_dataframe):
    """Test the sort_dataframe_by_scaffold function."""
    result = sort_dataframe_by_scaffold(sample_dataframe)

    # Check if the result is sorted by scaffold_smiles
    assert list(result["scaffold_smiles"]) == ["AAAA", "BBBB", "CCCC", "CCCC", "DDDD"]

    # Check if other columns are preserved
    assert list(result["other_column"]) == [2, 3, 1, 5, 4]

    # Check if the length of the DataFrame remains the same
    assert len(result) == len(sample_dataframe)


def test_sort_dataframe_by_scaffold_empty():
    """Test the sort_dataframe_by_scaffold function with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["scaffold_smiles", "other_column"])
    result = sort_dataframe_by_scaffold(empty_df)

    assert len(result) == 0
    assert list(result.columns) == ["scaffold_smiles", "other_column"]


def test_split_dataframe(sample_dataframe):
    """Test the split_dataframe function with default split ratio."""
    df_train, df_test = split_dataframe(sample_dataframe)

    assert len(df_train) == 4
    assert len(df_test) == 1

    # Check if the split preserves order
    assert list(df_train["scaffold_smiles"]) == ["CCCC", "AAAA", "BBBB", "DDDD"]
    assert list(df_test["scaffold_smiles"]) == ["CCCC"]


def test_split_dataframe_custom_ratio(sample_dataframe):
    """Test the split_dataframe function with a custom split ratio."""
    df_train, df_test = split_dataframe(sample_dataframe, split_ratio=0.6)

    assert len(df_train) == 3
    assert len(df_test) == 2

    # Check if the split preserves order
    assert list(df_train["scaffold_smiles"]) == ["CCCC", "AAAA", "BBBB"]
    assert list(df_test["scaffold_smiles"]) == ["DDDD", "CCCC"]


def test_split_dataframe_empty():
    """Test the split_dataframe function with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["scaffold_smiles", "other_column"])
    df_train, df_test = split_dataframe(empty_df)

    assert len(df_train) == 0
    assert len(df_test) == 0
    assert list(df_train.columns) == ["scaffold_smiles", "other_column"]
    assert list(df_test.columns) == ["scaffold_smiles", "other_column"]


def test_split_dataframe_single_row():
    """Test the split_dataframe function with a single-row DataFrame."""
    single_row_df = pd.DataFrame({"scaffold_smiles": ["CCCC"], "other_column": [1]})
    df_train, df_test = split_dataframe(single_row_df, 0.9)

    assert len(df_train) == 0
    assert len(df_test) == 1
    assert list(df_test["scaffold_smiles"]) == ["CCCC"]


def test_split_dataframe_extreme_ratios(sample_dataframe):
    """Test the split_dataframe function with extreme split ratios."""
    # Test with 0% split
    df_train, df_test = split_dataframe(sample_dataframe, split_ratio=0)
    assert len(df_train) == 0
    assert len(df_test) == 5

    # Test with 100% split
    df_train, df_test = split_dataframe(sample_dataframe, split_ratio=1)
    assert len(df_train) == 5
    assert len(df_test) == 0
