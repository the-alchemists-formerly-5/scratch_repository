from __future__ import annotations

import itertools

import polars as pl


def _enumerate_column(
    df: pl.DataFrame, column: str
) -> tuple[dict[int, str], dict[str, int]]:
    val_lookup = dict(
        enumerate(sorted(set(filter(None, df.select(pl.col(column))[column]))), 1)
    )
    key_lookup = {val: key for key, val in val_lookup.items()}

    return (val_lookup, key_lookup)


# Load data
df = pl.read_parquet("enveda_library_subset.parquet")

# Create numerical enum values with translation tables for going back and forth.
(adduct_lookup, adduct_id_lookup) = _enumerate_column(df, "adduct")
(collision_lookup, collision_id_lookup) = _enumerate_column(df, "collision_energy")
(instrument_lookup, instrument_id_lookup) = _enumerate_column(df, "instrument_type")
(compound_lookup, compound_id_lookup) = _enumerate_column(df, "compound_class")
# TODO: scaffold_smiles??

# Max mzs to pad (x2 since we will be interleaving with an equal number of intensities).
max_mzs = (
    df.with_columns(mzs_len=pl.col("mzs").list.len()).select("mzs_len").max().item() * 2
)
print(f"Max number of mzs + intensities: {max_mzs}")


# Convert to pure numbers.
def _conv(row: tuple) -> tuple:
    (
        _,
        _,
        mzs,
        intensities,
        in_silico,
        _,
        adduct,
        collision_energy,
        instrument_type,
        compound_class,
        _,
        _,
    ) = row
    # Interleave mzs & intensities, then pad to max length.
    mzs_intensities = list(itertools.chain.from_iterable(zip(mzs, intensities)))
    padding = [0.0] * (max_mzs - len(mzs_intensities))

    return (
        mzs_intensities + padding,
        int(in_silico),
        adduct_id_lookup.get(adduct, 0),
        collision_id_lookup.get(collision_energy, 0),
        instrument_id_lookup.get(instrument_type, 0),
        compound_id_lookup.get(compound_class, 0),
    )


# Comment out next line to give computer a workout.
df = df.head()

numeric_df = df.map_rows(_conv)

training_df = pl.DataFrame(
    {
        "precusor_mz": df["precursor_mz"],
        "precursor_charge": df["precursor_charge"],
        "mzs_intensities": numeric_df["column_0"],
        "in_silico": numeric_df["column_1"],
        "smiles": df["smiles"],
        "adduct": numeric_df["column_2"],
        "collision_energy": numeric_df["column_3"],
        "instrument_type": numeric_df["column_4"],
        "compound_class": numeric_df["column_5"],
        "entropy": df["entropy"],
        "scaffold_smiles": df["scaffold_smiles"],
    }
)

print(training_df.head())