from __future__ import annotations

import sys
from functools import partial
from itertools import chain
from pathlib import Path

import polars as pl
import torch
from transformers import AutoTokenizer

HEAD = 1000
MAX_MZS = 512


ADDUCT = {
    "[2M+C2H4N]+": 0,
    "[2M+CHO2]-": 1,
    "[2M+Ca-H]+": 2,
    "[2M+Ca]2+": 3,
    "[2M+H4N]+": 4,
    "[2M+H]+": 5,
    "[2M+K]+": 6,
    "[2M+Na-H2]-": 7,
    "[2M+Na2-H]+": 8,
    "[2M+Na]+": 9,
    "[2M-H3O2]+": 10,
    "[2M-HO]+": 11,
    "[2M-H]-": 12,
    "[3M+Ca-H]+": 13,
    "[3M+Ca]2+": 14,
    "[3M+H4N]+": 15,
    "[3M+K]+": 16,
    "[3M+Na]+": 17,
    "[4M+Ca]2+": 18,
    "[5M+Ca]2+": 19,
    "[Anion]-": 20,
    "[Cat]+": 21,
    "[M+C2H2NaO2]-": 22,
    "[M+C2H3NNa]+": 23,
    "[M+C2H3O2]-": 24,
    "[M+C2H4N]+": 25,
    "[M+C2H7N2]+": 26,
    "[M+C2H7OS]+": 27,
    "[M+CH3O2]+": 28,
    "[M+CH5O]+": 29,
    "[M+CHO2]-": 30,
    "[M+Ca-H]+": 31,
    "[M+Ca]2+": 32,
    "[M+Cl]-": 33,
    "[M+H-O3S]+": 34,
    "[M+H2]2+": 35,
    "[M+H3]3+": 36,
    "[M+H4N]+": 37,
    "[M+HNa]2+": 38,
    "[M+HO]+": 39,
    "[M+H]+": 40,
    "[M+K-H2]-": 41,
    "[M+K]+": 42,
    "[M+Li-H]+": 43,
    "[M+Li]+": 44,
    "[M+N-O2]+": 45,
    "[M+Na-H2]-": 46,
    "[M+Na-H]+": 47,
    "[M+Na2-H]+": 48,
    "[M+Na]+": 49,
    "[M-C12H19O9]+": 50,
    "[M-C2H3O]-": 51,
    "[M-C6H11O5]-": 52,
    "[M-C6H9O5]+": 53,
    "[M-C8H9O]+": 54,
    "[M-C9H9O5]+": 55,
    "[M-CH3F2O2]-": 56,
    "[M-CH3O]+": 57,
    "[M-CH3]-": 58,
    "[M-CHO2]-": 59,
    "[M-H2N]+": 60,
    "[M-H2O2]2+": 61,
    "[M-H2]-": 62,
    "[M-H2]2-": 63,
    "[M-H3O2]+": 64,
    "[M-H3O]-": 65,
    "[M-H4O3]2+": 66,
    "[M-H5O3]+": 67,
    "[M-H7O4]+": 68,
    "[M-H9O5]+": 69,
    "[M-HO]+": 70,
    "[M-H]-": 71,
}

PREPARED_PARQUET = "prepared"


def _enumerize(parquet: Path, column: str) -> dict[str, int]:
    values = dict(
        enumerate(
            pl.scan_parquet(parquet).select(column).unique().collect()[column].sort()
        )
    )
    return {v: k for k, v in values.items()}


def interleave(elem: pl.Struct) -> list[float]:
    labels = list(chain.from_iterable(zip(elem["mzs"], elem["intensities"])))
    padding = [0.0] * (MAX_MZS * 2 - len(labels))
    return labels[: 2 * MAX_MZS] + padding


def vectorize(lookup: dict[str, int], elem: str) -> list[int]:
    vec = [0] * len(lookup)
    vec[lookup[elem]] = 1
    return vec


def tokenize(tokenizer: AutoTokenizer, seq: str) -> list[int]:
    return tokenizer.encode(seq, padding="max_length")


def attention_mask(tokenizer: AutoTokenizer, seq: str) -> list[int]:
    return tokenizer(seq, padding="max_length")["attention_mask"]


def prepare_data(df: pl.DataFrame, head: int = 0, filename: Path = PREPARED_PARQUET) -> None:
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    if head:
        df = df.head(head)

    print("Transforming raw data and writing prepared data to disk")
    df.lazy().with_columns(
        tokenized_smiles=pl.col("smiles").map_elements(
            partial(tokenizer.encode, padding="max_length"),
            return_dtype=pl.List(pl.Int64),
        ),
        attention_mask=pl.col("smiles").map_elements(
            partial(attention_mask, tokenizer), return_dtype=pl.List(pl.Int64)
        ),
        labels=pl.struct(["mzs", "intensities"]).map_elements(
            interleave, return_dtype=pl.List(pl.Float64)
        ),
        enum_adduct=pl.col("adduct").map_elements(
            partial(vectorize, ADDUCT),
            return_dtype=pl.List(pl.Int64),
            skip_nulls=False,
        ),
        enum_in_silico=pl.col("in_silico").cast(pl.Int64),
    ).select(
        [
            "tokenized_smiles",
            "attention_mask",
            "labels",
            pl.concat_list(
                "precursor_mz", "precursor_charge", "enum_in_silico", "enum_adduct"
            ).alias("supplementary_data"),
        ]
    ).sink_parquet(
        filename
    )


def tensorize(df: pl.DataFrame, head: int = 0, split: str = "train") -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    filename = Path(f"{PREPARED_PARQUET}_{split}.parquet")
    if not filename.exists():
        prepare_data(df, head=head, filename=filename)
    else:
        print(f"Prepared data already exists for {split} split, skipping")

    print(f"Reading prepared data from disk ({filename})")
    df = pl.read_parquet(filename)

    return (
        torch.tensor(df["tokenized_smiles"], dtype=torch.long),
        torch.tensor(df["attention_mask"], dtype=torch.long),
        torch.tensor(df["labels"], dtype=torch.float),
        torch.tensor(df["supplementary_data"], dtype=torch.float),
    )


if __name__ == "__main__":
    print(f"Only run this way for testing/debugging! This only reads {HEAD} rows.")
    (tokenized_smiles, attention_mask, labels, supplementary_data) = tensorize(
        pl.read_parquet(sys.argv[1]), head=HEAD
    )

    print("tokenized_smiles: ", tokenized_smiles)
    print("attention_mask: ", attention_mask)
    print("labels: ", labels)
    print("supplementary_data: ", supplementary_data)
