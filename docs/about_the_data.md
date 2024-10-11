# Information about the data

The data should be downloaded to `data/raw/enveda_library_subset.parquet`

## Initial exploration
We can look at the data using the `parq` tool (which will be install as a dev dependency with poetry).

First look at the metadata:
```bash
parq data/raw/enveda_library_subset.parquet

 # Metadata
 <pyarrow._parquet.FileMetaData object at 0x10589e660>
  created_by: parquet-cpp-arrow version 17.0.0
  num_columns: 12
  num_rows: 1038607
  num_row_groups: 1
  format_version: 2.6
  serialized_size: 7194
  ```
We can get the schema with

  ```bash
   parq data/raw/enveda_library_subset.parquet --schema

 # Schema
 <pyarrow._parquet.ParquetSchema object at 0x107b8be40>
required group field_id=-1 schema {
  optional double field_id=-1 precursor_mz;
  optional double field_id=-1 precursor_charge;
  optional group field_id=-1 mzs (List) {
    repeated group field_id=-1 list {
      optional double field_id=-1 element;
    }
  }
  optional group field_id=-1 intensities (List) {
    repeated group field_id=-1 list {
      optional double field_id=-1 element;
    }
  }
  optional boolean field_id=-1 in_silico;
  optional binary field_id=-1 smiles (String);
  optional binary field_id=-1 adduct (String);
  optional binary field_id=-1 collision_energy (String);
  optional binary field_id=-1 instrument_type (String);
  optional binary field_id=-1 compound_class (String);
  optional double field_id=-1 entropy;
  optional binary field_id=-1 scaffold_smiles (String);
}
```

And to look at the records, remembering that parquet is a column first format:

```bash
% parq data/raw/enveda_library_subset.parquet --head 3
   precursor_mz  precursor_charge  \
0    401.414178               1.0
1    399.399626              -1.0
2    331.191483              -1.0

                                                 mzs  \
0  [41.03858, 43.05423, 45.06988, 55.05423, 57.06...
1  [69.07097, 71.08662, 81.07097, 83.08662, 85.10...
2  [41.00329, 43.01894, 43.05532, 57.03459, 59.01...

                                         intensities  in_silico  \
0  [0.3332327195894959, 0.5552067612435858, 0.012...       True
1  [0.0043, 0.005966666666666667, 0.0026333333333...       True
2  [0.47981042654028433, 0.1261611374407583, 0.00...       True

                                              smiles  adduct collision_energy  \
0  CC1CCC2(C)C(CCC3C2CCC2(C)C(C(C)CCC(C)C(C)C)CCC...  [M+H]+         10-20-40
1  CC1CCC2(C)C(CCC3C2CCC2(C)C(C(C)CCC(C)C(C)C)CCC...  [M-H]-         10-20-40
2     CC(C)=CC1OC(C)(CC(=O)O)C2=C1C(C)C1CCC(C)C1C2=O  [M-H]-         10-20-40

  instrument_type             compound_class   entropy  \
0   cfm-predict 4        Cholestane steroids  3.837670
1   cfm-predict 4        Cholestane steroids  1.254399
2   cfm-predict 4  Amphilectane diterpenoids  3.678560

               scaffold_smiles
0  C1CCC2C(C1)CCC1C3CCCC3CCC21
1  C1CCC2C(C1)CCC1C3CCCC3CCC21
2      O=C1C2=C(COC2)CC2CCCC12
```


## Splitting small subset of data for experimentation

As the full data is a big file, for experimentation we can split it into a 10% subset using the following script

```bash
python3 scripts/extract_subset.py
```

This will generate the file 'data/raw/enveda_library_subset_10percent.parquet'

# How to split the test/train data

[Blog to read](https://practicalcheminformatics.blogspot.com/2023/06/getting-real-with-molecular-property.html)
