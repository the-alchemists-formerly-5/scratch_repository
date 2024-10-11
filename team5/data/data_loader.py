# import the data (with pandas?)
import pandas as pd

## Load the dataset (for some reason this didn't work for me)
# df = pd.read_parquet('enveda_library_subset 2.parquet')

# print(df.head())


# tokenize the SMILES. Do we need to pad? If so, what's the max length
def tokenize_function(examples):
    return tokenizer(
        examples["smiles"], truncation=True, padding="max_length", max_length=128
    )


# custom Dataset class for all the types of data.
# I think we might want to make a new 'column' of data that combines mzs and intensities into "label"

from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.smiles = dataframe["smiles"].tolist()
        self.precursor_mz = dataframe["precursor_mz"].tolist()
        self.precursor_charge = dataframe["precursor_charge"].tolist()
        self.collision_energy = dataframe["collision_energy"].tolist()
        self.instrument_type = dataframe["instrument_type"].tolist()
        self.in_silico_label = dataframe["in_silico_label"].tolist()
        self.adduct = dataframe["adduct"].tolist()
        self.compound_class = dataframe["compound_class"].tolist()
        self.mzs = dataframe["mzs"].tolist()
        self.intensities = dataframe["intensities"].tolist()

        # Create labels as a 2D array of mzs and intensities put together. Or have it flat and just concat both
        # self.labels = #TODO

        # Create supplementary data as a long concatinated list of all the supplementary data
        # self.supplementary_data = #TODO

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        precursor_mz = self.precursor_mz[idx]
        label = self.labels[idx]

        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        # Prepare item
        item = {
            key: val.squeeze(0) for key, val in inputs.items()
        }  # Remove batch dimension
        item["precursor_mz"] = torch.tensor(precursor_mz, dtype=torch.float)
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item
