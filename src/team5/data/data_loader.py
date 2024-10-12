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
import torch


class SMILESDataset(Dataset):
    def __init__(self, tokenized_smiles, attention_mask, labels, supplementary_data):
        self.tokenized_smiles = tokenized_smiles
        self.attention_mask = attention_mask
        self.labels = labels
        self.supplementary_data = supplementary_data

        # Create labels as a 2D array of mzs and intensities put together. Or have it flat and just concat both
        # self.labels = #TODO

        # Create supplementary data as a long concatinated list of all the supplementary data
        # self.supplementary_data = #TODO

    def __len__(self):
        return len(self.tokenized_smiles)

    def __getitem__(self, idx):
        tokenized_smiles = self.tokenized_smiles[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        supplementary_data = self.supplementary_data[idx]

        return {
            "input_ids": tokenized_smiles,
            "attention_mask": attention_mask,
            "labels": label,
            "supplementary_data": supplementary_data
        }
    
   