import polars as pl
from src.team5.data.data_split import sort_dataframe_by_scaffold, split_dataframe
from src.team5.data.prepare import tensorize
from torch.utils.data import Dataset


class SMILESDataset(Dataset):

    def __init__(self, tokenized_smiles, attention_mask, labels, supplementary_data):
        self.tokenized_smiles = tokenized_smiles
        self.attention_mask = attention_mask
        self.labels = labels
        self.supplementary_data = supplementary_data

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

if __name__ == "__main__":
    print(f"Loading the data!.")

    parquet_file = "data/raw/enveda_library_subset_10percent.parquet"
    df = pl.read_parquet(parquet_file)

    df_sorted = sort_dataframe_by_scaffold(df)

    df_train, df_test = split_dataframe(df_sorted, 0.9)

    (tokenized_smiles, labels, supplementary_data) = tensorize(df_train, len(df_train))

    print("tokenized_smiles: ", tokenized_smiles)
    print("labels: ", labels)
    print("supplementary_data: ", supplementary_data)
