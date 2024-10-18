import polars as pl
from src.team5.data.data_loader import SMILESDataset
from src.team5.data.data_split import sort_dataframe_by_scaffold, split_dataframe
from src.team5.data.prepare import tensorize
from config import DATASET

def preprocess_data():
    # Load the dataset
    df = pl.read_parquet(DATASET)
    
    # Sort and split the dataframe
    df_sorted = sort_dataframe_by_scaffold(df)
    df_train, df_test = split_dataframe(df_sorted, split_ratio=0.9)
    
    # Print column names and head of dataframes
    print("Train columns:", df_train.columns)
    print("Test columns:", df_test.columns)
    print("\nTrain head:")
    print(df_train.head())
    print("\nTest head:")
    print(df_test.head())
    
    # Tensorize the data
    (train_tokenized_smiles, train_attention_mask, train_labels, train_supplementary_data) = tensorize(df_train, split="train")
    (test_tokenized_smiles, test_attention_mask, test_labels, test_supplementary_data) = tensorize(df_test, split="test")
    
    # Create datasets
    train_dataset = SMILESDataset(train_tokenized_smiles, train_attention_mask, train_labels, train_supplementary_data)
    test_dataset = SMILESDataset(test_tokenized_smiles, test_attention_mask, test_labels, test_supplementary_data)
    # Debug information: print data types of the first item in train_dataset
    print("\nData types of the first item in train_dataset:")
    print({k: v.dtype for k, v in train_dataset[0].items()})
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = preprocess_data()
    print("Data preprocessing completed.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")