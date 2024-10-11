# test/train split
# Use Murcko scaffold and spectral entropy splitting for this, rather than random.
# This will ensure that similar molecules don't go into both training and test,
# causing cross contamination and over fitting.


def split_data(df):
    # implement something not random here
    return train_test_split(df, test_size=0.1, random_state=42)


# train_df, eval_df = split_data(df)

# train_dataset = SMILESDataset(train_df, tokenizer)
# eval_dataset = SMILESDataset(eval_df, tokenizer)

## batch?
