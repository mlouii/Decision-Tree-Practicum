from random import sample

def remove_data(df, columns, removal_percent=10):
    new_df = df.copy(deep=True)
    num_rows = len(df)
    num_to_remove = int(num_rows*(removal_percent/100))
    for col in columns:
        remove_indexes = sample(range(num_rows), num_to_remove)
        new_df.loc[remove_indexes, col] = np.nan
    return new_df
    