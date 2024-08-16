def normalize_data(df, columns):
    """Normalize specified columns or all columns if none are specified."""
    columns = columns if isinstance(columns, list) else [columns]
    df[columns] = df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df
