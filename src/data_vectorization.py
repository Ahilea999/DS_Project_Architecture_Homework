def create_composite_feature(df, features, new_feature_name):
    """Create a composite feature by summing selected features."""
    df[new_feature_name] = df[features].sum(axis=1)
    return df