# Create a composite feature by summing selected features
def create_composite_feature(df, features, new_feature_name):
    df[new_feature_name] = df[features].sum(axis=1)
    return df