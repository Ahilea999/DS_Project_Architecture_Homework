def preprocess_data(df):
    """Complete preprocessing pipeline."""
    features_to_scale = ['LungFunctionFVC', 'LungFunctionFEV1', 'DustExposure']
    df = scale_features(df, features_to_scale)

    interaction_pairs = [('LungFunctionFVC', 'Wheezing'), ('LungFunctionFEV1', 'Coughing')]
    df = create_interaction_terms(df, interaction_pairs)

    df = drop_correlated_features(df, threshold=0.8)

    return df