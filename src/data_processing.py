import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Scale numerical features using StandardScaler
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# Create interaction terms between specified features
def create_interaction_terms(df, interactions):
    for feature_a, feature_b in interactions:
        interaction_name = f"{feature_a}_{feature_b}_interaction"
        df[interaction_name] = df[feature_a] * df[feature_b]
    return df

# Drop features that are highly correlated
def drop_correlated_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df
