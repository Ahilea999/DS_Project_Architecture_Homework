import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def scale_features(df, features):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def create_interaction_terms(df, interactions):
    """Create interaction terms between specified features."""
    for feature_a, feature_b in interactions:
        interaction_name = f"{feature_a}_{feature_b}_interaction"
        df[interaction_name] = df[feature_a] * df[feature_b]
    return df

def drop_correlated_features(df, threshold=0.8):
    """Drop features that are highly correlated."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df
