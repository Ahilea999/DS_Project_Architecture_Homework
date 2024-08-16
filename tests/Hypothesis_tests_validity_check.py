def test_no_high_correlations(self):
    """Ensure no features remaining have a high correlation."""
    df_reduced = drop_correlated_features(self.df.copy(), threshold=0.8)
    corr_matrix = df_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    self.assertTrue((upper.max().max() < 0.8))