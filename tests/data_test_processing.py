import unittest
import pandas as pd
import numpy as np
from src.data_processing import scale_features, create_interaction_terms, drop_correlated_features, normalize_data, \
    create_composite_feature


class TestDataProcessing(unittest.TestCase):

    def test_scale_features(self):
        """Test scaling of features."""
        df_scaled = scale_features(self.df.copy(), ['A', 'B'])
        self.assertAlmostEqual(df_scaled['A'].mean(), 0)
        self.assertAlmostEqual(df_scaled['A'].std(), 1)
        self.assertAlmostEqual(df_scaled['B'].mean(), 0)
        self.assertAlmostEqual(df_scaled['B'].std(), 1)

    def test_create_interaction_terms(self):
        """Test creation of interaction terms."""
        df_interaction = create_interaction_terms(self.df.copy(), [('A', 'B'), ('C', 'D')])
        self.assertIn('A_B_interaction', df_interaction.columns)
        self.assertIn('C_D_interaction', df_interaction.columns)
        self.assertEqual(df_interaction['A_B_interaction'][0], 5)  # 1 * 5
        self.assertEqual(df_interaction['C_D_interaction'][0], 1000)  # 10 * 100

    def test_drop_correlated_features(self):
        """Test dropping of correlated features."""
        df_reduced = drop_correlated_features(self.df.copy(), threshold=0.95)
        self.assertNotIn('D', df_reduced.columns)  # D should be dropped due to high correlation with E

    def test_normalize_data(self):
        """Test normalization of data."""
        df_normalized = normalize_data(self.df.copy(), ['A', 'B'])
        self.assertAlmostEqual(df_normalized['A'].min(), 0)
        self.assertAlmostEqual(df_normalized['A'].max(), 1)
        self.assertAlmostEqual(df_normalized['B'].min(), 0)
        self.assertAlmostEqual(df_normalized['B'].max(), 1)

    def test_create_composite_feature(self):
        """Test creation of a composite feature."""
        df_composite = create_composite_feature(self.df.copy(), ['A', 'B'], 'composite_AB')
        self.assertIn('composite_AB', df_composite.columns)
        self.assertEqual(df_composite['composite_AB'][0], 6)  # 1 + 5


if __name__ == '__main__':
    unittest.main()