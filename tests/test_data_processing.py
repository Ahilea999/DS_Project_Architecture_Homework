import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Add the src directory to the Python path
sys.path.append(os.path.abspath('../src'))

from data_processing import scale_features

class TestDataProcessing(unittest.TestCase):
    def __init__(self, methodName='test_scale_features', df=None, features=None):
        super(TestDataProcessing, self).__init__(methodName)
        self.df = df
        self.features = features

    def test_scale_features(self):
        """Test the scaling of features."""
        if self.df is not None and self.features is not None:
            scaled_df = scale_features(self.df.copy(), self.features)
            for feature in self.features:
                self.assertAlmostEqual(scaled_df[feature].mean(), 0, places=6)
                self.assertAlmostEqual(scaled_df[feature].std(), 1, delta=0.1)
        else:
            self.fail("DataFrame or features not provided")


def create_test_case(df, features):
    """Factory function to create a dynamic test case class with df and features."""

    class DynamicTestCase(unittest.TestCase):
        def test_scale_features(self):
            scaled_df = scale_features(df.copy(), features)
            for feature in features:
                self.assertAlmostEqual(scaled_df[feature].mean(), 0, places=6)
                self.assertAlmostEqual(scaled_df[feature].std(), 1, delta=0.1)

    # Set essential class attributes
    DynamicTestCase.__name__ = "DynamicTestCase"
    DynamicTestCase.__module__ = __name__

    return DynamicTestCase('test_scale_features')
