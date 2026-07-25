"""
tests/test_validate_submission.py
===================================
Pytest and Unittest compatible validation test suite.
"""

import os
import sys
import unittest

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from validate_submission import (
    validate_feature_list,
    validate_inference_dataset,
    run_validation,
)

class TestSubmissionValidation(unittest.TestCase):
    def test_feature_list_cleanliness(self):
        tabular_features = validate_feature_list()
        self.assertEqual(len(tabular_features), 18)

    def test_inference_dataset_cleanliness(self):
        csv_path = validate_inference_dataset()
        self.assertTrue(os.path.exists(csv_path))

    def test_full_submission_validation(self):
        run_validation()

if __name__ == "__main__":
    unittest.main()
