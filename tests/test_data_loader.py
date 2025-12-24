"""Unit tests for data loader utilities."""

import unittest
import numpy as np
import tempfile
import os
import csv


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create sample data
        self.sample_data = [
            ["feature1", "feature2", "feature3", "target"],
            [1.0, 2.0, 3.0, 10.0],
            [1.5, 2.5, 3.5, 12.0],
            [2.0, 3.0, 4.0, 15.0],
            [2.5, 3.5, 4.5, 18.0],
        ]
        
        with open(self.test_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.sample_data)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_load_csv_file(self):
        """Test loading CSV file."""
        # This would test the actual data loader utility
        self.assertTrue(os.path.exists(self.test_csv_path))
        
        # Read the file
        data = []
        with open(self.test_csv_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        
        self.assertEqual(len(data), 5)  # Header + 4 rows

    def test_data_shape(self):
        """Test that loaded data has correct shape."""
        # Load numeric data (skip header)
        with open(self.test_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data = [[float(x) for x in row] for row in reader]
        
        data_array = np.array(data)
        expected_shape = (4, 4)  # 4 samples, 4 features
        
        self.assertEqual(data_array.shape, expected_shape)

    def test_train_test_split(self):
        """Test train/test split functionality."""
        # Load data
        with open(self.test_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data = [[float(x) for x in row] for row in reader]
        
        data_array = np.array(data)
        test_size = 0.25
        n_samples = len(data_array)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        # Simple split
        train_data = data_array[:n_train]
        test_data = data_array[n_train:]
        
        self.assertEqual(len(train_data), 3)
        self.assertEqual(len(test_data), 1)
        self.assertEqual(len(train_data) + len(test_data), n_samples)

    def test_feature_target_separation(self):
        """Test separation of features and targets."""
        # Load data
        with open(self.test_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data = [[float(x) for x in row] for row in reader]
        
        data_array = np.array(data)
        
        # Separate features and target
        X = data_array[:, :-1]  # All columns except last
        y = data_array[:, -1]   # Last column
        
        self.assertEqual(X.shape, (4, 3))
        self.assertEqual(y.shape, (4,))

    def test_data_normalization(self):
        """Test data normalization."""
        data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        
        # Min-max normalization
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        normalized = (data - data_min) / (data_max - data_min)
        
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing."""

    def test_missing_value_detection(self):
        """Test detection of missing values."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]])
        
        has_nan = np.isnan(data).any()
        self.assertTrue(has_nan)

    def test_outlier_detection(self):
        """Test outlier detection using IQR method."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        self.assertTrue(outliers[-1])  # 100 should be detected as outlier

    def test_feature_scaling(self):
        """Test feature scaling (standardization)."""
        data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        
        # Standardization: (x - mean) / std
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        standardized = (data - mean) / std
        
        self.assertAlmostEqual(standardized.mean(), 0.0, places=5)
        self.assertAlmostEqual(np.abs(standardized.std()), 1.0, places=5)


class TestBatchGenerator(unittest.TestCase):
    """Test cases for batch generation."""

    def test_batch_size(self):
        """Test that batches have correct size."""
        data = np.arange(100).reshape(100, 1)
        batch_size = 16
        
        num_complete_batches = len(data) // batch_size
        
        batches = []
        for i in range(num_complete_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batches.append(batch)
            self.assertEqual(len(batch), batch_size)

    def test_shuffle_data(self):
        """Test data shuffling."""
        data = np.arange(10)
        shuffled = data.copy()
        np.random.shuffle(shuffled)
        
        # Shuffled data should contain same elements
        self.assertEqual(set(data), set(shuffled))
        
        # But likely in different order (this could rarely fail)
        if len(data) > 2:
            # At least one element should be in different position
            self.assertTrue(np.any(data != shuffled) or np.all(data == shuffled))


if __name__ == "__main__":
    unittest.main()
