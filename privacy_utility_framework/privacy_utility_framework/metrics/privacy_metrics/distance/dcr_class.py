import numpy as np
from scipy.spatial import distance
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator
from sklearn.metrics import pairwise_distances

# DONE


class DCRCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 original_name: str = None, synthetic_name: str = None,
                 distance_metric: str = 'euclidean',
                 weights: np.ndarray = None):
        """
        Initializes the DCRCalculator with datasets and a specified distance metric.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            original_name (str, optional): Name for the original dataset (default: None).
            synthetic_name (str, optional): Name for the synthetic dataset (default: None).
            distance_metric (str): The metric for calculating distances ('euclidean', 'cityblock', etc.).
            weights (np.ndarray, optional): Array of weights for each feature in the datasets.
        """
        # Initialize the superclass with datasets and settings
        super().__init__(original, synthetic, distance_metric=distance_metric,
                         original_name=original_name, synthetic_name=synthetic_name)

        # Validate that distance_metric is set
        if distance_metric is None:
            raise ValueError("Parameter 'distance_metric' is required in DCRCalculator.")

        # Define distance metric and feature weights for calculations
        self.distance_metric = distance_metric
        # print(f"original data shape is {self.original.data.shape[1]}")
        self.weights = weights if weights is not None else np.ones(self.original.data.shape[1])

    def evaluate(self) -> float:
        """
        Computes the Distance to Closest Record (DCR) between the synthetic and original datasets.

        Returns:
            float: The average minimum distance from each synthetic record to the closest original record.
        """
        # Retrieve transformed and normalized data
        original = self.original.transformed_normalized_data
        synthetic = self.synthetic.transformed_normalized_data

        # Apply feature weights to both datasets
        # weighted_original_data = original * self.weights
        # weighted_synthetic_data = synthetic * self.weights
        weighted_original_data = original
        weighted_synthetic_data = synthetic

        # Compute pairwise distances between synthetic and original data
        dists = distance.cdist(weighted_synthetic_data, weighted_original_data, metric=self.distance_metric)

        # Find and average the minimum distances for each synthetic record
        min_distances = np.min(dists, axis=1)

        return np.mean(min_distances)

        # #----------------------------2024-11-22, shiyu add, from TabDDPM framework
        # X_fake = np.asarray(synthetic)
        # X_real = np.asarray(original)
        # # Computing pair-wise distances between real and synthetic
        # dist_rf = pairwise_distances(X_fake, Y=X_real, metric='l2', n_jobs=-1)
        # # Computing first and second smallest nearest neighbour distances between real and synthetic
        # smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
        # smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]
        # min_dist_rf = np.array([i[0] for i in smallest_two_rf])
        # return np.median(min_dist_rf)

    def set_metric(self, metric: str):
        """
        Sets or updates the distance metric for the DCR calculation.

        Parameters:
            metric (str): The distance metric to use in DCR calculation.
        """
        self.distance_metric = metric
