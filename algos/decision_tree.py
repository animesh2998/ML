import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # number of features to consider at each split
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(self.n_features, n_features)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_gini = 1.0
        best_idx, best_thresh = None, None
        for idx in feature_indices:
            thresholds = np.unique(X[:, idx])
            for threshold in thresholds:
                left_indices = X[:, idx] <= threshold
                right_indices = X[:, idx] > threshold
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])
                gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / n_samples
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = threshold
        return best_idx, best_thresh

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(value=predicted_class)

        if depth < (self.max_depth if self.max_depth is not None else np.inf):
            idx, threshold = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                if len(y_left) > 0 and len(y_right) > 0:
                    node = self.Node(feature_index=idx, threshold=threshold)
                    node.left = self._grow_tree(X_left, y_left, depth + 1)
                    node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _traverse_tree(self, x, node):
        if node.value is not None and node.feature_index is None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
