import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _get_n_features(self, n_features):
        if isinstance(self.max_features, int):
            return self.max_features
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.trees = []
        n_feats = self._get_n_features(n_features)
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feats,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(row).most_common(1)[0][0] for row in tree_preds]
        return np.array(y_pred)
