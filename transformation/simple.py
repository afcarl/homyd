import numpy as np

from .transformation import Transformation


class Standardization(Transformation):

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, X, Y=None):
        self.mean = X.mean(axis=0) if self.mean is None else self.mean
        self.std = X.std(axis=0) if self.std is None else self.std

    def apply(self, X, Y=None):
        return (X - self.mean) / self.std


class Decorrelation(Standardization):

    def __init__(self):
        super().__init__()
        self.weights = None
        self.sigma = None

    def fit(self, X, Y=None):
        super().fit(X)
        _, self.sigma, self.weights = np.linalg.svd(X - self.mean)

    def apply(self, X, Y=None):
        return X.dot(self.weights)


class Whiten(Decorrelation):

    def __init__(self):
        super().__init__()
        self.param = None


class Mahalanobis(Decorrelation):

    def __init__(self):
        super().__init__()

    def fit(self, X, Y=None):
        super().fit(X)

