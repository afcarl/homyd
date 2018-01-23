import numpy as np

from .abstract import Transformation
from ..utilities.vectorop import unravel_matrix


class Rescaling(Transformation):

    def __init__(self, newmin=None, newmax=1.):
        super().__init__()
        self._oldmin = self._oldmax = None
        self._min, self._max = newmin, newmax

    def fit(self, X, Y=None):
        self._save_shape_and_ravel_to_matrix(X)
        self._oldmin = X.min(axis=0)
        self._oldmax = X.max(axis=0)
        self._min = self._oldmin if self._min is None else self._min
        self._max = self._oldmax if self._max is None else self._max

    def apply(self, X, Y=None):
        X = self._ensure_shape(X)
        out = (X - self._oldmin) / self._oldmax
        out *= self._max
        out += self._min
        return unravel_matrix(out, self._input_shape)

    @property
    def fitted(self):
        return self._min is not None


class Normalization(Transformation):

    def __init__(self, factor=None, order=None):
        super().__init__()
        self._factor = 1. if factor is None else factor
        self._ord = order

    def fit(self, X, Y=None):
        self._save_shape_and_ravel_to_matrix(X)

    def apply(self, X, Y=None):
        X = self._ensure_shape(X)
        normed = (X / np.linalg.norm(X, axis=1, keepdims=True, ord=self._ord)) * self._factor
        return unravel_matrix(normed, self._input_shape)

    @property
    def fitted(self):
        return True


class Standardization(Transformation):

    def __init__(self):
        super().__init__()
        self._mean = None
        self._std = None

    @property
    def fitted(self):
        return self._mean is not None

    def fit(self, X, Y=None):
        X = self._save_shape_and_ravel_to_matrix(X)
        self._mean = X.mean(axis=0) if self._mean is None else self._mean
        self._std = X.std(axis=0) if self._std is None else self._std

    def apply(self, X, Y=None):
        X = self._ensure_shape(X)
        standardized = (X - self._mean) / self._std
        return unravel_matrix(standardized, self._input_shape)


class Decorrelation(Standardization):

    def __init__(self):
        super().__init__()
        self._decomposition = None
        self._weights = None

    def fit(self, X, Y=None):
        X = self._ensure_shape(X)
        super().fit(X)
        self._decomposition = np.linalg.svd(X - self._mean)
        self._weights = self._decomposition[-1]

    def apply(self, X, Y=None):
        X = self._ensure_shape(X)
        return unravel_matrix(X @ self._weights, self._input_shape)
