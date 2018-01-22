import abc

import numpy as np


class Transformation(abc.ABC):

    __slots__ = []

    @abc.abstractmethod
    def fit(self, X, Y): raise NotImplementedError

    @abc.abstractmethod
    def apply(self, X, Y): raise NotImplementedError


class Standardization(Transformation):

    __slots__ = ["_mean", "_std"]

    def __init__(self):
        super().__init__()
        self._mean = None
        self._std = None

    def fit(self, X, Y=None):
        self._mean = X.mean(axis=0) if self._mean is None else self._mean
        self._std = X.std(axis=0) if self._std is None else self._std

    def apply(self, X, Y=None):
        return (X - self._mean) / self._std


class Decorrelation(Standardization):

    __slots__ = ["_decomposition", "_weights"]

    def __init__(self):
        super().__init__()
        self._decomposition = None
        self._weights = None

    def fit(self, X, Y=None):
        super().fit(X)
        self._decomposition = np.linalg.svd(X - self._mean)
        self._weights = self._decomposition[-1]

    def apply(self, X, Y=None):
        return X @ self._weights


class Whitening(Decorrelation):

    def fit(self, X, Y=None):
        super().fit(X)
        U, S, V = self._decomposition
        self._weights = V * np.sqrt(S)**-1


class Mahalanobis(Whitening):

    def fit(self, X, Y=None):
        super().fit(X)
        self._weights = self._decomposition[-1] @ self._weights
