import numpy as np

from .simple import Decorrelation


class Whitening(Decorrelation):

    def fit(self, X, Y=None):
        super().fit(X)
        U, S, V = self._decomposition
        self._weights *= np.sqrt(S)**-1.


class MahalanobisWhitening(Whitening):

    def fit(self, X, Y=None):
        super().fit(X)
        self._weights = self._decomposition[-1] @ self._weights
