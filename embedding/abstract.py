import numpy as np

from ..utilities.sanity import ensure_dimensionality


class EmbeddingBase:

    __slots__ = ["_categories"]

    def __init__(self):
        self._categories = None  # type: np.ndarray

    @property
    def fitted(self):
        return self._categories is not None

    def fresh(self):
        return self.__class__()

    def translate(self, X):
        raise NotImplementedError

    def fit(self, labels):
        labels = ensure_dimensionality(labels, 1)
        self._categories = np.sort(np.unique(labels))

    @property
    def outputs_required(self):
        return 1

    def apply(self, Y):
        ensure_dimensionality(Y, 1)
        if not self.fitted:
            raise RuntimeError("Not yet fitted! Call fit() first!")

    def __str__(self):
        return self.__class__.__name__.lower()
