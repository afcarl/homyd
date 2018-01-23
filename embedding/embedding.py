import numpy as np

from .dummycode import Dummycode
from ..utilities.metrics import metric_functions, MetricFunction
from ..utilities.sanity import ensure_dimensionality


class Embedding(Dummycode):

    __slots__ = ["_dim", "_embedments", "dtype", "_metric"]

    def __init__(self, embeddim, metric="euclidean", dtype="float32"):
        super().__init__()
        self._dim = embeddim
        self._embedments = None
        self.dtype = dtype
        self._metric = metric if isinstance(metric, MetricFunction) else metric_functions[metric]

    def fresh(self):
        return self.__class__(embeddim=self._dim, metric=self._metric, dtype=self.dtype)

    def translate(self, prediction):
        ensure_dimensionality(prediction, 2)
        dummycodes = np.argmin(self._metric(prediction[:, None, :], self._embedments), axis=1)
        return super().translate(dummycodes)

    def fit(self, labels):
        super().fit(labels)
        self._embedments = np.random.randn(len(self._categories), self._dim).astype(self.dtype)

    def apply(self, Y):
        dummy = super().apply(Y)
        return self._embedments[dummy]
