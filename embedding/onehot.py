import numpy as np

from .dummycode import Dummycode
from ..utilities.sanity import ensure_dimensionality


class OneHot(Dummycode):

    __slots__ = ["_dim", "_yes", "_no", "_eye", "dtype"]

    def __init__(self, yes=1., no=0., dtype="float32"):
        super().__init__()
        self._dim = 0
        self._yes, self._no = yes, no
        self._eye = None
        self.dtype = dtype

    @property
    def outputs_required(self):
        return self._dim

    def fresh(self):
        return self.__class__(self._yes, self._no)

    def translate(self, prediction):
        ensure_dimensionality(prediction, 2)
        return super().translate(prediction.argmax(axis=1))

    def fit(self, labels):
        super().fit(labels)
        self._dim = len(self._categories)
        self._eye = np.zeros((self._dim, self._dim)) + self._no
        np.fill_diagonal(self._eye, self._yes)
        self._eye = self._eye.astype(self.dtype)
        return self

    def apply(self, Y):
        dummy = super().apply(Y)
        return self._eye[dummy]
