import numpy as np

from .abstract import EmbeddingBase
from ..utilities.sanity import ensure_dimensionality


class Dummycode(EmbeddingBase):

    __slots__ = ["_dim"]

    def __init__(self):
        super().__init__()
        self._dim = 1

    def apply(self, Y: np.ndarray):
        Y = ensure_dimensionality(Y, 1)
        super().apply(Y)
        return np.argwhere(Y[:, None] == self._categories[None, :])[:, 1]

    def translate(self, indices):
        ensure_dimensionality(indices, 1)
        return self._categories[indices.flat]
