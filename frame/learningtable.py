import numpy as np
import pandas as pd

from ..features.transformation import Transformation
from ..features.embedding import EmbeddingBase


class LearningTable:

    def __init__(self, df: pd.DataFrame, features: list, paramset=None):
        self.raw = df
        self.X = self.Y = None
        self.transformation = None
        self.embedding = None
        self.features = features
        self.paramset = ([p for p in df.columns if p not in features]
                         if paramset is None else paramset)
        self.reset()

    def reset(self):
        self.X = self.raw[self.paramset].as_matrix()
        self.Y = self.raw[self.features].as_matrix()
        if self.transformation:
            self.X = self.transformation.apply(self.X, self.Y)
        if self.embedding:
            self.Y = self.embedding.apply(self.X, self.Y)

    def set_transformation(self, trobj):
        if trobj is None:
            self.transformation = None
            self.reset()
            return
        if not isinstance(trobj, Transformation):
            raise ValueError("Supplied argument is not a Transformation object!")
        if not trobj.fitted:
            trobj.fit(self.X, self.Y)
        self.reset()

    def set_embedding(self, embobj):
        if embobj is None:
            self.embedding = None
            self.reset()
            return
        if not isinstance(embobj, EmbeddingBase):
            raise ValueError("Supplied argument is not an EmbeddingBase object!")
        if not embobj.fitted:
            embobj.fit(self.Y)
        self.reset()

    def batch_stream(self, size, infinite=True, randomize=True):
        arg = self._get_arguments_for_subsampling(randomize)
        while 1:
            if randomize:
                np.random.shuffle(arg)
            for start in range(0, len(self), size):
                yield self.X[start:start+size], self.Y[start:start+size]
            if not infinite:
                break

    def _get_arguments_for_subsampling(self, randomize):
        arg = np.arange(self.N)
        if randomize:
            np.random.shuffle(arg)
        return arg

    def split(self, alpha):
        if alpha <= 0. or alpha >= 1.:
            raise ValueError("Alpha should be between 1. and 0.")
        if isinstance(alpha, float):
            alpha = int(alpha * self.N)
        arg = self._get_arguments_for_subsampling(randomize=True)
        if self.X is None:
            raise RuntimeError("Empty learning table!")
        X1, X2 = self.X[arg[:alpha]], self.X[arg[alpha:]]
        Y1, Y2 = (self.Y[arg[:alpha]], self.Y[arg[alpha:]])\
            if self.Y is not None else (None, None)
        lt1, lt2 = self.__class__((X1, Y1)), self.__class__((X2, Y2))
        return lt1, lt2

    def subsample(self, m, randomize=True):
        if isinstance(m, float):
            m = int(m * self.N)
        arg = self._get_arguments_for_subsampling(randomize)
        return self.__class__(self.X[arg[:m]], self.Y[arg[:m]])

    def copy(self):
        return self.__class__(self.raw.copy(), self.features[:], self.paramset[:])

    def __iter__(self):
        return self

    def __next__(self):
        for member in (self.X, self.Y):
            yield member

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.X, self.Y)[item]
        if isinstance(item, str):
            return self.__dict__[item]

    def __len__(self):
        return len(self.X)
