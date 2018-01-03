import numpy as np
import pandas as pd

from ..features.transformation import Transformation
from ..features.embedding import EmbeddingBase


class LearningTable:

    def __init__(self, df: pd.DataFrame, labels, paramset=None, transformation=None, embedding=None):
        self.raw = df
        self.X = self.Y = None
        self._stream = None
        self.transformation = transformation
        self.embedding = embedding
        self.labels = [labels] if isinstance(labels, str) else labels
        self.paramset = ([p for p in df.columns if p not in labels]
                         if paramset is None else paramset)
        self.reset()
        self.set_transformation(transformation, apply=False)
        self.set_embedding(embedding, apply=True)

    @classmethod
    def from_xlsx(cls, source, labels, paramset=None, transformation=None, embedding=None, **pdkw):
        df = pd.read_excel(source, **pdkw)
        return cls(df, labels, paramset, transformation, embedding)

    @property
    def N(self):
        return len(self.X)

    @property
    def neurons_required(self):
        return self.X.shape[1:], self.Y.shape[1:]

    def dropna(self, inplace=False):
        if not inplace:
            df = self.raw.dropna(subset=self.labels + self.paramset)
            return self.__class__(df, *self._copy_configuration())
        self.raw.dropna(subset=self.labels + self.paramset, inplace=True)
        self.reset()

    def reset(self):
        self.X = self.raw[self.paramset].as_matrix()
        self.Y = self.raw[self.labels].as_matrix()
        if self.transformation:
            self.X = self.transformation.apply(self.X, self.Y)
        if self.embedding:
            self.Y = self.embedding.apply(self.Y)
        return self

    def set_transformation(self, trobj, apply=True):
        if trobj is None:
            self.transformation = None
            self.reset()
            return
        if not isinstance(trobj, Transformation):
            raise ValueError("Supplied argument is not a Transformation object!")
        if not trobj.fitted:
            trobj.fit(self.X, self.Y)
        self.transformation = trobj
        if apply:
            self.reset()
        return self

    def set_embedding(self, embobj, apply=True):
        if embobj is None:
            self.embedding = None
            self.reset()
            return
        if not isinstance(embobj, EmbeddingBase):
            raise ValueError("Supplied argument is not an EmbeddingBase object!")
        if not embobj.fitted:
            embobj.fit(self.Y)
        self.embedding = embobj
        if apply:
            self.reset()
        return self

    def batch_stream(self, size, infinite=True, randomize=True):
        arg = self._get_indices_for_resampling(randomize)
        while 1:
            if randomize:
                np.random.shuffle(arg)
            for start in range(0, len(self), size):
                yield self.X[start:start+size], self.Y[start:start+size]
            if not infinite:
                break

    def split(self, alpha, randomize=True):
        if self.X is None:
            raise RuntimeError("Empty learning table!")
        if alpha <= 0. or alpha >= 1.:
            raise ValueError("Alpha should be between 1. and 0.")
        alpha = int(alpha * self.N)
        arg = self._get_indices_for_resampling(randomize)
        rarg, larg = arg[:alpha], arg[alpha:]
        df1, df2 = self.raw.iloc[rarg], self.raw.iloc[larg]
        return (self.__class__(df1, *self._copy_configuration()),
                self.__class__(df2, *self._copy_configuration()))

    def copy(self):
        return self.__class__(self.raw.copy(), *self._copy_configuration())

    def merge(self, other):
        self.raw = self.raw.append(other.raw)
        print("Transformation and embedding is reset to None!")
        self.transformation = self.embedding = None
        self.reset()
        return self

    def shuffle(self):
        arg = self._get_indices_for_resampling(randomize=True)
        self.X, self.Y = self.X[arg], self.Y[arg]
        return self

    def _get_indices_for_resampling(self, randomize=True):
        arg = np.arange(self.N)
        if randomize:
            np.random.shuffle(arg)
        return arg

    def _copy_configuration(self):
        return self.labels[:], self.paramset[:], self.transformation.fresh(), self.embedding.fresh()

    def __iter__(self):
        self._stream = self.batch_stream(32, infinite=False, randomize=True)
        return self._stream

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.X, self.Y)[item]
        if isinstance(item, str):
            return {"x": self.X, "y": self.Y}[item.lower()]

    def __len__(self):
        return len(self.X)
