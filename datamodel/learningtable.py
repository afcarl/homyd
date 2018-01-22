import numpy as np
import pandas as pd


class LearningTable:

    __slots__ = ["raw", "X", "Y", "labels", "paramset", "shape", "dtypes"]

    def __init__(self, df: pd.DataFrame, labels, paramset=None, dtypes=None):
        self.raw = df
        self.dtypes = (None, None) if dtypes is None else dtypes
        self.X = self.Y = None
        self.labels = [labels] if isinstance(labels, str) else labels
        self.paramset = ([p for p in df.columns if p not in labels] if paramset is None else paramset)
        self.reset()

    @classmethod
    def from_multidimensional(cls, X, Y, dtypes=None):
        flat = X.reshape(len(X), np.prod(X.shape[1:]))
        df = pd.DataFrame(data=flat)
        df["LABEL"] = Y.argmax(axis=1)
        return cls(df, labels=["LABEL"], dtypes=dtypes)

    @property
    def shapes(self):
        return self.X.shape[1:], self.Y.shape[1:]

    @property
    def header(self):
        return self.labels + self.paramset

    def dropna(self, inplace=False):
        if not inplace:
            df = self.raw.dropna(subset=self.labels + self.paramset)
            return self.__class__(df, *self._copy_configuration())
        self.raw.dropna(subset=self.labels + self.paramset, inplace=True)
        self.reset()

    def reset(self):
        self.X = self.raw[self.paramset].as_matrix()
        self.Y = self.raw[self.labels].as_matrix()
        self._set_dtypes()

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
            raise RuntimeError("Empty learning learning_table!")
        if alpha <= 0. or alpha >= 1.:
            raise ValueError("Alpha should be between 1. and 0.")
        alpha = int(alpha * len(self))
        arg = self._get_indices_for_resampling(randomize)
        df1, df2 = self.raw.iloc[arg[:alpha]], self.raw.iloc[arg[alpha:]]
        return (self.__class__(df1, *self._copy_configuration()),
                self.__class__(df2, *self._copy_configuration()))

    def copy(self):
        return self.__class__(self.raw.copy(), *self._copy_configuration())

    def merge(self, other):
        if len(other.header) != len(self.header) and \
                not all(left == right for left, right in zip(self.header, other.header)):
            err = f"Cannot merge learning tables with different headers:\n{self.header} != {other.header}!"
            raise RuntimeError(err)
        self.raw = self.raw.append(other.raw)
        self.reset()

    def shuffle(self):
        arg = self._get_indices_for_resampling(randomize=True)
        self.X, self.Y = self.X[arg], self.Y[arg]
        return self

    def _get_indices_for_resampling(self, randomize=True):
        arg = np.arange(len(self))
        if randomize:
            np.random.shuffle(arg)
        return arg

    def _copy_configuration(self):
        return self.labels[:], self.paramset[:], self.dtypes[:]

    def _set_dtypes(self):
        if self.dtypes[0] is not None and self.X.dtype != self.dtypes[0]:
            self.X = self.X.astype(self.dtypes[0])
        if self.dtypes[1] is not None and self.Y.dtype != self.dtypes[1]:
            self.Y = self.Y.astype(self.dtypes[1])

    def __len__(self):
        return len(self.X)
