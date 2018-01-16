import numpy as np
import pandas as pd


class LearningTable:

    __slots__ = ["raw", "X", "Y", "labels", "paramset"]

    def __init__(self, df: pd.DataFrame, labels, paramset=None):
        self.raw = df
        self.X = self.Y = None
        self.labels = [labels] if isinstance(labels, str) else labels
        self.paramset = ([p for p in df.columns if p not in labels]
                         if paramset is None else paramset)
        self.reset()

    @classmethod
    def from_xlsx(cls, source, labels, paramset=None, **pdkw):
        df = pd.read_excel(source, **pdkw)
        return cls(df, labels, paramset)

    @property
    def shapes(self):
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
            raise RuntimeError("Empty learning learning_table!")
        if alpha <= 0. or alpha >= 1.:
            raise ValueError("Alpha should be between 1. and 0.")
        alpha = int(alpha * len(self))
        arg = self._get_indices_for_resampling(randomize)
        rarg, larg = arg[:alpha], arg[alpha:]
        df1, df2 = self.raw.iloc[rarg], self.raw.iloc[larg]
        return (self.__class__(df1, *self._copy_configuration()),
                self.__class__(df2, *self._copy_configuration()))

    def copy(self):
        return self.__class__(self.raw.copy(), *self._copy_configuration())

    def merge(self, other):
        self.raw = self.raw.append(other.raw)
        self.reset()
        return self

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
        return self.labels[:], self.paramset[:]

    def __iter__(self):
        return self.batch_stream(32, infinite=False, randomize=True)

    def __len__(self):
        return len(self.X)
