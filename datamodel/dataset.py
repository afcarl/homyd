import pandas as pd

from .learningtable import LearningTable
from ..embedding import EmbeddingBase, embedding_factory
from ..transformation import Transformation, transformation_factory


class Dataset:

    __slots__ = ["_subsets", "_config", "_raw", "_shapes"]

    def __init__(self, data: LearningTable, shape=()):
        self._raw = data.raw
        self._subsets = dict()
        self._config = {}
        self._shapes = data.shapes if shape else shape
        self.add_subset("learning", data)

    @classmethod
    def from_xlsx(cls, source, labels, paramset=None, **pdkw):
        df = pd.read_excel(source, **pdkw)
        return cls(LearningTable(df, labels, paramset))

    @classmethod
    def from_multidimensional(cls, X, Y):
        return cls(LearningTable.from_multidimensional(X, Y), shape=X.shape[1:])

    @property
    def shapes(self):
        return self._shapes

    def dropna(self):
        for subset in self._subsets:
            self[subset].dropna(inplace=True)

    def batch_stream(self, batchsize=32, infinite=True, randomize=True, subset="learning"):
        return self[subset].batch_stream(batchsize, infinite, randomize)

    def split_new_subset_from(self, source_subset, new_subset, split_ratio, randomize=True):
        if source_subset not in self._subsets:
            raise KeyError(f"No such subset: {source_subset}")
        if new_subset in self._subsets:
            raise RuntimeError(f"Subset already exists: {new_subset}")
        source = self._subsets.pop(source_subset)
        new, old = source.split(split_ratio, randomize)
        self.add_subset(source_subset, old)
        self.add_subset(new_subset, new)

    def add_subset(self, name, learningtable):
        if learningtable in self._subsets:
            raise RuntimeError(f"Subset already exists: {learningtable}")
        self._subsets[name] = learningtable

    def merge_into_learning(self, subset):
        if subset not in self._subsets:
            raise ValueError(f"No such subset: {subset}")
        self._subsets["learning"].merge(self._subsets[subset])

    def table(self, subset, shuffle=True):
        tab = self._subsets[subset]
        if shuffle:
            tab.shuffle()
        return tab.X, tab.Y

    def __getitem__(self, item):
        if item not in self._subsets:
            raise KeyError(f"No such subset: {item}")
        return self._subsets[item]


class FlatDataset(Dataset):

    def __init__(self, data: LearningTable, validation_split=0.1, transformation=None, embedding=None):
        super().__init__(data)
        self._config.update({"transformation": None, "embedding": None})
        self.dropna()
        if validation_split:
            self.split_new_subset_from("learning", "validation", validation_split, randomize=True)
        if transformation is not None:
            self.set_transformation(transformation)
        if embedding is not None:
            self.set_embedding(embedding)

    @classmethod
    def from_multidimensional(cls, X, Y):
        return cls(LearningTable.from_multidimensional(X, Y))

    def set_transformation(self, transformation, **kw):
        if isinstance(transformation, str):
            transformation = transformation_factory(transformation, **kw)
        elif isinstance(transformation, tuple):
            transformation = transformation_factory(*transformation, **kw)
        elif isinstance(transformation, Transformation) and not transformation.fitted:
            transformation.fit(self["learning"].X, self["learning"].Y)
        self._config["transformation"] = transformation

    def set_embedding(self, embedding, **kw):
        if not isinstance(embedding, EmbeddingBase):
            embedding = embedding_factory(embedding, **kw)
        if not embedding.fitted:
            embedding.fit(self._subsets["learning"].Y)
        self._config["embedding"] = embedding

    def batch_stream(self, batchsize=32, infinite=True, randomize=True, subset="learning"):
        return map(self._apply_features_on_data, super().batch_stream(batchsize, infinite, randomize, subset))

    def table(self, subset, shuffle=True):
        return self._apply_features_on_data(super().table(subset, shuffle))

    def _apply_features_on_data(self, data):
        x, y = data
        trf, emb = self._config["transformation"], self._config["embedding"]
        return x if trf is None else trf.apply(x), y if emb is None else emb.apply(y)
