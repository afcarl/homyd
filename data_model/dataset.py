from data_model.general import LearningTable
from embedding.embedding import EmbeddingBase
from transformation import Transformation

from ..features import embedding_factory, transformation_factory


class Dataset:

    __slots__ = ["subsets", "config", "raw"]

    def __init__(self, data: LearningTable, transformation=None, embedding=None, dropna=True):
        self.raw = data.raw
        self.subsets = dict()
        self.config = {"transformation": None, "embedding": None}
        self.add_subset("learning", data)
        if dropna:
            self.dropna()
        if transformation is not None:
            self.set_transformation(transformation)
        if embedding is not None:
            self.set_embedding(embedding)

    def set_transformation(self, transformation, **kw):
        if isinstance(transformation, str):
            transformation = transformation_factory(transformation, **kw)
        elif isinstance(transformation, tuple):
            transformation = transformation_factory(*transformation, **kw)
        elif isinstance(transformation, Transformation) and not transformation.fitted:
            transformation.fit(self["learning"].X, self["learning"].Y)
        self.config["transformation"] = transformation

    def set_embedding(self, embedding, **kw):
        if isinstance(embedding, int):
            embedding = embedding_factory(embedding, **kw)
        if isinstance(embedding, EmbeddingBase) and not embedding.fitted:
            embedding.fit(self.raw.Y)
        self.config["embedding"] = embedding

    def dropna(self):
        for subset in self.subsets:
            self[subset].dropna(inplace=True)

    def batch_stream(self, batchsize=32, infinite=True, randomize=True, subset="learning"):
        return map(self._apply_features_on_data, self[subset].batch_stream(batchsize, infinite, randomize))

    def split_new_subset_from(self, source_subset, new_subset, split_ratio, randomize=True):
        if source_subset not in self.subsets:
            raise KeyError(f"No such subset: {source_subset}")
        if new_subset in self.subsets:
            raise RuntimeError(f"Subset already exists: {new_subset}")
        source = self.subsets.pop(source_subset)
        new, old = source.split(split_ratio, randomize)
        self.add_subset(source_subset, old)
        self.add_subset(new_subset, new)

    def add_subset(self, name, learningtable):
        if learningtable in self.subsets:
            raise RuntimeError(f"Subset already exists: {learningtable}")
        self.subsets[name] = learningtable

    def table(self, subset, shuffle=True):
        tab = self.subsets[subset]
        if shuffle:
            tab.shuffle()
        return self._apply_features_on_data([tab.X, tab.Y])

    def _apply_features_on_data(self, data):
        x, y = data
        trf, emb = self.config["transformation"], self.config["embedding"]
        return x if trf is None else trf.apply(x), y if emb is None else emb.apply(y)

    def __getitem__(self, item):
        if item not in self.subsets:
            raise KeyError(f"No such subset: {item}")
        return self.subsets[item]
