from .learningtable import LearningTable
from ..features import embedding_factory, transformation_factory


class Problem:

    def __init__(self, data, labels, paramset):
        self.__dict__.update({"learning": None})
        self.subsets = []
        self.config = {"labels": labels, "paramset": paramset,
                       "transformation": None, "embedding": None}
        self.add_subset("learning", LearningTable(data, labels, paramset))
        self.raw = self["learning"].raw

    def set_transformation(self, transformation, no_features=None, **kw):
        if isinstance(transformation, str):
            transformation = transformation_factory(transformation, no_features, **kw)
        if not transformation.fitted:
            transformation.fit(self["learning"].X, self["learning"].Y)
        for subset in self.subsets:
            self[subset].set_transformation(transformation, apply=True)
        self.config["transformation"] = transformation

    def set_embedding(self, embeddim, **kw):
        if not isinstance(embeddim, int):
            raise ValueError("Embeddim should be an integer (0 for onehot, >0 for actual embedding)")
        embeddim = embedding_factory(embeddim, **kw)
        if not embeddim.fitted:
            embeddim.fit(self.raw.Y)
        for subset in self.subsets:
            self[subset].set_embedding(embeddim, apply=True)

    def dropna(self):
        for subset in self.subsets:
            self[subset].dropna(inplace=True)

    def batch_stream(self, batchsize=32, infinite=True, randomize=True, subset="learning"):
        return self[subset].batch_stream(batchsize, infinite, randomize)

    def split_new_subset_from(self, source_subset, new_subset, split_ratio, randomize=True):
        if source_subset not in self.subsets:
            raise KeyError(f"No such subset: {source_subset}")
        if new_subset in self.subsets:
            raise RuntimeError(f"Subset already exists: {new_subset}")
        source = self.__dict__.pop(source_subset)
        new, old = source.split(split_ratio, randomize)
        self.add_subset(source_subset, old)
        self.add_subset(new_subset, new)
        self.subsets.append(new_subset)

    def add_subset(self, name, learningtable):
        if learningtable in self.subsets:
            raise RuntimeError(f"Subset already exists: {learningtable}")
        self.__dict__[name] = learningtable
        self.subsets.append(name)

    def table(self, subset, shuffle=True):
        tab = self[subset]
        if shuffle:
            tab.shuffle()
        return tab.X, tab.Y

    def __getitem__(self, item):
        if item not in self.subsets:
            raise KeyError(f"No such subset: {item}")
        return self.__dict__[item]
