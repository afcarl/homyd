import numpy as np

from ..utilities.vectorop import dummycode
from ..utilities.highlevel import transform


class Transformation:

    def __init__(self, **kw):
        self.model = None

    @property
    def fitted(self):
        return self.model is not None

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def fit(self, X, Y=None):
        self.model = transform(X, get_model=True, method=self.name, y=Y)[-1]

    def apply(self, X, Y=None):
        return self.model.transform(X)[..., :self.factors]

    def __call__(self, X, Y=None):
        return self.apply(X, Y)


class Standardization(Transformation):
    name = "std"

    def fit(self, X, Y=None):
        from ..utilities.vectorop import standardize
        self.model = standardize(X, return_factors=True)[1]

    def apply(self, X: np.ndarray, Y=None):
        mean, std = self.model
        return (X - mean) / std


class PLS(Transformation):
    name = "pls"

    def apply(self, X: np.ndarray, Y=None):
        Y = dummycode(Y, get_translator=False)
        ret = self.model.transform(X, Y)[0]
        return ret


def transformation_factory(name, number_of_features=None, **kw):
    name = name.lower()[:5]
    if number_of_features is None:
        if name not in ("std", "stand"):
            raise RuntimeError("Please supply the number_of_features argument!")
    exc = {"std": Standardization,
           "stand": Standardization,
           "pls": PLS}
    if name not in exc:
        return Transformation(name, number_of_features, **kw)
    return exc[name](number_of_features, **kw)
