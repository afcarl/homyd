import abc

import numpy as np


def _ensure_vector(array):
    ar = array.squeeze()
    if ar.ndim != 1:
        raise RuntimeError("Supplied argument must be a vector!")
    return ar


class EmbeddingBase(abc.ABC):

    def __init__(self):
        self._categories = None
        self.fitted = False

    def fresh(self):
        return self.__class__()

    @abc.abstractmethod
    def translate(self, X):
        raise NotImplementedError

    def fit(self, labels):
        labels = _ensure_vector(labels)
        self._categories = np.sort(np.unique(labels)).tolist()  # type: list

    @property
    def outputs_required(self):
        return 1

    @abc.abstractmethod
    def apply(self, Y):
        _ensure_vector(Y)
        if not self.fitted:
            raise RuntimeError("Not yet fitted! Call fit() first!")

    def __str__(self):
        return self.__class__.__name__.lower()


class Dummycode(EmbeddingBase):

    @np.vectorize
    def apply(self, Y):
        return self._categories.index(Y)

    @np.vectorize
    def translate(self, X):
        return self._categories[X]


class OneHot(Dummycode):

    def __init__(self, yes=1., no=0., dtype="float32"):
        super().__init__()
        self._dim = 0
        self._yes = yes
        self._no = no
        self._eye = None
        self.dtype = dtype

    @property
    def outputs_required(self):
        return self._dim

    def fresh(self):
        return self.__class__(self._yes, self._no)

    def translate(self, prediction: np.ndarray):
        if prediction.ndim == 2:
            prediction = np.argmax(prediction, axis=1)

        return super().translate(prediction)

    def fit(self, labels):
        super().fit(labels)

        self._dim = len(self._categories)

        self._eye = np.zeros((self._dim, self._dim)) + self._no
        np.fill_diagonal(self._eye, self._yes)
        self._eye = self._eye.astype(self.dtype)

        self.fitted = True
        return self

    def apply(self, Y):
        dummy = super().apply(Y)
        return self._eye[dummy]


class Embedding(Dummycode):

    name = "embedding"

    def __init__(self, embeddim, dtype="float32"):
        super().__init__()
        self._dim = embeddim
        self._embedments = None
        self.dtype = dtype

    def fresh(self):
        return self.__class__(self._dim)

    def translate(self, prediction: np.ndarray):
        from ..utilities.vectorop import euclidean
        if prediction.ndim > 2:
            raise RuntimeError("<prediction> must be a matrix!")

        dummycodes = [np.argmin(euclidean(pred, self._embedments)) for pred in prediction]
        return super().translate(dummycodes)

    def fit(self, labels):
        super().fit(labels)
        self._embedments = np.random.randn(len(self._categories), self._dim).astype(self.dtype)
        self.fitted = True

    @np.vectorize
    def apply(self, Y):
        dummy = super().apply(Y)
        return self._embedments[dummy]


def embedding_factory(embedding, **kw):
    if embedding == "onehot":
        return OneHot(**kw)
    if embedding == "dummycode":
        return Dummycode()
    if isinstance(embedding, int):
        if embedding < 1:
            raise ValueError(f"Embedding dimension invalid: {embedding}")
        return Embedding(embedding, **kw)
    raise ValueError(f"Embedding specification not understood: {embedding}")
