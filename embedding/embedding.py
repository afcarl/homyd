import numpy as np


class EmbeddingBase:

    name = ""

    def __init__(self, **kw):
        self._categories = None
        self.dtype = kw.get("dtype", "float32")
        self.fitted = False
        self.dim = 1

    def fresh(self):
        return self.__class__(floatX=self.dtype)

    def translate(self, X):
        raise NotImplementedError

    def fit(self, labels):
        self._categories = np.sort(np.unique(labels)).tolist()  # type: list

    @property
    def outputs_required(self):
        return self.dim

    def apply(self, Y):
        Y = np.squeeze(Y)
        if not self.fitted:
            raise RuntimeError("Not yet fitted! Call fit() first!")
        if Y.ndim != 1:
            raise RuntimeError("Y must be a vector!")

    def __str__(self):
        return self.__class__.__name__.lower()

    def __call__(self, X):
        return self.apply(X)


class Dummycode(EmbeddingBase):

    name = "dummycode"

    @np.vectorize
    def apply(self, Y):
        return self._categories.index(Y)

    @np.vectorize
    def translate(self, X):
        return self._categories[X]


class OneHot(EmbeddingBase):

    name = "onehot"

    def __init__(self, yes=1., no=0.):
        super().__init__()
        self._yes = yes
        self._no = no
        self.dim = 0

    def fresh(self):
        return self.__class__(self._yes, self._no)

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        if prediction.ndim == 2:
            prediction = np.argmax(prediction, axis=1)
            if dummy:
                return prediction

        return super().translate(prediction)

    def fit(self, labels):
        super().fit(labels)

        self.dim = len(self._categories)

        self._embedments = np.zeros((self.dim, self.dim)) + self._no
        np.fill_diagonal(self._embedments, self._yes)
        self._embedments = self._embedments.astype(self.dtype)

        self.fitted = True
        return self


class Embedding(EmbeddingBase):

    name = "embedding"

    def __init__(self, embeddim):
        super().__init__()
        self.dim = embeddim

    def fresh(self):
        return self.__class__(self.dim)

    def translate(self, prediction: np.ndarray, dummy: bool=False):
        from ..utilities.vectorop import euclidean
        if prediction.ndim > 2:
            raise RuntimeError("<prediction> must be a matrix!")

        dummycodes = [np.argmin(euclidean(pred, self._embedments)) for pred in prediction]
        if dummy:
            return dummycodes

        return super().translate(dummycodes)

    def fit(self, labels):
        super().fit(labels)
        cats = len(self._categories)

        self._embedments = np.random.randn(cats, self.dim)
        self.fitted = True


def embedding_factory(embedding, **kw):
    if embedding == "onehot":
        return OneHot(**kw)
    if embedding == "dummycode":
        return Dummycode(**kw)
    if isinstance(embedding, int):
        if embedding < 1:
            raise ValueError(f"Embedding dimension invalid: {embedding}")
        return Embedding(embedding)
    raise ValueError(f"Embedding specification not understood: {embedding}")
