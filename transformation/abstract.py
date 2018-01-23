from ..utilities.vectorop import ravel_to_matrix


class Transformation:

    def __init__(self):
        self._input_shape = None

    @property
    def fitted(self):
        raise NotImplementedError

    def _ensure_shape(self, X):
        return X.reshape(len(X), *self._input_shape)

    def _save_shape_and_ravel_to_matrix(self, X):
        self._input_shape = X.shape[1:]
        return ravel_to_matrix(X)

    def fit(self, X, Y=None):
        raise NotImplementedError

    def apply(self, X, Y=None):
        raise NotImplementedError

    @property
    def __str__(self):
        return self.__class__.__name__.lower()
