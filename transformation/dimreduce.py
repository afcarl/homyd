from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA as _PCA, FastICA as _ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA

from .abstract import Transformation
from..utilities.vectorop import unravel_matrix


def _get_model_type(modeltype):
    if isinstance(modeltype, BaseEstimator):
        return modeltype
    return {"pca": _PCA, "lda": _LDA, "ica": _ICA}[modeltype]


class Dimreduce(Transformation):

    __slots__ = ["model", "_fitted"]

    def __init__(self, modeltype, features, **kw):
        super().__init__()
        self.model = _get_model_type(modeltype)(n_components=features, **kw)
        self._fitted = None

    def fit(self, X, Y=None):
        X = self._save_shape_and_ravel_to_matrix(X)
        self.model.fit(X, Y)
        self._fitted = True

    def apply(self, X, Y=None):
        X = self._ensure_shape(X)
        return unravel_matrix(self.model.transform(X, Y), self._input_shape)

    def fitted(self):
        return self._fitted
