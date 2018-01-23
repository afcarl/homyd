import warnings

import numpy as np


class Filterer:

    def __init__(self, X, Y, header):
        if len(header) != X.shape[1] + Y.shape[1]:
            raise RuntimeError("Invalid header for X and Y!")
        if isinstance(header, np.ndarray):
            header = header.tolist()
        self.X = X
        self.Y = Y
        self.raw = X.copy(), Y.copy()
        self.indeps = X.shape[1]
        self.header = header

    def _feature_name_to_index(self, featurename):
        if isinstance(featurename, int):
            if self.indeps < featurename:
                raise ValueError("Invalid feature number. Max: " + str(self.indeps))
            return featurename
        elif not featurename:
            return 0
        if featurename not in self.header:
            raise ValueError("Unknown feature: {}\nAvailable: {}"
                             .format(featurename, self.header))
        if self.header.count(featurename) > 1:
            warnings.warn("Ambiguity in feature selection! Using first occurence!",
                          RuntimeWarning)
        return self.header.index(featurename)

    def select_feature(self, featurename):
        featureno = self._feature_name_to_index(featurename)
        return self.Y[:, featureno]

    def filter_by(self, featurename, *selections):
        filterno = self._feature_name_to_index(featurename)
        mask = np.ones(len(self.Y), dtype=bool)
        for sel in selections:
            mask &= self.Y[:, filterno] == sel
        self.X, self.Y = self.X[mask], self.Y[mask]

    def revert(self):
        self.X, self.Y = self.raw
