import warnings

import numpy as np

from .dataset import Dataset
from ..features import embedding_factory
from ..utilities.const import log


class BasicSequence(Dataset):

    type = "basicsequence"

    def __init__(self, source, time, cross_val=0.2, **kw):
        if isinstance(source, str):
            source = parser.txt(source, ngram=kw.pop("ngram", 1))
        X, Y = [], []
        for start in range(0, len(source)-time-2):
            X.append(source[start:start+time])
            Y.append(source[start+time+1])
        super().__init__(
            (np.array(X), np.array(Y)), cross_val, labels=0, headers=None, **kw
        )

    def reset_data(self, shuff, transform, trparam=None):
        pass

    @property
    def neurons_required(self):
        pass

    def concatenate(self, other):
        pass


class LazySequence:

    def __init__(self, X, timestep, cross_val=0.2):

        def chop_up_to_timesteps():
            newN = self.data.shape[0] // timestep
            if self.data.shape[0] % timestep != 0:
                warnings.warn("Trimming data to fit into timesteps!", RuntimeWarning)
                self.data = self.data[:self.data.shape[0] - (self.data.shape[0] % timestep)]
            newshape = newN, timestep
            print("Reshaping from: {} to: {}".format(self.data.shape, newshape))
            self.data = self.data.reshape(*newshape)

        self.timestep = timestep
        self._crossval = cross_val

        self.data = X
        chop_up_to_timesteps()
        self.n_testing = int(self.data.shape[0] * cross_val)
        self.N = self.data.shape[0]

    @property
    def neurons_required(self):
        return (self.timestep, self.data.shape[-1]), self.data.shape[-1]

    def batchgen(self, bsize):
        index = 0
        epochs_passed = 0
        while 1:
            start = bsize * index
            end = start + bsize

            slc = self.data[start:end]

            X, y = slc[:, :-1, :], slc[:, -1]

            if end > self.N:
                warnings.warn("\nEPOCH PASSED!", RuntimeWarning)
                epochs_passed += 1
                log("{} MASSIVE_SEQUENCE EPOCHS PASSED!".format(epochs_passed))

                index = 0
            index += 1

            yield X, y

    def primer(self):
        from random import randrange
        primer = self.data[randrange(self.N)]
        return primer.reshape(1, *primer.shape)


def _translate(preds, use_proba, embedding):
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]
    elif preds.ndim == 2:
        pass
    else:
        raise NotImplementedError("Oh-oh...")

    if use_proba:
        preds = np.log(preds)
        e_preds = np.exp(preds)
        preds = e_preds / e_preds.sum()
        preds = np.random.multinomial(1, preds, size=preds.shape)

    human = embedding.translate(preds)
    return human
