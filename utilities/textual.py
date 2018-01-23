import warnings

import numpy as np


def pull_text(source, encoding="utf8"):
    return open(source, encoding=encoding).read()


def to_ngrams(txt, ngram):
    txar = np.array(list(txt))
    N = txar.shape[0]
    if N % ngram != 0:
        warnings.warn(
            "Text length not divisible by ngram. Disposed some elements at the end of the seq!",
            RuntimeWarning)
        txar = txar[:-(N % ngram)]
    txar = txar.reshape(N // ngram, ngram)
    return ["".join(ng) for ng in txar] if ngram > 1 else np.ravel(txar)


def to_wordarray(txt):
    return np.array(txt.split(" "))
