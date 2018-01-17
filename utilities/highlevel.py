"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""
import warnings

import numpy as np
from .vectorop import ravel_to_matrix as rtm, dummycode


def transform(X, factors, get_model, method: str, y=None):
    if method == "raw" or method is None:
        return X
    if not factors or factors == "full" or not isinstance(factors, int):
        factors = np.prod(X.shape[1:])
        if method == "lda":
            factors -= 1

    method = method.lower()

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=factors, whiten=True)
    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        model = LDA(n_components=factors)
    elif method == "ica":
        from sklearn.decomposition import FastICA as ICA
        model = ICA(n_components=factors)
    elif method == "cca":
        from sklearn.cross_decomposition import CCA
        model = CCA(n_components=factors)
    elif method == "pls":
        from sklearn.cross_decomposition import PLSRegression as PLS
        model = PLS(n_components=factors)
        if str(y.dtype)[:3] not in ("flo", "int"):
            y = dummycode(y, get_translator=False)
    else:
        raise ValueError("Method {} unrecognized!".format(method))

    latent = model.fit_transform(rtm(X), y)

    if isinstance(latent, tuple):
        latent = latent[0]
    return (latent, model) if get_model else latent


def image_to_array(imagepath):
    """Opens an image file and returns it as a NumPy array of pixel values"""
    from PIL import Image
    return np.array(Image.open(imagepath))


def image_sequence_to_array(imageroot, outpath=None, generator=False):
    """Opens and merges an image sequence into a 3D tensor"""
    import os

    flz = os.listdir(imageroot)

    print("Merging {} images to 3D array...".format(len(flz)))
    if not generator:
        ar = np.stack([image_to_array(imageroot + image) for image in sorted(flz)])
        if outpath is not None:
            try:
                ar.dump(outpath)
            except MemoryError:
                warnings.warn("OOM, skipped array dump!", ResourceWarning)
            else:
                print("Images merged and dumped to {}".format(outpath))
        return ar
    for image in sorted(flz):
        yield image_to_array(imageroot + image)
