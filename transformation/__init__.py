from .abstract import Transformation
from .dimreduce import Dimreduce
from .simple import Standardization, Decorrelation, Normalization, Rescaling
from .whitening import Whitening, MahalanobisWhitening


transformation_types = {k.lower(): v for k, v in locals().items() if k != "Transformation"}
transformation_types.update({k[:5]: v for k, v in transformation_types.items()})
dimreduce_aliases = {"lda": Dimreduce, "ica": Dimreduce, "pca": Dimreduce}
transformation_types.update(dimreduce_aliases)


def transformation_factory(name, features=None, **kw):
    if name not in transformation_types:
        raise ValueError(f"Unknown transformation: {name}")
    if name not in dimreduce_aliases:
        return transformation_types[name](**kw)
    return Dimreduce(name, features, **kw)
