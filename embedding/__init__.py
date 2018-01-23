from .embedding import Embedding
from .onehot import OneHot
from .dummycode import Dummycode
from .abstract import EmbeddingBase


def embedding_factory(embedding, **kw):
    if isinstance(embedding, int):
        if embedding < 1:
            raise ValueError(f"Embedding dimension invalid: {embedding}")
        return Embedding(embedding, **kw)
    error = ValueError(f"Embedding specification not understood: {embedding}")
    if not isinstance(embedding, str):
        raise error
    if embedding == "onehot":
        return OneHot(**kw)
    if embedding[:5] == "dummy":
        return Dummycode()
    raise error
