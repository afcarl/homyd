import numpy as np
import pandas as pd

from .reparse import reparse_txt
from ..utilities.textual import to_ngrams, to_wordarray


def txt(source, ngram=None, **kw):
    if ("\\" in source or "/" in source) and len(source) < 200:
        with open(source, mode="r", encoding=kw.pop("encoding", "utf-8")) as opensource:
            source = opensource.read()
    source = reparse_txt(source, **kw)
    return to_ngrams(np.array(list(source)), ngram) if ngram else to_wordarray(source)


def massive_txt(source, bsize, ngram=1, **kw):
    from .reparse import dehungarize
    with open(source, mode="r", encoding=kw.pop("coding", "utf-8")) as opensource:
        chunk = opensource.read(n=bsize)
        if not chunk:
            raise StopIteration("File ended")
        if kw.pop("dehungarize"):
            chunk = dehungarize(chunk)
        if kw.pop("endline_to_space"):
            chunk = chunk.replace("\n", " ")
        if kw.pop("lower"):
            chunk = chunk.lower()
        chunk = to_ngrams(np.ndarray(list(chunk)), ngram)
        yield chunk


def csv(path, header=1, skiprows=None, skip_footer=0, **kw):
    """Extracts a data learning_table from a file, returns X, Y, header"""
    df = pd.read_csv(path, sep=kw.pop("sep", "\t"), lineterminator=kw.pop("end", "\n"),
                     encoding=kw.pop("encoding", "utf-8"), skiprows=skiprows, skip_footer=skip_footer,
                     header=max(0, header-1))
    return df


def xlsx(source, header=1, sheetname=0, skiprows=None, skip_footer=0, **kw):
    df = pd.read_excel(source, sheetname, max(0, header-1), skiprows, skip_footer)
    return df


def parse_source(source, indeps, headers, **kw):
    if not isinstance(source, str):
        raise TypeError("String (filepath) required!")
    if source.lower()[-4:] in (".csv", ".txt"):
        return csv(source, indeps, headers, **kw)
    elif source.lower()[-4:] in (".xls", "xlsx"):
        return xlsx(source, indeps, headers, **kw)
    else:
        raise ValueError(f"Unsupported source: {source}")
