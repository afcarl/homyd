import unittest

import numpy as np

from homyd.embedding import embedding_factory

EMBDIM = 5
INPUT = np.array(list("aaabbcbcbaabcbabc"))
DUMMY = (np.array(list(map(int, "11122323211232123"))) - 1)
ONEHOT = np.eye(3, 3)[DUMMY]

np.random.seed(1337)


class TestEmbedding(unittest.TestCase):

    def test_dummycoding_forward(self):
        dc = embedding_factory("dummycoding")
        dc.fit(INPUT)
        dcd = dc.apply(INPUT)
        self.assertTrue(np.all(dcd == DUMMY))

    def test_onehot_forwad(self):
        oh = embedding_factory("onehot")
        oh.fit(INPUT)
        ohd = oh.apply(INPUT)
        self.assertTrue(np.all(ohd == ONEHOT))

    def test_dummycoding_backward(self):
        dc = embedding_factory("dummycoding")
        dc.fit(INPUT)
        back = dc.translate(DUMMY)
        self.assertTrue(np.all(back == INPUT))

    def test_onehot_backward(self):
        oh = embedding_factory("onehot")
        oh.fit(INPUT)
        back = oh.translate(ONEHOT)
        self.assertTrue(np.all(back == INPUT))

    def test_embedding_forward_backward(self):
        emb = embedding_factory(EMBDIM)
        emb.fit(INPUT)
        embd = emb.apply(INPUT)
        self.assertEqual(embd.ndim, 2)
        N, d = embd.shape
        self.assertEqual(N, len(INPUT))
        self.assertEqual(d, EMBDIM)
        reverse = emb.translate(embd)
        self.assertTrue(np.all(reverse == INPUT))


if __name__ == '__main__':
    unittest.main()
