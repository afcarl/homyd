import unittest

import numpy as np

from homyd.embedding import embedding_factory


INPUT = np.array(list("aaabbcbcbaabcbabc"))[:, None]
DUMMY = (np.array(list(map(int, "11122323211232123"))) - 1)[:, None]
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


if __name__ == '__main__':
    unittest.main()
