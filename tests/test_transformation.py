import unittest

import numpy as np

from homyd.transformation import (
    Rescaling, Standardization, Normalization, Decorrelation, Whitening
)


class TestSimple(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(100).reshape(10, 10)

    def _fit_transform(self, model):
        model.fit(self.X)
        return model.apply(self.X)

    def test_rescaling(self):
        tr = self._fit_transform(Rescaling(5, 8))
        self.assertEqual(tr.min(), 5)
        self.assertEqual(tr.max(), 8)

    def test_standardization(self):
        tr = self._fit_transform(Standardization())
        self.assertAlmostEqual(tr.mean(), 0., 1)
        self.assertAlmostEqual(tr.std(), 1., 1)

    def test_normalization(self):
        tr = self._fit_transform(Normalization())
        self.assertTrue(np.allclose(np.linalg.norm(tr, axis=1), np.ones(len(tr))))

    def test_decorrelation(self):
        tr = self._fit_transform(Whitening())
        self.assertTrue(np.allclose(np.cov(tr.T), np.eye(tr.shape[-1])))


if __name__ == '__main__':
    unittest.main()
