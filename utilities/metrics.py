import numpy as np


class MetricFunction:

    def apply(self, array1, array2):
        raise NotImplementedError

    def __call__(self, array1, array2):
        return self.apply(array1, array2)

    def __str__(self):
        return self.__class__.__name__.lower()


class _Manhattan(MetricFunction):

    def apply(self, array1, array2):
        return np.sum(np.abs(array1 - array2), axis=-1)


class _Euclidean(MetricFunction):

    def apply(self, array1, array2):
        return np.sqrt(np.sum(np.square(array1 - array2), axis=-1))


class _Haversine(MetricFunction):
    """Distance of two points on the surface of Earth given their GPS (WGS) coordinates"""

    def apply(self, array1, array2):
        assert array1.shape == array2.shape, "Please supply two arrays of coordinate-pairs!"
        R = 6367.  # Approximate radius of Mother Earth in kms
        np.radians(array1, out=array1)
        np.radians(array2, out=array2)
        lon1, lat1 = array1[..., 0], array1[..., 1]
        lon2, lat2 = array2[..., 0], array2[..., 1]
        dlon = lon1 - lon2
        dlat = lat1 - lat2
        d = np.sin(dlat / 2.) ** 2. + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2.
        e = 2. * np.arcsin(np.sqrt(d))
        return e * R


manhattan = _Manhattan()
euclidean = _Euclidean()
haversine = _Haversine()

metric_functions = locals().copy()
metric_functions.pop("np", None)
