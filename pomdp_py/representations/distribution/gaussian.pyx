from pomdp_py.framework.basics cimport GenerativeDistribution
import numpy as np

# Check if scipy exists
import importlib
scipy_spec = importlib.util.find_spec("scipy")
if scipy_spec is not None:
    import scipy
    import scipy.stats
else:
    raise ImportError("scipy not found."\
                      "Requires scipy.stats.multivariate_normal to use Gaussian")
    

cdef class Gaussian(GenerativeDistribution):

    """Note that Gaussian distribution is defined over a vector of numerical variables,
    but not a State variable."""

    def __init__(self, mean, cov):
        """
        mean (list): mean vector
        cov (list): covariance matrix
        """
        self._mean = mean
        self._cov = cov

    @property
    def mean(self):
        return self._mean

    @property
    def covariance(self):
        return self._cov

    @property
    def cov(self):
        return self.covariance

    @property
    def sigma(self):
        return self.covariance

    def __getitem__(self, value):
        print(value)
        print(np.array(self._mean))
        print(np.array(self._cov))
        return scipy.stats.multivariate_normal(value,
                                               np.array(self._mean),
                                               np.array(self._cov))
            
    def __setitem__(self, value, prob):
        # It isn't supported to arbitrarily change a value in
        # a Gaussian distribution to have a given probability
        raise NotImplementedError("Gaussian does not support density assignment.")

    def __hash__(self):
        return hash(tuple(self._mean))

    def __eq__(self, other):
        if not isinstance(other, Gaussian):
            return False
        else:
            return self._mean == other.mean\
                and self._cov == other.cov

    def mpe(self):
        # The mean is the MPE
        return self._mean

    def random(self, n=1):
        d = len(self._mean)
        Xstd = np.random.randn(n, d)
        X = np.dot(Xstd, scipy.linalg.sqrtm(self._cov)) + self._mean
        if n == 1:
            return X.tolist()[0]
        else:
            return X.tolist()
