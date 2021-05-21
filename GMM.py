import numpy as np
from scipy import stats

class GMM(object):
    """
    This class allows to compute the EM algorithm for gaussians on the given data.
    """
    def __init__(self, k=2):
        self.k = k
    
    def fit(self, X, tol=1e-4, max_iters=100):
        # The data:
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        
        # Initializing:
        self._init()
        num_iters = 0
        ll = tol*2
        previous_ll = 0
        
        # EM iterations:
        while ll-previous_ll > tol and num_iters < max_iters:
            previous_ll = self.loglikelihood()
            self._fit()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
        
    def _init(self):
        # Init mixture mus/sigmas:
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.pi = np.ones(self.k)/self.k
        self.z = np.asmatrix(np.empty((self.m, self.k), dtype=float))
    
    def _fit(self):
        self._e_step()
        self._m_step()
        
    def _e_step(self):
        # Considring every sample:
        for i in range(self.m):
            # Sum of gaussian pdf contributions:
            den = 0
            for j in range(self.k):
                num = stats.multivariate_normal.pdf(
                    self.data[i, :], 
                    self.mean_arr[j].A1, 
                    self.sigma_arr[j],
                    allow_singular=True
                ) * self.pi[j]
                den += num
                self.z[i, j] = num
                
            # Normalizing we get the latent distribution estimation:
            self.z[i, :] /= den
            assert self.z[i, :].sum() - 1 < 1e-4
            
    def _m_step(self):
        # Every gaussian must be re-estimated...
        for j in range(self.k):
            # ...considering the updated sample contributions:
            const = self.z[:, j].sum()
            self.pi[j] = 1/self.m * const
            
            # Vanilla mean and covariance matrix computation:
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.z[i, j])
                _sigma_j += self.z[i, j] * (
                    (self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :])
                )
            
            # Updated distributions:
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
    
    def loglikelihood(self):
        # Accumulating the likelihoods from all samples and gaussians:
        ll = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                # Probabiity value:
                tmp += stats.multivariate_normal.pdf(
                    self.data[i, :], 
                    self.mean_arr[j, :].A1, 
                    self.sigma_arr[j, :],
                    allow_singular=True
                ) * self.pi[j]
             
            # The log of the computed likelihood:
            ll += np.log(tmp) 
        return ll
