import numpy as np
import tensorflow as tf
from gpflow.likelihoods.base import ScalarLikelihood

class WishartLikelihoodBase(ScalarLikelihood):
    """
    Abstract class for all Wishart Processes likelihoods.
    Class written by Creighton Heaukulani and Mark van der Wilk, and is adapted for gpflow 2.
    """
    def __init__(self, D, nu, R=10, model_inverse=True, additive_noise=True, multiple_observations=False, **kwargs):
        """
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use inverse Wishart Process if true, otherwise standard Wishart Process.
        :param additive_noise (bool) Use additive white noise model likelihood if true.
        :param multiple_observations (bool) At each timepoint, multiple observations are available. (i.e. the data is TxNxD)
        """
        super().__init__()  # todo: check likelihoods' specification of dimensions
        self.D = D
        self.nu = nu
        self.R = R
        self.model_inverse = model_inverse
        self.additive_noise = additive_noise
        self.multiple_observations = multiple_observations

    def variational_expectations(self, f_mean, f_cov, Y):
        """
        Calculate log p(Y | variational parameters)

        :param f_mean: (N, D*nu), mean parameters of latent GP points F
        :param f_cov: (N, D*nu), covariance parameters of latent GP points F
        :param Y: (N, D) or (T,N,D), observations
        :return logp: (N,), log probability density of the data.
        where N is the minibatch size, D the covariance matrix dimension and nu the degrees of freedom
        """
        _, latent_dim = f_mean.shape
        N = tf.shape(Y)[0]

        # Produce R samples of F (latent GP points at the input locations X).
        # TF automatically differentiates through this.
        W = tf.dtypes.cast(tf.random.normal(shape=[self.R, N, latent_dim]), tf.float64)
        f_sample = W * f_cov**0.5 + f_mean

        f_sample = tf.reshape(f_sample, [self.R, N, self.cov_dim, -1])

        # compute the mean of the likelihood
        logp = self._log_prob(f_sample, Y) #(N,)
        return logp

    def _log_prob(self, F, Y): # (R,N) -> (N)

        if self.multiple_observations:
            logps = []
            for t in range(Y.shape[1]):
                Y_t = Y[:,t,:]
                logps.append(tf.math.reduce_mean(self._scalar_log_prob(F,Y_t), axis=0)) # (R,N) -> (N,)
            return tf.math.reduce_sum(logps, axis=0) # (T,N) -> (N,)
        else:
            return tf.math.reduce_mean(self._scalar_log_prob(F, Y), axis=0) # take mean across D dimension

    def _scalar_log_prob(self, F, Y):
        """
        Log probability of covariance matrix Sigma_n = A F_n F_n^time_window A^time_window
        Implements equation (5) in Heaukulani-van der Wilk
        :param F (N,D,D) the (sampled) matrix of GP outputs
        :param Y (N,D) observations
        """
        D = tf.dtypes.cast(self.D, tf.float64)
        log_det_cov, yt_inv_y = self.make_gaussian_components(F,Y) # (R, N), (R,N)
        log_p = - 0.5 * D * np.log(2*np.pi) - 0.5*log_det_cov - 0.5*yt_inv_y # (R,N)
        return log_p # (R,N)

    def make_gaussian_components(self, F, Y):
        """
        Returns components used in the Gaussian density kernels
        Abstract method, should be implemented by concrete classes.
        :param F (R, N, D, ),  the (sampled) matrix of GP outputs.
        :param Y (N,D) observations
        """
        raise NotImplementedError