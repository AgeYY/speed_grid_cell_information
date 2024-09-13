import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from gpflow.utilities import positive
from gpflow import Parameter
from grid_cell.gaussian_wishart.src.likelihoods.BaseWishartLikelihood import WishartLikelihoodBase

class FactorizedWishartLikelihood(WishartLikelihoodBase):

    def __init__(self, D, nu, n_factors, A=None, **kwargs):
        """
        :param D (int) Covariance matrix dimension
        :param nu (int) Degrees of freedom
        :param n_factors (int) Dimensionality of factorized covariance matrix.
        :param R (int) Number of monte carlo samples used to approximate reparameterized gradients.
        :param inverse (bool) Use inverse Wishart Process if true, otherwise standard Wishart Process.
        """
        super().__init__(D, nu, additive_noise=True, **kwargs)  # todo: check likelihoods' specification of dimensions
        self.D = D
        self.cov_dim = n_factors
        self.nu = nu
        self.n_factors = n_factors

        # no such thing as a non-full scale matrix in this case
        self.A = A if A is not None else Parameter(np.ones((D, n_factors)), transform=positive(), dtype=tf.float64)
        gpflow.set_trainable(self.A, True)

        # all factored models are approximate models
        self.q_sigma2inv_conc = Parameter(1. * np.ones(self.D), transform=positive(), dtype=tf.float64) # 0.1
        self.q_sigma2inv_rate = Parameter(0.1 * np.ones(self.D), transform=positive(), dtype=tf.float64) # 0.0001

    def make_gaussian_components(self, F, Y):
        """
        In the case of the factored covariance matrices, we should never directly represent the covariance or precision
        matrix. The following computation makes use of the matrix inversion formula(s).
        Function written entirely by Creighton Heaukulani and Mark van der Wilk.

        :param F: (R, N, K, nu2) - the (samples of the) matrix of GP outputs.
        :param Y: (N, D)
        :return:
        """
        AF = tf.einsum('kl,ijlm->ijkm', self.A, F)  # (S, N, D, nu*2) # todo: why did the doc say 2nu here? the final shape is still nu as only l (n_factors) is summed and multiplied out
        n_samples = tf.shape(F)[0]  # could be 1 if making predictions
        dist = tfd.Gamma(self.q_sigma2inv_conc, self.q_sigma2inv_rate)
        sigma2_inv = dist.sample([n_samples])  # (S, D)
        sigma2_inv = tf.clip_by_value(sigma2_inv, 1e-8, np.inf)
        sigma2 = sigma2_inv ** -1.0

        # if tf.is_tensor(Y):
        #     Y.set_shape([None, self.D])

        y_Sinv_y = tf.reduce_sum((Y ** 2.0) * sigma2_inv[:, None, :], axis=2)  # (S, N)

        if self.model_inverse:
            # no inverse necessary for Gaussian exponent
            SAF = sigma2[:, None, :, None] * AF  # (S, N, D, nu2)
            faSaf = tf.matmul(AF, SAF, transpose_a=True)  # (S, N, nu2, nu2)
            faSaf = tf.linalg.set_diag(faSaf, tf.linalg.diag_part(faSaf) + 1.0)
            L = tf.linalg.cholesky(faSaf)  # (S, N, nu2, nu2)
            log_det_cov = tf.reduce_sum(tf.math.log(sigma2), axis=1)[:, None] \
                           - 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (S, N)
                # note: first line had negative because we needed log(s2^-1) and then another negative for |precision|

            yaf_or_afy = tf.einsum('jk,ijkl->ijl', Y, AF)  # (S, N, nu2)
            yt_inv_y = y_Sinv_y + tf.reduce_sum(yaf_or_afy ** 2, axis=2)  # (S, N)

        else:
            # Wishart case: take the inverse to create Gaussian exponent
            SinvAF = sigma2_inv[:, None, :, None] * AF  # (S, N, D, nu^2)
            faSinvaf = tf.matmul(AF, SinvAF, transpose_a=True)  # (S, N, nu2, nu2), computed efficiently, O(S * N * n_factors^2 * D)
            faSinvaf = tf.linalg.set_diag(faSinvaf, tf.linalg.diag_part(faSinvaf) + 1.0)
            L = tf.linalg.cholesky(faSinvaf)  # (S, N, nu2, nu2)
            log_det_cov = tf.reduce_sum(tf.math.log(sigma2), axis=1)[:, None] \
                           + 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=2)  # (S, N), just log |AFFA + S| (no sign)

            ySinvaf_or_afSinvy = tf.einsum('jk,ijkl->ijl', Y, SinvAF)  # (S, N, nu2)
            L_solve_ySinvaf = tf.linalg.triangular_solve(L, ySinvaf_or_afSinvy[:, :, :, None], lower=True)  # (S, N, nu2, 1)
            ySinvaf_inv_faSinvy = tf.reduce_sum(L_solve_ySinvaf ** 2.0, axis=(2, 3))  # (S, N)
            yt_inv_y = y_Sinv_y - ySinvaf_inv_faSinvy  # (S, N), this is Y^time_window (AFFA + S)^-1 Y

        return log_det_cov, yt_inv_y
