import tensorflow as tf
import numpy as np
from gkr_example.gpr import GP_Fitter
from sklearn.model_selection import train_test_split
from sklearn.covariance import GraphicalLasso
from global_setting import *

def create_train_test_split_indices(num_samples, test_size=0.33):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return train_indices, test_indices

class Kernel_Cov(tf.Module):
    def __init__(self, n_input, n_output, circular_period=None, diag_factor=1e-6):
        self.circular_period = circular_period
        self.n_input = n_input
        self.n_output = n_output
        self.diag_factor = diag_factor
        initial_L = tf.linalg.band_part(tf.random.normal((self.n_input, self.n_input), dtype=FLOAT_TYPE), -1, 0)
        self.kernel_prec_L = tf.Variable(initial_L, dtype=FLOAT_TYPE)

    def fit(self, r, x):
        self.r = np.array(r, dtype=FLOAT_TYPE)
        self.x = np.array(x, dtype=FLOAT_TYPE)

    # def compute_gram(self, r, x):

    #     if x.shape[1] != self.n_input:
    #         raise ValueError('The input dimension does not match the kernel dimension')
    #     self.r = np.array(r, dtype=FLOAT_TYPE)
    #     self.x = np.array(x, dtype=FLOAT_TYPE)
    #     self.gram = self._compute_gram(self.r, 0)  # (n_sample, n_neuron, n_neuron)

    def predict_cov(self, query, pred_batch_size=1000):
        query = np.array(query, dtype=FLOAT_TYPE)
        L = tf.linalg.band_part(self.kernel_prec_L, -1, 0)
        kernel_prec_mat = tf.matmul(L, tf.transpose(L)) # (n_input, n_input)
        n_sample = self.x.shape[0]
        n_query = query.shape[0]

        cov_pred = tf.zeros((n_query, self.n_output, self.n_output), dtype=FLOAT_TYPE)
        kernel_sum_total = tf.zeros(n_query, dtype=FLOAT_TYPE)
        for start in range(0, n_sample, pred_batch_size):
            end = min(start + pred_batch_size, n_sample)

            diff = self.x[start:end, None, :] - query[None, ...] # (pred_batch_size, n_query, n_input)
            diff = self._circular_diff(diff, self.circular_period)

            diff_prec = tf.einsum('ijk,kl,ijl->ij', diff, kernel_prec_mat, diff) # (pred_batch_size, n_query)
            kernel_matrix = tf.exp(-diff_prec)
            kernel_sum = tf.reduce_sum(kernel_matrix, axis=0)

            gram = self._compute_gram(self.r[start:end], 0) # (pred_batch_size, n_output, n_output)

            cov_pred += tf.einsum('qi,qjk->ijk', kernel_matrix, gram)
            kernel_sum_total += kernel_sum

        cov_pred = cov_pred / kernel_sum_total[:, None, None]
        cov_pred = cov_pred + tf.eye(self.n_output, dtype=FLOAT_TYPE) * self.diag_factor
        return cov_pred

    @staticmethod
    def _compute_gram(response_train, response_pred):
        diff = response_train - response_pred
        gram = tf.einsum('ij,ik->ijk', diff, diff)
        return gram

    @staticmethod
    def _circular_diff(diff, periods):
        """
        Modify the difference based on the circular periods.

        Parameters:
        diff (Tensor [n_sample, n_query, n_input]):
        periods: The periods for circularity, which can be None, a scalar, or a list.
          - If None, no modification is made.
          - If a scalar, the same period is applied to all dimensions.
          - If a list, each dimension is modified according to the corresponding period in the list (None for no modification).

        Returns:
        Tensor: The modified differences.
        """
        if periods is None: return diff

        if np.isscalar(periods):
            # Apply the same period to all dimensions
            return tf.math.square(tf.math.sin(np.pi * diff / periods))

        # Apply different periods for each dimension
        for i, period in enumerate(periods):
            if period is not None:
                diff[..., i] = tf.math.square(tf.math.sin(np.pi * diff[..., i] / period))
        return diff

class GKR_Fitter():
    def __init__(self, n_input, n_output, circular_period=None, fit_valid_split=0.3, learning_rate=0.1, n_epochs=100, gpr_params={}, cov_fit_batch_size=3000, kernel_params={}):
        self.n_input = n_input
        self.n_output = n_output
        self.circular_period = circular_period
        self.fit_valid_split = fit_valid_split
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.gpr_mean = GP_Fitter(circular_period=self.circular_period, **gpr_params)
        self.kc = Kernel_Cov(self.n_input, self.n_output, circular_period, **kernel_params)
        self.cov_fit_batch_size = cov_fit_batch_size

    def fit(self, r, x):
        self.r = np.array(r, dtype=FLOAT_TYPE)
        self.x = np.array(x, dtype=FLOAT_TYPE)

        ##########
        # Fit the mean
        ##########
        self.gpr_mean.fit(self.r, self.x)
        r_pred, _ = self.gpr_mean.predict(self.x)

        ##########
        # Fit the covariance
        ##########
        r_ = self.r - r_pred # shift the mean to 0

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        dataset = tf.data.Dataset.from_tensor_slices((r_, self.x)).batch(self.cov_fit_batch_size)

        print('Fitting the covariance...')

        epoch_loss_list = []

        for ne in range(self.n_epochs):
            print(f'Epochs: {ne+1}/{self.n_epochs}')
            epoch_loss = 0  # Initialize a variable to accumulate loss for the epoch
            for i, (r_batch, x_batch) in enumerate(dataset):
                print(f'Batch: {i+1}/{ int( len(r_)/ self.cov_fit_batch_size ) }')

                r_train, r_valid, x_train, x_valid = train_test_split(r_batch.numpy(), x_batch.numpy(), test_size=0.33)
                with tf.GradientTape() as tape:
                    self.kc.fit(r_train, x_train)
                    cov_pred = self.kc.predict_cov(x_valid)
                    loss = - gaussian_log_likelihood(r_valid, cov_pred)
                gradients = tape.gradient(loss, self.kc.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.kc.trainable_variables))
                epoch_loss += loss.numpy()

            epoch_loss_list.append(epoch_loss)
            print(f'Training epoch loss: {epoch_loss}')

        self.kc.fit(r_, self.x)
        return epoch_loss_list

    def predict(self, query, return_cov=True, with_GLASSO=None):
        '''
        we found with_GLASSO is usually unnecessary. It will make the result better if the result itself is extremely worse, but even if using with_GLASSO, the result is not good.
        '''
        query_ = query.astype(FLOAT_TYPE)
        r_pred, _ = self.gpr_mean.predict(query_)
        if return_cov:
            cov_pred = self.kc.predict_cov(query_).numpy()
            if with_GLASSO is not None:
                cov_glasso_matrices = np.zeros_like(cov_pred)
                n_sample = cov_pred.shape[0]

                # Apply Graphical LASSO to each covariance matrix
                for i in range(n_sample):
                    glasso = GraphicalLasso(alpha=with_GLASSO, assume_centered=True)
                    glasso.fit(cov_pred[i])
    
                    cov_glasso_matrices[i] = glasso.covariance_
                cov_pred = cov_glasso_matrices
        else:
            cov_pred = None

        return r_pred, cov_pred

@tf.function
def gaussian_log_likelihood(r, cov_pred, diag_factor=1e-5):
    diag = tf.eye(cov_pred.shape[-1], dtype=FLOAT_TYPE) * diag_factor
    chol = tf.linalg.cholesky(cov_pred + diag)
    log_det = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)

    sol = tf.linalg.triangular_solve(chol, tf.expand_dims(r, -1), lower=True)
    quadratic_term = tf.reduce_sum(tf.square(sol), axis=[1, 2])

    log_likelihood = -0.5 * tf.reduce_mean(log_det + quadratic_term)
    return log_likelihood
