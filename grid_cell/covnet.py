import tensorflow as tf
import tensorflow_probability as tfp

class CovMLP(tf.keras.Model):
    """
    A multi-layer perceptron model for estimating the covariance matrix.

    Attributes:
        input_dim (int): Dimension of the input vector.
        output_dim (int): Dimension of the output vector. The output is the lower triangular matrix L,
                          which is the Cholesky decomposition of the precision matrix.
    """

    def __init__(self, input_dim, output_dim):
        super(CovMLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense4 = tf.keras.layers.Dense(units=output_dim * (output_dim + 1) // 2)
        # self.dense4 = tf.keras.layers.Dense(units=output_dim * (output_dim + 1) // 2)

    def call(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs (tf.Tensor, shape [batch_size, input_dim]): Input data. Input can be obtained by concatenating predicted_mean (f) and label, i.e. tf.concat([f, label], axis=-1).

        Returns:
            tf.Tensor: Lower triangular matrix L (shape [batch_size, output_dim, output_dim]),
                       the Cholesky decomposition of the precision matrix.
        """
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        hidden = self.dense3(hidden)
        output = self.dense4(hidden)
        L = tfp.math.fill_triangular(output)
        return L

    def predict_precision(self, inputs):
        """
        Predict the precision matrix using the network.

        Args:
            inputs (tf.Tensor, shape [batch_size, input_dim]): Input data.

        Returns:
            tf.Tensor: Precision matrix (shape [batch_size, output_dim, output_dim]).
        """
        L = self(inputs)
        Lt = tf.linalg.matrix_transpose(L)
        LLt = tf.matmul(L, Lt)
        return LLt

    def predict_covariance(self, inputs):
        """
        Predict the covariance matrix using the network.

        Args:
            inputs (tf.Tensor, shape [batch_size, input_dim]): Input data.

        Returns:
            tf.Tensor: Covariance matrix (shape [batch_size, output_dim, output_dim]).
        """
        L = self(inputs)
        Linv = tf.linalg.inv(L)
        Linv_t = tf.linalg.matrix_transpose(Linv)
        LLt_inv = tf.matmul(Linv_t, Linv)
        return LLt_inv

def custom_loss(f, response, label=None, lambda_val=None, sigma=None):
    """
    Custom loss function.

    Args:
        f (tf.Tensor, shape [batch_size, n_features]): Predicted neural response.
        response (tf.Tensor, shape [batch_size, n_features]): True neural response.
        label (tf.Tensor, shape [batch_size, n_labels]): Input data (optional).
        lambda_val (float): Weight of the contrastive loss term (optional).
        sigma (float): Bandwidth of the RBF kernel (optional).

    Returns:
        Function: A function that calculates the custom loss given the lower-triangular matrix L.
    """
    n_batch = f.shape[0]

    def loss(L):
        term1 = - tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(L)))) / n_batch
        Lt = tf.linalg.matrix_transpose(L)
        term2 = tf.reduce_sum(tf.square(tf.matmul(Lt, tf.expand_dims(response - f, axis=-1)))) / n_batch

        term3 = 0.0
        if lambda_val is not None and label is not None:
            term3 = contrastive_loss(label, L, lambda_val, sigma) / n_batch

        return term1 + term2 + term3

    return loss

def contrastive_loss(label, L, lambda_val, sigma):
    """
    Contrastive loss function.

    Args:
        label (tf.Tensor, shape [batch_size, n_labels]): Input data.
        L (tf.Tensor, shape [batch_size, output_dim, output_dim]): Lower triangular matrix L.
        lambda_val (float): Weight of the contrastive loss term.
        sigma (float): Bandwidth of the RBF kernel.

    Returns:
        tf.Tensor: Calculated contrastive loss.
    """
    pairwise_differences = tf.expand_dims(label, 1) - tf.expand_dims(label, 0)
    pairwise_distances = tf.reduce_sum(tf.square(pairwise_differences), axis=2)
    kernel_values = tf.exp(-pairwise_distances / (2 * tf.square(sigma)))
    
    L_differences = tf.expand_dims(L, 1) - tf.expand_dims(L, 0)
    L_norms = tf.reduce_sum(tf.abs(L_differences), axis=[-2, -1])

    return lambda_val * tf.reduce_sum(kernel_values * L_norms)
