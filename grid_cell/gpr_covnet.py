import os
import numpy as np
import hickle as hkl
import copy
from skopt.space import Real
from skopt import gp_minimize
from sklearn.model_selection import KFold
import concurrent.futures
import tensorflow as tf
import tensorflow_probability as tfp
from grid_cell.manifold_fitter import GP_Fitter

from grid_cell.data_preprocessing import split_dataset, create_dataset
from grid_cell.covnet import CovMLP, custom_loss
from scipy.optimize import minimize
from grid_cell.util import convert_to_circular_form

class GPR_CovNet():
    """
    Gaussian Process Regression with Covariance Neural Network (GPR-CovNet).
    Combines Gaussian Process Regression (GPR) with a neural network for covariance estimation.

    Attributes:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        with_cov (bool): Whether to use covariance network.
        lambda_value (float): Regularization parameter for the loss function.
        scale (float): Scale parameter for the loss function.
        n_rand (int): Number of random search for lambda and scale.
        lambda_value_list (list): List of lambda values for random search.
        scale_value_list (list): List of scale values for random search.
    """

    def __init__(self, epochs=100, random_search_epochs=50, batch_size=256, with_cov=False, lambda_value=None, scale=0.15, n_rand=None, lambda_value_list=None, scale_value_list=None, circular_period=None, standardize=True):

        self.att_name = ['epochs', 'random_search_epochs', 'batch_size', 'with_cov', 'lambda_value', 'scale', 'gpr_fitter', 'input_dim', 'output_dim', 'circular_period', 'lambda_value_list', 'scale_value_list', 'n_rand', 'standardize'] # non tf attributes

        self.epochs = epochs
        self.random_search_epochs = random_search_epochs
        self.batch_size = batch_size
        self.with_cov = with_cov
        self.lambda_value = lambda_value
        self.scale = scale
        self.gpr_fitter = None
        self.circular_period = circular_period

        # config for covnet. This would be fitted if with_cov is true and fit() is called.
        self.covnet=None
        self.input_dim=None
        self.output_dim=None

        # random search for lambda and scale
        self.n_rand = n_rand
        self.lambda_value_list = lambda_value_list
        self.scale_value_list = scale_value_list

        self.standardize = standardize
        self.standardize_mean = None
        self.standardize_std = None

    def fit(self, response_train, label_train):
        """
        Fit the GPR model and optionally the covariance network.

        Args:
            response_train (array-like, shape [n_samples, n_features]): Training response data.
            label_train (array-like, shape [n_samples, n_labels]): Training label data.
        """
        if self.standardize:
            self.standardize_mean = np.mean(response_train, axis=0)
            self.standardize_std = np.std(response_train, axis=0)
            response_train = (response_train - self.standardize_mean) / self.standardize_std

        # Initialize and fit the GPR fitter
        self.gpr_fitter = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize)
        self.gpr_fitter.fit(response_train, label_train)

        if self.with_cov:
            predicted_rates, _ = self.gpr_fitter.predict(label_train)
            self._fit_covariance_network(predicted_rates, response_train, label_train, circular_period=self.circular_period)

    def predict(self, label_test, cov_to_prec=False):
        """
        Make predictions for test labels.

        Args:
            label_test (array-like, shape [n_samples, n_labels]): Test label data.

        Returns:
            tuple: Tuple containing predicted rates (shape [n_samples, n_features])
                   and optionally covariance matrix (shape [n_samples, n_features, n_features]) if with_cov is true.
        """
        predicted_rates, _ = self.gpr_fitter.predict(label_test)

        label_test = convert_to_circular_form(label_test, [self.circular_period])

        output_matrix = None

        if self.with_cov:
            inputs = tf.concat([predicted_rates, label_test], axis=-1)
            if cov_to_prec:
                precision = self.covnet.predict_precision(inputs)
                if self.standardize:
                    inv_std_outer = np.outer(1.0 / self.standardize_std, 1.0 / self.standardize_std)
                    precision = precision * inv_std_outer
                output_matrix = precision
            else:
                covariance = self.covnet.predict_covariance(inputs)

                if self.standardize:
                    std_outer = np.outer(self.standardize_std, self.standardize_std)
                    covariance = covariance * std_outer
                output_matrix = covariance

        if self.standardize:
            predicted_rates = predicted_rates * self.standardize_std + self.standardize_mean
        return predicted_rates, output_matrix

    def evaluate_f_loss(self, f_pred, response_test):
        """
        Evaluate the mean squared error loss between predicted and actual responses.

        Args:
            f_pred (array-like, shape [n_samples, n_features]): Predicted neural response.
            response_test (array-like, shape [n_samples, n_features]): True neural response.

        Returns:
            float: Mean Squared Error between the predicted and actual responses.
        """
        f_pred = tf.cast(f_pred, tf.float32)
        response_test = tf.cast(response_test, tf.float32)
        return tf.keras.losses.MSE(response_test, f_pred).numpy()

    def save(self, filepath):
        # Create a dictionary to hold non-TensorFlow state
        state = {an: getattr(self, an) for an in self.att_name}

        # Save non-TensorFlow state with hickle
        hkl.dump(state, filepath + '_state.hkl', mode='w')

        # Save TensorFlow model weights separately
        if self.covnet is not None:
            self.covnet.save_weights(filepath + '_weights.h5')

    def load(self, filepath):
        # Load the non-TensorFlow state with hickle
        state = hkl.load(filepath + '_state.hkl')
        for an, value in state.items():
            setattr(self, an, value)
    
        # Rebuild and load weights
        self.covnet = CovMLP(self.input_dim, self.output_dim)
        self.covnet.build((None, self.input_dim))
        self.covnet.load_weights(filepath + '_weights.h5')

    def copy(self):
        # Create a deep copy of the current instance's state
        new_instance = GPR_CovNet()

        # Manually copy attributes
        for an in self.att_name:
            setattr(new_instance, an, copy.deepcopy(getattr(self, an)))

        if self.covnet is None:
            return new_instance
        else:
            # Making sure TensorFlow model weights are also copied
            new_instance.covnet = CovMLP(self.input_dim, self.output_dim)
            new_instance.covnet.build((None, self.input_dim))
            # tf.keras.models.clone_model(self.covnet)
            new_instance.covnet.set_weights(self.covnet.get_weights())

        return new_instance

    def copy_covnet(self, covnet_ori):
        covnet_copy = CovMLP(self.input_dim, self.output_dim)
        covnet_copy.build((None, self.input_dim))
        covnet_copy.set_weights(covnet_ori.get_weights())
        return covnet_copy

    def _perform_random_search(self, predicted_rates, response, label, n_splits=5, lr=0.1):
        """ Perform random search with cross-validation to find optimal lambda and scale values.

        Parameters:
            predicted_rates: The predicted rates data.
            response: The response data.
            label: The label data.
            n_splits (int): Number of folds for cross-validation.

        Returns:
            opt_si (tuple): Tuple containing optimal lambda and scale values.
        """
        epochs_cp = self.epochs
        self.epochs = self.random_search_epochs # use smaller epochs for random search

        print('Random searching lambda and scale with cross-validation...')
        samples = [(np.random.choice(self.lambda_value_list), np.random.choice(self.scale_value_list)) for _ in range(self.n_rand)]

        opt_si, opt_val_loss = samples[0], np.inf
        kf = KFold(n_splits=n_splits)

        for si in samples:
            val_loss_sum = 0

            for train_index, val_index in kf.split(predicted_rates):
                # Split data into training and validation sets for each fold
                train_set = (predicted_rates[train_index], response[train_index], label[train_index])
                val_set = (predicted_rates[val_index], response[val_index], label[val_index])

                train_dataset = create_dataset(*train_set, self.batch_size)
                val_dataset = create_dataset(*val_set, self.batch_size)

                # Train network with current lambda and scale values
                self.covnet = self.copy_covnet(self.covnet_ori)
                self.lambda_value, self.scale = si
                _, val_loss = self._train_network(train_dataset, val_dataset, lr=lr)

                val_loss_sum += val_loss

            avg_val_loss = val_loss_sum / n_splits

            # Update optimal lambda and scale if average validation loss is improved
            if avg_val_loss < opt_val_loss:
                opt_si, opt_val_loss = si, avg_val_loss

            print(f'si: {si}, avg_val_loss: {avg_val_loss}')

        self.epochs = epochs_cp

        return opt_si

    def _fit_covariance_network(self, predicted_rates, response, label, lr=0.1, circular_period=None):
        """ Main method for training the model.

        Parameters:
            predicted_rates: The predicted rates data.
            response: The response data.
            label: The label data.
            lr (float): Learning rate.

        Returns:
            None
        """
        if circular_period is not None:
             label_convert = convert_to_circular_form(label, [circular_period])
        else:
            label_convert = label

        # Prepare the dataset for training
        dataset_set, _ = split_dataset(predicted_rates, response, label_convert, test_ratio=0)
        all_dataset = create_dataset(*dataset_set, self.batch_size)

        # Initialize the covariance network
        self.input_dim = predicted_rates.shape[-1] + label_convert.shape[-1]
        self.output_dim = predicted_rates.shape[-1]

        self.covnet = CovMLP(self.input_dim, self.output_dim)
        self._train_network(all_dataset, lr=lr)

        if self.n_rand is not None:
            self.covnet_ori = self.copy_covnet(self.covnet) # retraining

            try:
                self.lambda_value_list = tf.constant(self.lambda_value_list, dtype=tf.float32)
                self.scale_value_list = tf.constant(self.scale_value_list, dtype=tf.float32)
            except ValueError as e:
                raise ValueError('If n_rand is not None, lambda_value_list and scale_value_list must be filled.') from e

            opt_si = self._perform_random_search(predicted_rates, response, label_convert, lr=lr)
            self.lambda_value, self.scale = opt_si
            print(f"Optimal lambda: {self.lambda_value}, Optimal scale: {self.scale}")

            # train on the whole dataset
            self.covnet = self.copy_covnet(self.covnet_ori)

            epochs_cp = self.epochs
            self.epochs = self.random_search_epochs
            self._train_network(all_dataset, lr=lr)
            self.epochs = epochs_cp

    def _train_network(self, train_dataset, val_dataset=None, verbose=False, lr=0.1):
        """
        Train the covariance network for a specified number of epochs.

        Args:
            train_dataset (tf.data.Dataset): Training dataset. Each element is a tuple 
                of (f_batch, response_batch, label_batch) with shapes ([batch_size, n_features], 
                [batch_size, n_features], [batch_size, n_labels]).
            val_dataset (tf.data.Dataset): Validation dataset with the same structure as train_dataset.
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for epoch in range(self.epochs):
            train_loss = self._run_batches(train_dataset, is_training=True)
            if val_dataset is not None:
                val_loss = self._run_batches(val_dataset, is_training=False)
                if verbose:
                    print(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                val_loss = None
        return train_loss, val_loss # return the final loss

    def _run_batches(self, dataset, is_training):
        """
        Run the model on batches of data from the dataset.

        Args:
            dataset (tf.data.Dataset): Dataset to process. Each element is a tuple 
                of (f_batch, response_batch, label_batch) with shapes ([batch_size, n_features], 
                [batch_size, n_features], [batch_size, n_labels]).
            is_training (bool): Flag indicating if the model is in training mode.

        Returns:
            float: Average loss over the processed batches.
        """
        total_loss = 0
        for f_batch, response_batch, label_batch in dataset:
            if is_training:
                loss_value = self._train_step(f_batch, response_batch, label_batch)
            else:
                loss_value = self._test_step(f_batch, response_batch, label_batch)
            total_loss += loss_value
        return total_loss / len(dataset)

    def _train_step(self, f, response, label):
        """
        Perform a single training step.

        Args:
            f (array-like, shape [batch_size, n_features]): Predicted neural response.
            response (array-like, shape [batch_size, n_features]): True neural response.
            label (array-like, shape [batch_size, n_labels]): Input data.

        Returns:
            float: Loss value for the step.
        """
        with tf.GradientTape() as tape:
            inputs = tf.concat([f, label], axis=-1)
            L = self.covnet(inputs)
            loss_value = custom_loss(f, response, label, self.lambda_value, self.scale)(L)
        grads = tape.gradient(loss_value, self.covnet.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.covnet.trainable_variables))
        return loss_value

    def _test_step(self, f, response, label):
        """
        Perform a single test step.

        Args:
            f (array-like, shape [batch_size, n_features]): Predicted neural response.
            response (array-like, shape [batch_size, n_features]): True neural response.
            label (array-like, shape [batch_size, n_labels]): Input data.

        Returns:
            float: Loss value for the step.
        """
        inputs = tf.concat([f, label], axis=-1)
        L = self.covnet(inputs)
        return custom_loss(f, response, label, None, None)(L)

def save_gpr_covnet_all(speed_win, result, save_dir):
    """
    Saves the `speed_win` list and all `GPR_CovNet` models in `result`.

    :param speed_win: List of speed windows (e.g., [[0.025, 0.05], [0.05, 0.075]]).
    :param result: Nested list of GPR_CovNet instances, shape (len(speed_win), num_bootstraps).
    :param save_dir: Directory to save the model files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save speed_win
    speed_win_path = os.path.join(save_dir, 'speed_win.hkl')
    hkl.dump(speed_win, speed_win_path, mode='w')

    # Save each GPR_CovNet instance
    for i, model_list in enumerate(result):
        for j, model in enumerate(model_list):
            model_prefix = os.path.join(save_dir, f'model_{i}_{j}')
            model.save(model_prefix)

def load_gpr_covnet_all(save_dir):
    """
    Loads the `speed_win` list and all `GPR_CovNet` models from `save_dir`.

    :param save_dir: Directory to load the model files from.
    :return: Dictionary containing 'speed_win' list and 'result' nested list_of GPR_CovNet instances,
             shape (len(speed_win), num_bootstraps).
    """
    # Load speed_win
    speed_win_path = os.path.join(save_dir, 'speed_win.hkl')
    speed_win = hkl.load(speed_win_path)

    result = []
    for i in range(len(speed_win)):
        result_row = []
        j = 0
        while True:
            try:
                model_prefix = os.path.join(save_dir, f'model_{i}_{j}')
                instance = GPR_CovNet()
                instance.load(model_prefix)
                result_row.append(instance)
                j += 1
            except:
                break
        result.append(result_row)

    return {'speed_win': speed_win, 'result': result}
