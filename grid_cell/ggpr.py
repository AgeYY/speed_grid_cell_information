from grid_cell.manifold_fitter import GP_Fitter
import numpy as np
from sklearn.covariance import graphical_lasso, GraphicalLasso
from grid_cell.higham import nearestPD, isPD
from global_setting import *

class GGPR_Fitter():
    '''
    Double Gaussian Process Regression Fitter. One is for fitting the mean (manifold), second is for fitting the covariance.
    You can save and load ggpr class by hkl.dump and hkl.load.
    '''
    def __init__(self, with_cov=True, circular_period=None, standardize=True, with_higham=True, split_cov_diag=True, n_inducing=300, glasso_alpha=0.1):
        self.att_name = [
            'with_cov',
            'circular_period',
            'standardize',
            'standardize_mean',
            'standardize_std',
            'gpr_mean', # for fitting the mean (manifold)
            'gpr_cov', # for fitting the covariance
            'with_higham',
            'split_cov_diag',
            'gram_p',
            'n_inducing', # for sparse GPR, number of inducing points. If None, use GPR.
            'glasso_alpha' # for graphical lasso
        ]

        self.with_cov = with_cov
        self.circular_period = circular_period

        self.standardize = standardize

        self.gpr_mean = None
        self.gpr_cov = None
        self.with_higham = with_higham

        self.split_cov_diag = split_cov_diag
        self.gram_p = Gram_Processor()

        self.n_inducing = n_inducing

        self.glasso_alpha = glasso_alpha

    def fit(self, response_train, label_train):
        '''
        Fit the model.
        input:
        response_train: (n_samples, n_features)
        label_train: (n_samples, n_labels)
        '''
        response_train_ = np.array(response_train, dtype=FLOAT_TYPE)
        label_train_ = np.array(label_train, dtype=FLOAT_TYPE)

        self.gpr_mean = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize, n_inducing=self.n_inducing)
        self.gpr_mean.fit(response_train_, label_train_)

        if self.with_cov:
            response_pred, _ = self.gpr_mean.predict(label_train_)
            gram = self._compute_gram(response_train_, response_pred)
            self.gram_p.set_gram(gram)

            if self.split_cov_diag:
                gram_diag = self.gram_p.gram_diag()
                gram_non_diag = self.gram_p.gram_non_diag()
                self.gpr_cov_diag = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize, n_inducing=self.n_inducing)
                self.gpr_cov_non_diag = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize, n_inducing=self.n_inducing)
                self.gpr_cov_diag.fit(gram_diag, label_train)
                self.gpr_cov_non_diag.fit(gram_non_diag, label_train)
            else:
                gram_flat = self.gram_p.gram_flat()
                self.gpr_cov = GP_Fitter(circular_period=self.circular_period, standardize=self.standardize)
                self.gpr_cov.fit(gram_flat, label_train_)

    def predict(self, label_test):
        '''
        Predict the stats manifold, i.e. mean and covariance for each label.
        input:
            label_test: (n_samples, n_labels)
        output:
            tuple: Tuple containing predicted rates (shape [n_samples, n_features])
                   and optionally covariance matrix (shape [n_samples, n_features, n_features]) if with_cov is true.
        '''
        label_test = np.array(label_test, dtype=FLOAT_TYPE)
        predicted_rates, _ = self.gpr_mean.predict(label_test)
        predicted_cov = None
        if self.with_cov:
            if self.split_cov_diag:
                predicted_diag, _ = self.gpr_cov_diag.predict(label_test)
                predicted_non_diag, _ = self.gpr_cov_non_diag.predict(label_test)
                predicted_cov = self.gram_p.diag_non_diag_to_gram(predicted_diag, predicted_non_diag)
            else:
                predicted_cov_flat, _ = self.gpr_cov.predict(label_test)
                predicted_cov = self.gram_p.flat_to_gram(predicted_cov_flat)

            if self.with_higham: # find the nearest positive definite matrix
                for i, cov in enumerate(predicted_cov):
                    if isPD(cov): continue
                    predicted_cov[i] = nearestPD(cov)

            # for i, cov in enumerate(predicted_cov):
            #     predicted_cov[i], _ = graphical_lasso(cov, alpha=self.glasso_alpha)

        return predicted_rates, predicted_cov

    def _compute_gram(self, response_train, response_pred):
        '''
        Compute the gram matrix.
        '''
        diff = response_train - response_pred
        gram = np.einsum('ij,ik->ijk', diff, diff)

        return gram

def sym_mats_to_flat_tris(matrices):
    """
    Flatten the upper triangular part of each symmetric matrix in a collection to a 2D array.
    :param matrices: An array of symmetric matrices with shape (n_samples, n_features, n_features)
    :return: A 2D array where each row is the flattened upper triangular part of a symmetric matrix
    """
    n_samples, n_features, _ = matrices.shape
    flat_tris = matrices[:, np.triu_indices(n_features)[0], np.triu_indices(n_features)[1]]
    return flat_tris

def flat_tris_to_sym_mats(flat_tris, n_features):
    """
    Reconstruct symmetric matrices from the flattened upper triangular parts in a 2D array.
    
    :param flat_tris: A 2D array where each row is the flattened upper triangular part of a symmetric matrix
    :param n_features: The number of features (n) of the reconstructed n x n matrices.
    :return: An array of reconstructed symmetric matrices with shape (n_samples, n_features, n_features)
    """
    n_samples = flat_tris.shape[0]
    if flat_tris.shape[1] != (n_features * (n_features + 1)) // 2:
        raise ValueError("The size of flat_tris does not match n_features.")

    sym_matrices = np.zeros((n_samples, n_features, n_features))
    upper_indices = np.triu_indices(n_features)
    
    for i in range(n_samples):
        sym_matrices[i][upper_indices] = flat_tris[i]
        sym_matrices[i].T[upper_indices] = flat_tris[i]

    return sym_matrices

class Gram_Processor():
    def __init__(self, gram=None):
        self.gram = gram
        if gram is None:
            self.n_feature = None
            self.n_mat = None
        else:
            self.n_feature = gram.shape[1]

    def set_gram(self, gram):
        self.gram = gram
        self.n_feature = self.gram.shape[1]

    def gram_flat(self):
        return sym_mats_to_flat_tris(self.gram)

    def gram_diag(self):
        return np.diagonal(self.gram, axis1=1, axis2=2)

    def gram_non_diag(self):
        i_upper = np.triu_indices(self.n_feature, k=1)
        upper_triangular = self.gram[:, i_upper[0], i_upper[1]]
        return upper_triangular

    def diag_non_diag_to_gram_flat(self, diag, non_diag):
        diagonal_idxs = [i * self.n_feature - (i - 1) * i // 2 for i in range(self.n_feature)]
        n_non_diagonal = self.n_feature * (self.n_feature + 1) // 2
        non_diagonal_idxs = [i for i in range(n_non_diagonal) if i not in diagonal_idxs]

        n_mat = diag.shape[0]
        flat_recover = np.zeros((n_mat, len(diagonal_idxs) + len(non_diagonal_idxs)))
        flat_recover[:, diagonal_idxs] = diag
        flat_recover[:, non_diagonal_idxs] = non_diag
        return flat_recover

    def diag_non_diag_to_gram(self, diag, non_diagonal):
        flat_recover = self.diag_non_diag_to_gram_flat(diag, non_diagonal)
        return flat_tris_to_sym_mats(flat_recover, self.n_feature)

    def flat_to_gram(self, flat):
        return flat_tris_to_sym_mats(flat, self.n_feature)
