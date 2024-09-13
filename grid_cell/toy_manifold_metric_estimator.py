import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from grid_cell.higham import nearestPD, isPD
import itertools
from grid_cell.manifold_fitter import label_mesh
from grid_cell.sphere_manifold import Circle_Manifold
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold, Place_Cell_Manifold
import os
from grid_cell.torus_geometry import compute_jacobian_central, compute_riemannian_metric, compute_riemannian_metric_from_jacobian, compute_fisher_info
from global_setting import *

class Metric_Predictor():
    '''
    This class fit the data and output predictions
    '''
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def fit_predict(self, r, theta, theta_noiseless, jacob_h=0.01):
        self.model.fit(r, theta)
        if self.model_name in {'avg', 'kernel', 'lw'}:
            r_pred, _, cov_pred, _ = self.model.predict_1d(theta_noiseless, return_cov=True)
            jacob = compute_jacobian_central(avg_prediction_wrapper, theta_noiseless, model=self.model, h=jacob_h)
        else:
            r_pred, cov_pred = self.model.predict(theta_noiseless)
            jacob = compute_jacobian_central(gpr_prediction_wrapper, theta_noiseless, model=self.model, h=0.01)

        r_metric = compute_riemannian_metric_from_jacobian(jacob)

        precision_pred = np.linalg.pinv(cov_pred)
        fisher = compute_fisher_info(jacob, precision_pred)

        return r_pred, cov_pred, r_metric, precision_pred, fisher

def relative_error(matrix1, matrix2, mat1_is_true=True):
    # Ensure matrix1 and matrix2 have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape.")
    
    # Compute Frobenius norm of the difference matrix
    difference_matrix = matrix1 - matrix2
    if mat1_is_true:
        denominator = np.linalg.norm(matrix1)
    else:
        denominator = (np.linalg.norm(matrix1) + np.linalg.norm(matrix2)) / 2
    norm = np.linalg.norm(difference_matrix) / denominator
    return norm

def matrix_similarity(matrix1, matrix2, mat1_is_true=True):
    norm = relative_error(matrix1, matrix2, mat1_is_true)
    similarity = 1 / (1 + norm)  # Similarity measure
    return similarity

def noiseless_prediction_wrapper(theta, model=None):
    return model.point_on_manifold(theta)

def gpr_prediction_wrapper(theta, model=None):
    return model.predict(theta)[0]

def avg_prediction_wrapper(theta, model=None):
    return model.predict_1d(theta)

class Toy_Manifold_Wrapper():
    def __init__(self, toy_manifold_name, **manifold_kwargs):
        self.toy_manifold_name = toy_manifold_name
        if toy_manifold_name == 'circle':
            self.tm = Circle_Manifold(**manifold_kwargs)
            self.circular_period = 2 * np.pi
            self.theta_range = [[0, 2 * np.pi]]
            self.n_label = 1
        elif toy_manifold_name == 'ori_tuning':
            self.tm = Ori_Tuning_Manifold(**manifold_kwargs)
            self.circular_period = 2 * np.pi
            self.theta_range = [[0, 2 * np.pi]]
            self.n_label = 1
        elif toy_manifold_name == 'place_cell':
            self.tm = Place_Cell_Manifold(**manifold_kwargs)
            self.circular_period = None
            self.theta_range = np.array([self.tm.cc.field_range, self.tm.cc.field_range]) * 0.8 # to avoid end effect.
            self.n_label = 2

        self.n_cell = manifold_kwargs['n_cell']

    def get_train_data(self, n_points, blanket_spacing_each_dim=0.1):
        n_label = len(self.theta_range)
        theta_random = []
        for i in range(n_label):
            theta_i = np.random.uniform(self.theta_range[i][0], self.theta_range[i][1], n_points)
            theta_random.append(theta_i)
        theta_random = np.column_stack(theta_random)

        theta_blanket = []
        if blanket_spacing_each_dim is not None:
            theta_each_dim = [np.arange(a, b + blanket_spacing_each_dim, blanket_spacing_each_dim) for a, b in self.theta_range]
            theta_blanket = list(itertools.product(*theta_each_dim))

        theta = np.concatenate([theta_random, theta_blanket], axis=0)

        r = self.tm.random_point(theta)
        return r, theta

    def get_ground_truth(self, n_points=300):
        n_label = len(self.theta_range)
        theta_noiseless = []
        for i in range(n_label):
            theta_i = np.random.uniform(self.theta_range[i][0], self.theta_range[i][1], n_points)
            theta_noiseless.append(theta_i)
        theta_noiseless = np.column_stack(theta_noiseless)

        r_noiseless = self.tm.point_on_manifold(theta_noiseless) # ground truth circle
        cov_noiseless = self.tm.get_covariance_matrix(theta_noiseless)
        jacobian_noiseless = compute_jacobian_central(noiseless_prediction_wrapper, theta_noiseless, h=0.01, model=self.tm)
        r_metric_noiseless = compute_riemannian_metric_from_jacobian(jacobian_noiseless)
        precision_noiseless = np.linalg.pinv(cov_noiseless)
        fisher_noiseless = compute_fisher_info(jacobian_noiseless, precision_noiseless)
        return theta_noiseless, r_noiseless, cov_noiseless, r_metric_noiseless, precision_noiseless, fisher_noiseless
