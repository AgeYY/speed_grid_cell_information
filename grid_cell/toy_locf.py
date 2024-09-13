import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from grid_cell.manifold_fitter import label_mesh
from grid_cell.gkr import GKR_Fitter
import grid_cell.torus_geometry as tg
from scipy.stats import norm
from sklearn.decomposition import PCA
from grid_cell.ori_tuning_manifold import Ori_Tuning_Manifold, Place_Cell_Manifold
from grid_cell.lole import LOCF, get_random_dl_vector
from grid_cell import toy_manifold_metric_estimator as tmme

class LOCFExperiment:
    def __init__(self, n_neuron, n_data, dl=0.1, box_size=[0.05], std_frac=0.2, sigma=0.3, logistic_C=1.0, toy_manifold_name='ori', min_data=50, n_gkr_epoch=10):
        '''
        toy_manifold_name: 'ori' or 'place'
        '''
        self.n_neuron = n_neuron
        self.n_data = n_data
        self.dl = dl
        self.box_size = box_size
        self.std_frac = std_frac
        self.sigma = sigma
        self.logistic_C = logistic_C
        self.min_data = min_data
        self.n_gkr_epoch = n_gkr_epoch
        if toy_manifold_name == 'ori':
            self.toy_manifold = Ori_Tuning_Manifold(self.n_neuron, random_gain=True, std_frac=self.std_frac, sigma=self.sigma)
            self.bound = [[0, 2 * np.pi]]
            self.circular_period = 2 * np.pi
            self.n_input = 1
        elif toy_manifold_name == 'place':
            self.toy_manifold = Place_Cell_Manifold(n_cell=self.n_neuron, sigma=self.sigma, std_frac=self.std_frac, random_sigma=True)
            self.bound=[[-1, 1], [-1, 1]]
            self.circular_period = None
            self.n_input = 2

    def generate_data(self):
        self.theta = label_mesh(self.bound, self.n_data, random=True, grid=False)
        self.r = self.toy_manifold.random_point(self.theta)
        return self.theta, self.r

    def compute_lole(self, theta_train, r_train):
        lole = LOCF(box_size=self.box_size, min_data=self.min_data, model=LogisticRegression(C=self.logistic_C), dl=self.dl)
        accu, val_qm = lole.calculate_locf_accuracy(theta_train, r_train, query_mesh_size=200)
        return accu

    def compute_fisher(self, theta_noiseless):
        jacobian_noiseless = tg.compute_jacobian_central(
            tmme.noiseless_prediction_wrapper, theta_noiseless, h=0.01, model=self.toy_manifold
        )
        cov_noiseless = self.toy_manifold.get_covariance_matrix(theta_noiseless)
        precision_noiseless = np.linalg.pinv(cov_noiseless)
        fisher_noiseless = tg.compute_fisher_info(jacobian_noiseless, precision_noiseless)
        return fisher_noiseless

    def compute_gkr_fisher(self, theta_train, r_train, theta_noiseless):
        model = GKR_Fitter(circular_period=self.circular_period, n_input=self.n_input, n_output=r_train.shape[1], n_epochs=self.n_gkr_epoch)
        model.fit(r_train, theta_train)
        _, cov_pred = model.predict(theta_noiseless, return_cov=True)
        jacob = tg.compute_jacobian_central(tmme.gpr_prediction_wrapper, theta_noiseless, h=0.01, model=model)
        precision = np.linalg.pinv(cov_pred)
        fisher_gkr = tg.compute_fisher_info(jacob, precision)
        return fisher_gkr

    def calculate_pred_locf_accuracy(self, fisher):
        pred_locf_accuracy_from_fisher = calculate_locf_accuracy_upper_bound(n_input=self.n_input, dl=self.dl, fisher=fisher)
        return pred_locf_accuracy_from_fisher

    def run_experiment(self):
        theta_train, r_train = self.generate_data()
        accu = self.compute_lole(theta_train, r_train)
        theta_noiseless = label_mesh(self.bound, 300, random=True, grid=False)

        fisher_noiseless = self.compute_fisher(theta_noiseless)
        pred_locf_accuracy_from_fisher = self.calculate_pred_locf_accuracy(fisher_noiseless)

        fisher_gkr = self.compute_gkr_fisher(theta_train, r_train, theta_noiseless)
        pred_locf_accuracy_from_fisher_gkr = self.calculate_pred_locf_accuracy(fisher_gkr)

        return accu, pred_locf_accuracy_from_fisher, pred_locf_accuracy_from_fisher_gkr

def run_experiments(n_boot, parameter_name, parameter_values, base_params):
    accu_list, plaff_list, plaff_gkr_list = [], [], []

    for value in parameter_values:
        accu_list.append([])
        plaff_list.append([])
        plaff_gkr_list.append([])
        for _ in range(n_boot):
            params = base_params.copy()
            params[parameter_name] = value
            experiment = LOCFExperiment(**params)
            accu, pred_locf_accuracy_from_fisher, pred_locf_accuracy_from_fisher_gkr = experiment.run_experiment()
            accu_list[-1].append(accu)
            plaff_list[-1].append(pred_locf_accuracy_from_fisher)
            plaff_gkr_list[-1].append(pred_locf_accuracy_from_fisher_gkr)
        
    return accu_list, plaff_list, plaff_gkr_list

def compute_dsT_K_ds(DS, DK):
    # DS shape: (n_ds, n)
    # DK shape: (n_K, n, n)
    # return shape: (n_ds, n_K)
    
    n_ds, n = DS.shape
    n_K, _, _ = DK.shape
    
    # Reshape DS to (n_ds, 1, n) to enable broadcasting
    DS_expanded = DS[:, np.newaxis, :]  # Shape: (n_ds, 1, n)
    
    # Compute DS^T * DK * DS using broadcasting and einsum
    # einsum notation '...i,ijk,...k->...j' computes matrix multiplication for batched inputs
    results = np.einsum('dni,nij,dnj->dn', DS_expanded, DK, DS_expanded)
    
    return results

def calculate_locf_accuracy_upper_bound(n_input, dl, fisher, n_samples=500):
    ds = get_random_dl_vector(n_input, dl=dl, n_samples=n_samples)
    core = compute_dsT_K_ds(ds, fisher)
    pred_locf_accuracy_from_fisher = norm.cdf(np.sqrt(core)).flatten().mean()
    return pred_locf_accuracy_from_fisher
