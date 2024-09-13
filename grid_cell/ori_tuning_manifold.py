import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt
from global_setting import *

def exponential_gradient_fill(diagonal, alpha=0.5):
    '''
    diagonal (2D array): array of shape (n_sample, 10) representing the diagonal for each sample
    alpha (float): the exponential decay rate
    '''
    n_sample, size = diagonal.shape
    matrix = np.zeros((n_sample, size, size))

    # Fill the diagonal for each sample
    for k in range(n_sample):
        np.fill_diagonal(matrix[k], diagonal[k])

    # Create an array of distances
    indices = np.arange(size)
    distances = np.abs(indices[:, None] - indices)

    # Compute the exponential decay factor
    decay = np.exp(-alpha * distances)

    # Apply decay to the upper triangle (including the diagonal) for each sample
    matrix = matrix * decay[np.newaxis, :, :]

    return matrix

class Cell_Manifold():
    def __init__(self, n_unit, std_frac, decay_rate, *args, **kwargs):
        '''
        initialize a cell class in this init function. The cell class should have a fire method which takes a input and returns the firing rate of the cell.
        n_unit (int): the number of cell
        std_frac (float): the noise of each unit is a gaussian function with std = std_frac * r, where r is the firing rate of the cell
        env_std (float): the std of the environment noise
        '''
        pass

    def point_on_manifold(self, theta):
        return self.cc.fire(theta)

    def _get_covariance_matrix_single(self, theta):
        '''theta must be 1d'''
        r = self.cc.fire(theta)
        diag = r * self.std_frac + self.env_noise_std
        L = exponential_gradient_fill(diag, 1)
        cov = np.einsum('ijk,ilk->ijl', L, L)
        return cov

    def _get_covariance_matrix_arr(self, theta):
        '''theta must be a 1D array'''
        return self._get_covariance_matrix_single(theta)

    def get_covariance_matrix(self, theta):
        if isinstance(theta, np.ndarray):
            if theta.ndim != 2:
                raise ValueError("theta should be 2D array")
            return self._get_covariance_matrix_arr(theta)
        else:
            return self._get_covariance_matrix_single(theta)

    def random_point(self, theta):
        if not isinstance(theta, np.ndarray) and theta.ndim == 2:
            raise ValueError("theta should be 2D array")
        mean = self.point_on_manifold(theta)
        cov = self._get_covariance_matrix_arr(theta)
        results = np.array([np.random.multivariate_normal(mean[i], cov[i]) for i in range(len(mean))])
        return results

class Color_cell():
    '''
    convert degree to the firing rate of color cell (doesn't exist)
    '''
    def __init__(self, n_unit, sigma = None, random_gain=False):
        '''
        Args:
          n_unit (int): the number of color cell which are uniformly distributed from 0 to 360
          sigma (float): the width of the tuning curve
          random_gain (bool): whether to use random gain for each color cell
        '''
        self.n_unit = n_unit
        if sigma is None:
            if n_unit < 10:
                self.sigma = 2 * np.pi / n_unit / 2 # default sigma, note if sigma became to small, tuning curve will be too narrow and lead to all kinds of numerical error!!
            else:
                self.sigma = 0.3
        else:
            self.sigma = sigma
        self.yi = np.arange(0, 2 * np.pi, 2 * np.pi / n_unit)
        self.fire_max_before_gain = vonmises.pdf(0, 1 / (self.sigma * 2)**2, loc=0)
        if random_gain: self.gain = np.random.rand(1, self.n_unit) + 0.5
        else: self.gain = np.ones((1, self.n_unit))

    def fire(self, degree):
        '''
        input:
        degree (np.array [float](n)): n input degree
        output:
        ri (np.array [float](n, m)): m is the number of color cell
        '''
        n_input = degree.shape[0] # the number of degrees
        yi_tile = np.tile(self.yi, (n_input, 1)) # expand yi by repeating itself (as column) n times, where n is the number of degrees
        degree_tile = np.tile(degree.reshape(-1, 1), (1, self.n_unit)) # expand yi by repeating itself (as column) n times, where n is the number of degrees
        return self.ri_func(yi_tile, degree_tile)

    def ri_func(self, y, x):
        kappa = 1 / (self.sigma * 2)**2
        r =  vonmises.pdf(y, kappa, loc=x) / self.fire_max_before_gain
        r = r * self.gain
        return r

class Ori_Tuning_Manifold(Cell_Manifold):
    def __init__(self, n_cell, std_frac = 0.5, random_gain=False, env_noise_std=0.05, sigma=0.3):
        self.n_cell = n_cell
        self.std_frac= std_frac
        self.cc = Color_cell(n_cell, random_gain=random_gain)
        self.env_noise_std = env_noise_std
        self.sigma = sigma

class Place_Cell():
    def __init__(self, n_cell=256, field_range=[-1, 1], sigma=0.2, random_sigma=False):
        self.n_cell = n_cell
        self.field_range = field_range
        self.c_vec = np.random.uniform(*field_range, (n_cell, 2)) # the center of each place cell
        self.sigma = sigma
        self.max_fire = np.exp(- 1 / 2.0 / sigma**2)

        if random_sigma:
            self.sigma = np.random.uniform(0.5, 1.5, n_cell) * sigma
            self.sigma = self.sigma[:, np.newaxis]
            self.max_fire = np.exp(- 1 / 2.0 / np.min(self.sigma**2))

    # TEMP
    # def fire(self, x):
    #     '''
    #     x: 2d array [n_position, 2]
    #     '''
    #     r = []
    #     for xi in x:
    #         zi = np.sum((xi - self.c_vec)**2 / 2.0 / self.sigma**2, axis=1)
    #         # ri = np.exp(-zi) / self.max_fire
    #         ri = np.exp(-zi)
    #         r.append(ri)
    #     return np.array(r)

    def fire(self, x):
        '''
        x: 2d array [n_position, 2]
        '''
        zi = np.sum((x[:, np.newaxis, :] - self.c_vec)**2 / 2.0 / self.sigma**2, axis=2)
        ri = np.exp(-zi)
        return ri

class Place_Cell_Manifold(Cell_Manifold):
    def __init__(self, n_cell=256, field_range=[-1, 1], sigma=0.2, random_sigma=False, std_frac=0.5, env_noise_std=0.1):
        self.cc = Place_Cell(n_cell, field_range, sigma, random_sigma)
        self.n_cell = n_cell
        self.std_frac= std_frac
        self.env_noise_std = env_noise_std

if __name__ == "__main__":
    def test_sym_to_flat_and_back():
        n_samples = 5
        n_features = 3
        # Generate random symmetric matrices
        random_matrices = np.random.rand(n_samples, n_features, n_features)
        symmetric_matrices = np.array([(mat + mat.T) / 2 for mat in random_matrices])
        print(symmetric_matrices.shape)

        # Convert to flat upper triangular
        flat_tris = sym_mats_to_flat_tris(symmetric_matrices)

        # Reconstruct symmetric matrices
        reconstructed_mats = flat_tris_to_sym_mats(flat_tris, n_features)

        # Check if reconstruction is correct
        assert np.allclose(symmetric_matrices, reconstructed_mats), "Reconstructed matrices do not match the original."

        print("Test passed: Symmetric matrices were correctly flattened and reconstructed.")

    test_sym_to_flat_and_back()
