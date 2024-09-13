import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_sphere_mesh():
    u = np.linspace(0, 2 *np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    return x_sphere, y_sphere, z_sphere

# this is a template of what a manifold should look like
class Manifold_Template_Class():
    def point_on_manifold(self):
        pass
    def get_covariance_matrix(self, **kwargs):
        pass
    def random_point(self, **kwargs):
        '''sample random points on the manifold'''
        pass

class Sphere_Manifold():
    def __init__(self, sigma):
        self.sigma = sigma

    def point_on_sphere(self, theta, phi):
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    def point_on_manifold(self, theta, phi):
        return self.point_on_sphere(theta, phi)

    def _get_covariance_matrix_single(self, theta, phi):
        r = self.point_on_sphere(theta, phi)

        t_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])

        if theta == 0:
            t_phi = np.array([-np.sin(phi), np.cos(phi), 0])
        else:
            t_phi = np.array([-np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), 0])

        # Normalizing tangential vectors
        t_theta = t_theta / np.linalg.norm(t_theta)
        t_phi = t_phi / np.linalg.norm(t_phi)

        T = np.column_stack((r, t_theta, t_phi))
        Sigma = np.diag([2 * self.sigma**2, self.sigma**2, self.sigma**2])
        return T @ Sigma @ T.T

    def _get_covariance_matrix_arr(self, theta, phi):
        cov_list = []
        for t, p in zip(theta, phi):
            cov = self._get_covariance_matrix_single(t, p)
            cov_list.append(cov)

        return np.array(cov_list)

    def get_covariance_matrix(self, theta, phi):
        if isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
            if theta.ndim != 1 or phi.ndim != 1:
                raise ValueError("theta and phi should be 1D arrays")
            return self._get_covariance_matrix_arr(theta, phi)
        else:
            return self._get_covariance_matrix_single(theta, phi)

    def random_point(self, theta, phi):
        if isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
            if theta.ndim != 1 or phi.ndim != 1:
                raise ValueError("theta and phi should be 1D arrays")
            results = []
            for t, p in zip(theta, phi):
                mean = self.point_on_sphere(t, p)
                cov = self._get_covariance_matrix_single(t, p)
                result = np.random.multivariate_normal(mean, cov)
                results.append(result)
            results = np.array(results)
            return results
        else:
            raise ValueError("theta and phi should be 1D arrays")

def get_circle_mesh():
    theta = np.linspace(0, 2 *np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    return x_circle, y_circle

class Circle_Manifold():
    def __init__(self, sigma, center=[0, 0], x_scale=1):
        self.sigma = sigma
        self.center = center
        self.x_scale = x_scale

    def point_on_circle(self, theta):
        if not isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[1] == 1:
            raise ValueError("theta should be 2D array with one column")
        x_point = self.center[0] + self.x_scale * np.cos(theta.flatten())
        y_point = self.center[1] + np.sin(theta.flatten())
        return np.array([x_point, y_point]).T

    def point_on_manifold(self, theta):
        return self.point_on_circle(theta)

    def _get_covariance_matrix_single(self, theta):
        theta_flat = theta.flatten()
        r = np.array([np.cos(theta_flat), np.sin(theta_flat)])
        t_theta = np.array([-np.sin(theta_flat), np.cos(theta_flat)])
        T = np.column_stack((r, t_theta))
        Sigma = np.diag([4 * self.sigma**2, self.sigma**2])

        # Create a scaling matrix along the x-direction
        scaling_matrix = np.array([[self.x_scale, 0], [0, 1]])

        return scaling_matrix @ T @ Sigma @ T.T @ scaling_matrix.T

    def _get_covariance_matrix_arr(self, theta):
        cov_list = []
        for t in theta.flatten():
            cov = self._get_covariance_matrix_single(t)
            cov_list.append(cov)

        return np.array(cov_list)

    def get_covariance_matrix(self, theta):
        if isinstance(theta, np.ndarray):
            return self._get_covariance_matrix_arr(theta)
        else:
            return self._get_covariance_matrix_single(theta)

    def random_point(self, theta):
        if isinstance(theta, np.ndarray):
            results = []
            for t in theta:
                mean = self.point_on_circle(t)[0]
                cov = self._get_covariance_matrix_single(t)
                result = np.random.multivariate_normal(mean, cov)
                results.append(result)
            results = np.array(results)
            return results
        else:
            raise ValueError("theta should be 2D array, with one feature per row")
