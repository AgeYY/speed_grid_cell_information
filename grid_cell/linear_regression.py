import numpy as np
from scipy.stats import norm

class BayesianLinearRegression():
    """
    Bayesian Linear Regression model with iterative hyperparameter optimization. Please attach 1 in the first column of your x to indicate bias
    
    Parameters
    ----------
    prior_mean : float or array-like, optional
        The prior mean of the weights. If a scalar, it is treated as the prior mean for all weights. Default is 0.0.
    prior_precision : float or array-like, optional
        The prior precision of the weights. If a scalar, it is treated as alpha. Default is 1.0.
    beta : float, optional
        The precision (inverse of variance) of the noise in the data. Default is 1.0.
    
    Attributes
    ----------
    posterior_mean : array-like, shape (n_features,)
        The posterior mean of the weights.
    posterior_cov : array-like, shape (n_features, n_features)
        The posterior covariance of the weights.
    confidence_intervals : array-like, shape (n_features, 2)
        The 95% confidence intervals for the weights.
    r_squared : float
        The R-squared value of the training data.
    """
    
    def __init__(self, prior_mean=0.0, prior_precision=1, beta=1, EPS=1e-6):
        if np.isscalar(prior_mean):
            self.prior_mean = prior_mean
        else:
            self.prior_mean = np.array(prior_mean)
        
        if np.isscalar(prior_precision):
            self.prior_precision = prior_precision
        else:
            self.prior_precision = np.array(prior_precision)
        
        self.beta = beta
        self.posterior_mean = None
        self.posterior_cov = None
        self.confidence_intervals = None
        self.r_squared = None
        self.EPS = EPS # avoid the posterior mean to be zero

    def fit(self, X, y, compute_auxiliary=True):
        """
        Fit the Bayesian Linear Regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        compute_auxiliary : bool, optional
            If True, compute the 95% confidence intervals for the weights and the R-squared value for the training data. Default is True.
        """
        n_features = X.shape[1]
        
        # If prior_mean is a scalar, create a zero vector of appropriate size
        if np.isscalar(self.prior_mean):
            self.prior_mean = np.ones(n_features) * self.prior_mean
        
        # Compute posterior covariance
        XT_X = np.dot(X.T, X)
        self.prior_precision =  self.prior_precision if not np.isscalar(self.prior_precision) else np.eye(n_features) * self.prior_precision
        self.posterior_cov = np.linalg.inv(self.prior_precision + self.beta * XT_X)
        
        # Compute posterior mean
        XT_y = np.dot(X.T, y)
        self.posterior_mean = np.dot(self.posterior_cov, np.dot(self.prior_precision, self.prior_mean) + self.beta * XT_y)

        if np.all(np.sqrt(np.var(y)) < self.EPS): # if y is constant
            print('y is constant, output zeros for posterior mean and covariance')
            self.posterior_mean = np.zeros(n_features)
            self.posterior_cov = np.zeros((n_features, n_features))
            self.beta = 1.0 / self.EPS

        if compute_auxiliary:
            self._compute_auxiliary(X, y)
    
    def _compute_auxiliary(self, X, y):
        # Compute 95% confidence intervals for the weights
        std_devs = np.sqrt(np.diag(self.posterior_cov))
        z_value = norm.ppf(0.975)
        self.confidence_intervals = np.vstack((self.posterior_mean - z_value * std_devs, 
                                               self.posterior_mean + z_value * std_devs)).T

        # p-value
        z = (self.posterior_mean) / np.sqrt(np.diag(self.posterior_cov))
        self.p_values = 2 * (1 - norm.cdf(np.abs(z)))

        # Compute R-squared value for the training data
        y_pred = np.dot(X, self.posterior_mean)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        self.r_squared = 1 - (ss_res / ss_total)

    def predict(self, X):
        """
        Predict using the Bayesian Linear Regression model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        y_mean : array-like, shape (n_samples,)
            The predicted mean values.
        y_var : array-like, shape (n_samples,)
            The predicted variance values.
        """
        if self.posterior_mean is None or self.posterior_cov is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Predictive mean
        y_mean = np.dot(X, self.posterior_mean)
        
        # Predictive variance
        y_var = 1 / self.beta + np.sum(np.dot(X, self.posterior_cov) * X, axis=1)

        return y_mean, y_var

    def sample_weights(self, n_samples=1):
        """
        Sample weights from the posterior distribution.
        
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to generate. Default is 1.
        
        Returns
        -------
        samples : array-like, shape (n_samples, n_features)
            The sampled weights.
        """
        if self.posterior_mean is None or self.posterior_cov is None:
            raise ValueError("Model has not been fitted yet.")
        # if np.sqrt(np.var(self.posterior_cov.flatten())) < self.EPS: # return samples equal to the posterior_mean
        #     print('Posterior covariance is zero, return samples equal to the posterior mean')
        #     return np.tile(self.posterior_mean, (n_samples, 1))

        return np.random.multivariate_normal(self.posterior_mean, self.posterior_cov, n_samples)
    
    def sample_y(self, X, n_samples=1):
        """
        Sample target values from the posterior predictive distribution.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        n_samples : int, optional
            The number of samples to generate. Default is 1.
        
        Returns
        -------
        y_samples : array-like, shape (n_samples, n_samples)
            The sampled target values.
        """
        weights_samples = self.sample_weights(n_samples)
        y_samples = []
        for w in weights_samples:
            y_mean = np.dot(X, w)
            y_sample = np.random.normal(y_mean, 1 / np.sqrt(self.beta))
            y_samples.append(y_sample)
        return np.array(y_samples)
    
    def optimize_hyperparameters(self, X, y, max_iter=500):
        """
        Optimize the hyperparameters (prior_precision and beta) using an iterative method.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        max_iter : int, optional
            The maximum number of iterations for the optimization. Default is 100.
        """
        if np.all(np.sqrt(np.var(y)) < self.EPS): # if y is constant
            print("Warning: y is a constant. Hyperparameters cannot be optimized.")
            return

        N, M = X.shape
        alpha = self.prior_precision if np.isscalar(self.prior_precision) else 1.0 # only scalar precision supported
        beta = self.beta
        
        for i in range(max_iter):
            # Fit the model with current alpha and beta
            self.prior_precision = alpha
            self.beta = beta
            self.fit(X, y, compute_auxiliary=False)
            
            # Calculate gamma
            XT_X = np.dot(X.T, X)
            eigenvalues, _ = np.linalg.eigh(beta * XT_X)
            gamma = np.sum(eigenvalues / (alpha + eigenvalues))

            # Update alpha and beta
            alpha_new = gamma / (np.dot(self.posterior_mean.T, self.posterior_mean) + self.EPS) # + tol to avoid division by zero
            beta_new = (N - gamma) / (np.sum((y - np.dot(X, self.posterior_mean)) ** 2)) # + tol to avoid division by zero

            # Check for convergence
            if np.abs(alpha - alpha_new) < self.EPS and np.abs(beta - beta_new) < self.EPS:
                break
            
            alpha, beta = alpha_new, beta_new
        
        # Update the model with optimized hyperparameters
        self.prior_precision = alpha
        self.beta = beta

def fit_a_line(x, y):
    x_fit = np.hstack([np.ones((x.size, 1)), x.reshape(-1, 1)])
    model = BayesianLinearRegression()
    model.optimize_hyperparameters(x_fit, y)
    model.fit(x_fit, y)
    return model

def fit_a_line_boot(x, y, mode='BBLR'):
    if mode == 'BBLR':
        model = BootstrapBLR()
        x_fit = x[:, :, np.newaxis]
        x_fit = np.concatenate([np.ones_like(x_fit), x_fit], axis=-1)
        y_fit = y[:, :, np.newaxis]
        model.fit(x_fit, y_fit)
    elif mode == 'bootstrap':
        model = BootstrapLinearRegression()
        model.fit(x, y)
    return model

def output_params(model):
    slope = model.posterior_mean[1]
    intercept = model.posterior_mean[0]
    slope_conf_int = model.confidence_intervals[1]
    r2 = model.r_squared
    p_value_intercept = model.p_values[0]
    p_value_slope = model.p_values[1]
    return slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope

def output_params_boot(model):
    return output_params(model)

class BootstrapLinearRegression:
    def __init__(self):
        pass

    def fit_all_boot(self, x, y):
        '''
        x, y: (n_boot, n_samples)
        '''

        # Prepare the output arrays
        n_boot = x.shape[0]
        slopes = np.zeros(n_boot)
        intercepts = np.zeros(n_boot)

        # Fit a line for each row
        for i in range(n_boot):
            # Fit line: y = m*x + b
            m, b = np.polyfit(x[i], y[i], 1)
            slopes[i] = m
            intercepts[i] = b
        return slopes, intercepts

    def fit(self, x, y, compute_auxiliary=True):
        '''
        x, y: (n_boot, n_samples)
        '''
        slopes, intercepts = self.fit_all_boot(x, y)
        self.posterior_mean = np.array([np.mean(intercepts), np.mean(slopes)])
        self.posterior_cov = np.cov(np.vstack((intercepts, slopes)))

        if compute_auxiliary:
            self._compute_auxiliary(x, y)

    def _compute_auxiliary(self, x, y):
        std_devs = np.sqrt(np.diag(self.posterior_cov))
        z_value = norm.ppf(0.975)
        self.confidence_intervals = np.vstack((self.posterior_mean - z_value * std_devs, 
                                               self.posterior_mean + z_value * std_devs)).T

        # p-value
        z = (self.posterior_mean) / np.sqrt(np.diag(self.posterior_cov))
        self.p_values = 2 * (1 - norm.cdf(np.abs(z)))

        # R-squared
        y_flat = y.flatten(); x_flat = x.flatten()
        self.r_squared = 1 - np.sum((y_flat - self.predict(x_flat)[0]) ** 2) / np.sum((y_flat - np.mean(y_flat)) ** 2)

    def predict(self, x):
        '''
        x: (n_samples,)
        '''
        X = np.hstack([np.ones((x.shape[0], 1)), x.reshape(-1, 1)])
        y_mean = np.dot(X, self.posterior_mean)
        y_var = np.sum(np.dot(X, self.posterior_cov) * X, axis=1)
        return y_mean, y_var

def draw_line(ax, x, y, model, color='k', line_label='Fitted Line', data_label='Data', error_band_alpha=0.2):
    x_mesh = np.linspace(x.min(), x.max(), 100)
    x_mesh = np.hstack([np.ones((x_mesh.size, 1)), x_mesh.reshape(-1, 1)])
    y_mean, y_var = model.predict(x_mesh)
    y_std = np.sqrt(y_var)
    z_value = norm.ppf(0.975)
    y_upper = y_mean + z_value * y_std
    y_lower = y_mean - z_value * y_std

    ax.fill_between(x_mesh[:, 1], y_upper, y_lower, color=color, alpha=error_band_alpha)
    ax.plot(x_mesh[:, 1], y_mean, color=color, label=line_label)
    ax.scatter(x, y, color=color, s=10, marker='o', label=data_label) # convert to cm
    return ax

def draw_line_boot(ax, x, y, model, color='k', line_label='Fitted Line', data_label='Data', error_band_alpha=0.2, mode='BBLR'):
    if mode == 'BBLR':
        x_mesh = np.linspace(x.min(), x.max(), 100)
        x_mesh = x_mesh[np.newaxis, :, np.newaxis]
        x_mesh_cat = np.concatenate([np.ones_like(x_mesh), x_mesh], axis=-1)
        y_mean, y_var = model.predict(x_mesh_cat)

        y_mean, y_var = y_mean.flatten(), y_var.flatten()
        x_mesh = x_mesh.flatten()
    elif mode == 'bootstrap':
        x_mesh = np.linspace(x.min(), x.max(), 100)
        y_mean, y_var = model.predict(x_mesh)

    y_std = np.sqrt(y_var)
    z_value = norm.ppf(0.975)
    y_upper = y_mean + z_value * y_std
    y_lower = y_mean - z_value * y_std

    ax.fill_between(x_mesh, y_upper, y_lower, color=color, alpha=error_band_alpha)
    ax.plot(x_mesh, y_mean, color=color, label=line_label)
    ax.scatter(x, y, color=color, s=10, marker='o', label=data_label) # convert to cm
    return ax

class BootstrapBLR():
    def __init__(self, BLR_prior_mean=0.0, BLR_prior_precision=1.0, BLR_beta=1.0, EPS=1e-6, laplace=True):
        '''
        current version only supports laplace = Ture
        '''
        self.BLR_prior_mean = BLR_prior_mean
        self.BLR_prior_precision = BLR_prior_precision
        self.BLR_beta = BLR_beta
        self.EPS = EPS

    def fit(self, x, y, compute_auxiliary=True):
        '''
        x: (n_boot, n_samples, n_features)
        y: (n_boot, n_samples, 1)
        '''
        n_boot = x.shape[0]
        self.BLR_list = []
        for i in range(n_boot):
            model = BayesianLinearRegression(self.BLR_prior_mean, self.BLR_prior_precision, self.BLR_beta, self.EPS)
            model.optimize_hyperparameters(x[i], y[i].flatten())
            model.fit(x[i], y[i].flatten())
            self.BLR_list.append(model)

        posterior_mean = np.array([model.posterior_mean for model in self.BLR_list])
        posterior_cov = np.array([model.posterior_cov for model in self.BLR_list])
        beta = np.array([model.beta for model in self.BLR_list])
        self.beta = 1.0 / np.mean(1.0 / beta)

        self.posterior_mean = np.mean(posterior_mean, axis=0)
        bias = (posterior_mean - self.posterior_mean)
        bias_cov = np.einsum('ijk,ikl->ijl', bias[:, :, np.newaxis], bias[:, np.newaxis, :])
        self.posterior_cov = np.mean(posterior_cov, axis=0) + np.mean(bias_cov, axis=0)

        if compute_auxiliary: self._compute_auxiliary(x, y)

    def predict(self, xq):
        '''
        input:
        xq: (n_boot, n_samples, n_features)
        note the shape of posterior_mean is (n_features,), poterior_cov is (n_features, n_features)
        output:
        y_mean: (n_boot, n_samples, 1)
        y_var: (n_boot, n_samples, 1)
        '''
        y_mean = np.einsum('ijk, k -> ij', xq, self.posterior_mean) # ignore the first axis
        y_var = np.einsum('ijk, kl, ijl -> ij', xq, self.posterior_cov, xq) + 1.0 / self.beta
        y_mean, y_var = y_mean[:, :, np.newaxis], y_var[:, :, np.newaxis]
        return y_mean, y_var

    def _compute_auxiliary(self, x, y):
        '''
        x: (n_boot, n_samples, n_features)
        y: (n_boot, n_samples, 1)
        '''
        std_devs = np.sqrt(np.diag(self.posterior_cov))
        z_value = norm.ppf(0.975)
        self.confidence_intervals = np.vstack((self.posterior_mean - z_value * std_devs, 
                                               self.posterior_mean + z_value * std_devs)).T

        # p-value
        z = (self.posterior_mean) / np.sqrt(np.diag(self.posterior_cov))
        self.p_values = 2 * (1 - norm.cdf(np.abs(z)))

        # R-squared
        y_flat = y.flatten();
        self.r_squared = 1 - np.sum((y - self.predict(x)[0]).flatten() ** 2) / np.sum((y_flat - np.mean(y_flat)) ** 2)
