from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt

def fig_ax_by_dim(dim):
    '''
    Create a figure and axis object based on the dimension of the data
    input:
        dim: int, the dimension of the data. 2 or 3
    '''
    if dim == 2:
        fig, ax = plt.subplots(figsize=(3, 3))
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def plot_scatter_or_line_by_dim(r, theta, ax, pca=None, visualize_dim=-1, plot_mode='scatter', theta_range=[0, 2*np.pi]):
    '''
    Plot the scatter plot of the data points on the manifold
    input:
        r: np.ndarray [n_sample, n_feature], the data points on the manifold
        theta: np.ndarray [n_sample, 1], the angles of the data points
        ax: matplotlib.axes.Axes, the axis object to plot the data
        visualize_dim: int, the dimension to visualize the data. If -1, pca must be specified for transforming the data to 2D or 3D
        pca: sklearn.decomposition.PCA, the PCA object to transform the data. If None, a new PCA object will be created to fit the data. In this case, visualize_dim should be specified
        plot_mode: str, 'scatter', 'surface' or 'line', the mode to plot the data. If surface, ax must be a 3D axis
    '''
    if pca is None:
        r_pc = PCA(n_components=visualize_dim).fit_transform(r)
    else:
        r_pc = pca.transform(r)

    colors = plt.cm.hsv((theta - theta_range[0]) / (theta_range[1] - theta_range[0]))

    if plot_mode == 'scatter':
        if r_pc.shape[1] == 2:
            ax.scatter(r_pc[:, 0], r_pc[:, 1], c=colors, marker='+', s=30, alpha=0.6)
        elif r_pc.shape[1] == 3:
            ax.scatter(r_pc[:, 0], r_pc[:, 1], r_pc[:, 2], c=colors, s=30, marker='+', alpha=0.6)
    elif plot_mode == 'line':
        for i in range(len(theta) - 1):
            if r_pc.shape[1] == 2:
                ax.plot(r_pc[i:i+2, 0], r_pc[i:i+2, 1], color=colors[i], linewidth=4)
            elif r_pc.shape[1] == 3:
                ax.plot(r_pc[i:i+2, 0], r_pc[i:i+2, 1], r_pc[i:i+2, 2], color=colors[i], linewidth=4)
        if r_pc.shape[1] == 2:
            ax.plot(r_pc[-1:, 0], r_pc[:1, 1], color=colors[-1], linewidth=4)
        elif r_pc.shape[1] == 3:
            ax.plot(r_pc[-1:, 0], r_pc[:1, 1], r_pc[:1, 2], color=colors[-1], linewidth=4)
    elif plot_mode == 'surface':
        r_pc = r_pc.reshape((int(np.sqrt(r_pc.shape[0])), int(np.sqrt(r_pc.shape[0])), r_pc.shape[1]))
        colors = colors.reshape((int(np.sqrt(colors.shape[0])), int(np.sqrt(colors.shape[0])), colors.shape[1]))
        ax.plot_surface(r_pc[:, :, 0], r_pc[:, :, 1], r_pc[:, :, 2], alpha=0.6, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True)
    return ax

def fit_plot_manifold_figure(af, r_train, theta_train, theta_noiseless, eg_theta, fig=None, ax=None, model_name='avg', title=None, visualization_dim=2, pca=None, theta_range=[0, 2*np.pi], plot_mode='line'):
    if fig is None: fig, ax = fig_ax_by_dim(visualization_dim)

    af.fit(r_train, theta_train)

    if model_name == 'avg':
        r_pred, qm_valid, _, _ = af.predict_1d(theta_noiseless, return_cov=True)
    else:
        af.with_cov = False
        r_pred, _ = af.predict(theta_noiseless)
        qm_valid = theta_noiseless
        af.with_cov = True

    ax.set_aspect('equal')

    if pca is None:
        pca = PCA(n_components=visualization_dim).fit(r_pred)

    plot_scatter_or_line_by_dim(r_pred, qm_valid[:, 0], ax, pca, plot_mode=plot_mode, theta_range=theta_range)

    if model_name == 'avg':
        eg_r, qm_eg, eg_cov, _ = af.predict_1d(eg_theta, return_cov=True)
    else:
        eg_r, eg_cov = af.predict(eg_theta)

    plot_cov_ellipse(eg_r, eg_cov, ax=ax, zo=2, pca=pca)

    ax.set_title(title)

    return fig, ax

def plot_cov_ellipse(mean_arr, cov_arr, ax=None, zo=2, n_sig=2, pca=None):
    '''
    mean_arr: (n_points, m)
    cov_arr: (n_points, m, m)
    '''
    n_visualize_dim = pca.n_components_ if pca is not None else mean_arr.shape[1]
    if pca is not None: # projecting mean and covariance to pca
        mean_arr = pca.transform(mean_arr)
        cov_arr = np.array([pca.components_ @ cov @ pca.components_.T for cov in cov_arr])

    for mean, cov in zip(mean_arr, cov_arr):
        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Construct the ellipse using eigenvalues as the width and height
        shape_para = n_sig * 2 * np.sqrt(eigenvalues)
        # ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='r', fc='None')
        if n_visualize_dim == 2:
            # Calculate the angle of rotation in degrees
            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
            width, height = shape_para
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='k', fill=False, linewidth=1, zorder=zo)
            ax.add_patch(ellipse)

            ax.scatter(mean_arr[:, 0], mean_arr[:, 1], c='k', marker='+', zorder=zo+1) # center of ellipse
        elif n_visualize_dim == 3:
            plot_3d_ellipse(cov, mean, ax=ax, zorder=zo)
            ax.scatter(mean_arr[:, 0], mean_arr[:, 1], mean_arr[:, 2], c='k', marker='+', zorder=zo+1)

def plot_3d_ellipse(cov_matrix, center, ax=None, zorder=2):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale and rotate the points
    for i in range(len(x)):
        for j in range(len(x[0])):
            [x[i, j], y[i, j], z[i, j]] = np.dot(
                eigenvectors,
                [eigenvalues[0] ** 0.5 * x[i, j],
                 eigenvalues[1] ** 0.5 * y[i, j],
                 eigenvalues[2] ** 0.5 * z[i, j]]
            ) + center

    ax.plot_surface(x, y, z, rstride=5, cstride=5, color='grey', alpha=0.3, zorder=zorder)
