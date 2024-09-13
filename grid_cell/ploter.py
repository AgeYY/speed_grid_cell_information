import matplotlib.pyplot as plt
from umap import UMAP
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def better_spine(ax, spine_width=2, labelsize=20):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    return ax

def scatter_torus_ploter(feamap, visualize_method='umap', umap_components=3, umap_n_neighbors=100, umap_min_dist=0.8, umap_metric='cosine', umap_init='spectral', output_data=False, color=None):
    '''
    Plot the feamap of grid cell data
    input:
    ;;feamap: np.array of shape (n, m), where n is the number of samples, m is the number of features, could be pca features or grid cell firing rate
    ;;visualize_method: str, 'umap' or 'pca'
    ;;umap_components: int, number of components for umap and pca
    ;;umap_n_neighbors: int, number of neighbors for umap
    ;;umap_min_dist: float, min_dist for umap
    ;;umap_metric: str, metric for umap
    ;;umap_init: str, init for umap
    output:
    ;;fig, ax: matplotlib figure and axis
    '''
    print('visualizing feamap of shape {} using {}...'.format(feamap.shape, visualize_method))

    if visualize_method == 'umap':
        # fitter = UMAP(n_components=umap_components, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, init=umap_init, random_state=42)
        fitter = UMAP(n_components=umap_components, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, init=umap_init)
    elif visualize_method == 'pca':
        fitter = PCA(n_components=umap_components)
    feamap_umap = fitter.fit_transform(feamap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color is None:
        ax.scatter(feamap_umap[:, 0], feamap_umap[:, 1], feamap_umap[:, 2], c=feamap_umap[:, 2], s=1, alpha=0.7)
    else:
        ax.scatter(feamap_umap[:, 0], feamap_umap[:, 1], feamap_umap[:, 2], c=color, s=1, alpha=0.7)
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if output_data:
        return fig, ax, feamap_umap
    else:
        return fig, ax
    return fig, ax

def plot_barcodes(dgms, plot_prcnt=None, col_list='g', quantile_life_shuffle=None, replace_inf=20, figsize=(7.5, 2.5)):
    '''
    Plot the barcodes of persistence diagrams
    input:
    ;;dgms: list of np.array of shape (n, 2), where n is the number of bars in the persistence diagram. List length is the number of betti numbers
    ;;plot_prcnt: list of float, the percentile of bar length to plot.
    ;;col_list: list of str, the color of each betti number
    ;;quantile_life_shuffle: list of float, the quantile life of the shuffled bar. If None, no shuffled bar (grey bars) will be plotted
    ;;replace_inf: float, replace infinity with this value
    output:
    ;;fig, gs: matplotlib figure and gridspec
    '''
    n_h = len(dgms) # number of betti numbers
    if quantile_life_shuffle is None: quantile_life_shuffle = [None] * n_h # if quantile_life_shuffle is None, no shuffled bar will be plotted

    # filter out bars with length smaller than the percentile
    to_plot = []
    if plot_prcnt is None: plot_prcnt = [0] * n_h # defualt to plot all bars
    elif isinstance(plot_prcnt, (int, float)): plot_prcnt = [plot_prcnt] * n_h

    for h in dgms:
        h[~np.isfinite(h)] = replace_inf # replace inf with a large number, inf occurs in h0
        bar_lens = h[:,1] - h[:,0]
        to_plot.append(h[bar_lens > np.percentile(bar_lens, plot_prcnt[len(to_plot) % len(plot_prcnt)])])

    # plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, len(dgms))
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[:, curr_betti])

        if quantile_life_shuffle[curr_betti] is not None: # plot shuffled bar
            [ax.plot([interval[0], interval[0] + quantile_life_shuffle[curr_betti]], [i, i], color=(0.7, 0.7, 0.7, 1.0), lw=1.5) for i, interval in enumerate(curr_bar)]

        [ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti % len(col_list)], lw=1.5) # plot real bar
         for i, interval in enumerate(curr_bar)]

        ax.set_ylim([-1, len(curr_bar)])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('H{}'.format(curr_betti))
        if curr_betti == ( len(to_plot) - 1 ): ax.set_xlabel('Radius')
    return fig, gs

def gp_torus_surface(feamap, color_arr=None, cmap='viridis', cbar_label='', fig=None, ax=None, transparency_factor=0.2, reshape_feamap=True, with_colorbar=True, lw=0.6):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if reshape_feamap:
        n_bins = np.sqrt(feamap.shape[0]).astype(int)
        feamap = feamap.reshape(n_bins, n_bins, 3) # reshape to grid


    if color_arr is None:
        ax.plot_surface(feamap[:, :, 0], feamap[:, :, 1], feamap[:, :, 2], cmap=cmap, alpha=transparency_factor, rstride=2, cstride=2, edgecolors=(0, 0, 0, 0.1), lw=lw)
    else:
        cmap_instance = cm.get_cmap(cmap)  # Replace 'viridis' with your preferred colormap
        norm = plt.Normalize(color_arr.min(), color_arr.max())
        colors = cmap_instance(norm(color_arr))
        colors[..., -1] *= transparency_factor
        ax.plot_surface(feamap[:, :, 0], feamap[:, :, 1], feamap[:, :, 2], rstride=2, cstride=2, lw=lw, facecolors=colors)

        if with_colorbar:
            # Create a colorbar
            mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(color_arr)
            cbar = plt.colorbar(mappable, ax=ax)
            cbar.set_label(cbar_label)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return fig, ax

def error_bar_plot(x, y, fig=None, ax=None, color='tab:blue', label='', error_mode='se', mean_mode='mean', **plot_kargs):
    if fig is None: fig, ax = plt.subplots(figsize=(3, 3))

    if mean_mode == 'mean':
        mean_y = [np.mean(v) for v in y]
    else:
        mean_y = [np.median(v) for v in y]

    if error_mode == 'se':
        se_y = [np.std(v) / np.sqrt(len(v)) for v in y]
        ax.errorbar(x, mean_y, yerr=se_y, fmt='o', color=color)
    elif error_mode == 'std':
        se_y = [np.std(v) for v in y]
        print(x.shape, np.array(mean_y).shape, np.array(se_y).shape)
        ax.errorbar(x, mean_y, yerr=se_y, fmt='o', color=color)
    elif error_mode == 'quantile':
        y_25 = np.array([np.percentile(v, 25) for v in y])
        y_75 = np.array([np.percentile(v, 75) for v in y])
        ax.errorbar(x, mean_y, yerr=[mean_y - y_25, y_75 - mean_y], fmt='o', color=color)

    ax.plot(x, mean_y, color=color, label=label, **plot_kargs)
    return fig, ax
