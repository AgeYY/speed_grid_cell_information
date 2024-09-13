import numpy as np
import pandas as pd
from gph import ripser_parallel
from grid_cell.manifold_fitter import GP_Fitter, Avg_Fitter, label_mesh
from scipy import stats
from grid_cell.util import Shuffled_Matrix
from ripser import Rips
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy import sparse

def find_quantile_lifetime(dgms_list, inf_thre=1e10, quantile=0.95):
    '''
    quantile is the quantile of the lifetime, e.g. 0.95. If negative, it computes the maximum lifetime
    '''
    quantile_lifetimes = []
    for i in range(len(dgms_list[0])):
        lifetimes = [ds[i][:, 1] - ds[i][:, 0] for ds in dgms_list]
        concatenated = np.concatenate(lifetimes)
        if quantile < 0:
            quantile_lifetime = np.max(concatenated)
        else:
            quantile_lifetime = np.percentile(concatenated, quantile * 100)
        quantile_lifetimes.append(quantile_lifetime)
    quantile_lifetimes = [quantile_lifetimes[i] if np.abs(quantile_lifetimes[i]) < inf_thre else None for i in range(len(quantile_lifetimes))]
    return quantile_lifetimes

class Is_Torus():
    '''
    This class is used to determine whether the feamap is a torus.
    '''
    def __init__(self, n_bins=50, x_bound=[-0.75, 0.75], y_bound=[-0.75, 0.75], coeff=47, n_shuffle=10, quantile=-1, zthresh=5, long_life_method='zscore', life_time_cutoff_quantile=80):
        '''
        n_bins: number of bins for the grid, used for avg or gp transforming feamap
        x_bound, y_bound: x and y bounds for the grid, used for avg or gp transforming feamap
        coeff: coefficient for computing persistent homology
        n_shuffle: number of shuffles for computing the quantile lifetime of each barcode in the chance level
        quantile: quantile for determining whether a barcode is long enough to be considered as a torus
        '''
        self.n_bins = n_bins
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.coeff = coeff
        self.n_shuffle = n_shuffle
        self.quantile = quantile
        self.zthresh = zthresh
        self.long_life_method = long_life_method
        self.life_time_cutoff_quantile = life_time_cutoff_quantile

    def set_fitter(self, fitter=None):
        '''
        fitter: a fitter object, which has a fit method and a predict method. Fitter could be None then feamap will not be further preprocessed, directly used for persistent homology.
        '''
        self.fitter = fitter

    def _fit_predict(self, feamap, label):
        '''
        feamap: check whether this feamap is a torus or not
        label: the label of the feamap
        '''
        # fit and predict the feamap
        if self.fitter is None:
            return feamap.copy(), label.copy()

        self.fitter.fit(feamap, label)
        if isinstance(self.fitter, GP_Fitter):
            query_mesh = label_mesh([self.x_bound, self.y_bound], mesh_size=self.n_bins, grid=True)
            feamap, label = self.fitter.predict(query_mesh)
        elif isinstance(self.fitter, Avg_Fitter):
            feamap, label = self.fitter.predict(self.n_bins, self.x_bound, self.y_bound)
        else:
            feamap, label = feamap.copy(), label.copy()
        return feamap, label

    def is_torus(self, feamap, label, output_manifold=False):
        '''
        feamap: check whether this feamap is a torus or not
        label: the label of the feamap
        '''
        feamap, label = self._fit_predict(feamap, label)

        no_nan_rows = np.where(~np.isnan(feamap).any(axis=1))[0]
        feamap, label = np.array(feamap)[no_nan_rows, :], np.array(label)[no_nan_rows, :] # remove nan rows
        if feamap.shape[0] < 10:
            if output_manifold:
                return False, [[0]], [[0]] # too many nan we directly report failure
            else: return False

        # obtain the lifetime of bars in the original data and shuffled data
        dsparse = cloud_2_sparse_mat(feamap)
        rips = Rips(coeff=self.coeff, maxdim=1)
        dgms = rips.fit_transform(dsparse, distance_matrix=True)

        h1 = dgms[1]
        life_time = h1[:, 1] - h1[:, 0]
        cutoff = np.percentile(life_time, self.life_time_cutoff_quantile)
        life_time = life_time[life_time >= cutoff]

        if self.long_life_method == 'shuffle_quantile':
            sm = Shuffled_Matrix(feamap)
            dgms_shuffle = [rips.fit_transform(sm[i]) for i in range(self.n_shuffle)]
            quantile_life_shuffle = find_quantile_lifetime(dgms_shuffle, quantile=self.quantile) # the quantile lifetime of each barcode in the shuffled data, in each homology group.

            # make judgement whether it is a torus
            num_long_bar = np.sum(life_time > quantile_life_shuffle[1])
        elif self.long_life_method == 'zscore':
            zscores = stats.zscore(life_time)
            num_long_bar = np.sum(zscores > self.zthresh)

        print('number of long bars: ', num_long_bar)
        if output_manifold:
            return num_long_bar == 2, feamap, label
        else:
            return num_long_bar == 2

def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    source: check ripser tutorial https://ripser.scikit-tda.org/en/latest/notebooks/Sparse%20Distance%20Matrices.html?highlight=sparse

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]
    #By default, takes the first point in the permutation to be the
    #first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]

def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight
    source: check ripser tutorial https://ripser.scikit-tda.org/en/latest/notebooks/Sparse%20Distance%20Matrices.html?highlight=sparse

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()

def cloud_2_sparse_mat(feamap, eps=0.6, metric='euclidean'):
    '''
    convert data cloud to sparse distance matrix
    '''
    D = pairwise_distances(feamap, metric=metric)
    lambdas = getGreedyPerm(D)
    DSparse = getApproxSparseDM(lambdas, eps, D)
    return DSparse

def betti1_equal_2(feamap, coeff=47, eps=0.6, life_time_cutoff_quantile=0, long_life_method='zscore', n_shuffle=100, shuffle_quantile=95, zthresh=5, n_threads=-1):
    no_nan_rows = np.where(~np.isnan(feamap).any(axis=1))[0]
    feamap = np.array(feamap)[no_nan_rows, :] # remove nan rows
    if feamap.shape[0] < 10:
        return False

    # obtain the lifetime of bars in the original data and shuffled data
    dsparse = cloud_2_sparse_mat(feamap, eps=eps)
    dgms = ripser_parallel(dsparse, metric='precomputed', maxdim=1, n_threads=n_threads)
    dgms = dgms['dgms']

    h1 = dgms[1]
    life_time = h1[:, 1] - h1[:, 0]
    cutoff = np.percentile(life_time, life_time_cutoff_quantile)
    life_time = life_time[life_time >= cutoff]

    if long_life_method == 'shuffle_quantile':
        sm = Shuffled_Matrix(feamap)
        dgms_shuffle = [rips.fit_transform(sm[i]) for i in range(n_shuffle)]
        quantile_life_shuffle = find_quantile_lifetime(dgms_shuffle, quantile=shuffle_quantile) # the quantile lifetime of each barcode in the shuffled data, in each homology group.

        # make judgement whether it is a torus
        num_long_bar = np.sum(life_time > quantile_life_shuffle[1])
    elif long_life_method == 'zscore':
        zscores = stats.zscore(life_time)
        num_long_bar = np.sum(zscores > zthresh)

    print('number of long bars: ', num_long_bar)
    return num_long_bar == 2, dgms

def betti_torus_check(feamap, eps=0.2, life_time_cutoff_quantile=0, long_life_method='zscore', n_shuffle=100, shuffle_quantile=95, zthresh=5, n_threads=-1):
    no_nan_rows = np.where(~np.isnan(feamap).any(axis=1))[0]
    feamap = np.array(feamap)[no_nan_rows, :]  # Remove NaN rows
    if feamap.shape[0] < 10:
        return False, False  # Not enough data to proceed

    # Obtain the lifetime of bars in the original and shuffled data
    dsparse = cloud_2_sparse_mat(feamap, eps=eps)
    dgms = ripser_parallel(dsparse, metric='precomputed', maxdim=2, n_threads=n_threads)
    dgms = dgms['dgms']

    # Check Betti 1
    h1 = dgms[1]
    life_time_h1 = h1[:, 1] - h1[:, 0]
    cutoff_h1 = np.percentile(life_time_h1, life_time_cutoff_quantile)
    life_time_h1 = life_time_h1[life_time_h1 >= cutoff_h1]

    # Check Betti 2
    h2 = dgms[2] if len(dgms) > 2 else np.array([])  # Handle case when there is no Betti 2 data
    life_time_h2 = h2[:, 1] - h2[:, 0] if h2.size > 0 else np.array([])
    cutoff_h2 = np.percentile(life_time_h2, life_time_cutoff_quantile) if h2.size > 0 else 0
    life_time_h2 = life_time_h2[life_time_h2 >= cutoff_h2]

    if long_life_method == 'shuffle_quantile':
        sm = Shuffled_Matrix(feamap)
        dgms_shuffle = [rips.fit_transform(sm[i]) for i in range(n_shuffle)]
        quantile_life_shuffle_h1 = find_quantile_lifetime(dgms_shuffle, quantile=shuffle_quantile)[1]
        quantile_life_shuffle_h2 = find_quantile_lifetime(dgms_shuffle, quantile=shuffle_quantile)[2] if len(dgms_shuffle[0]) > 2 else np.array([])

        num_long_bar_h1 = np.sum(life_time_h1 > quantile_life_shuffle_h1)
        num_long_bar_h2 = np.sum(life_time_h2 > quantile_life_shuffle_h2) if life_time_h2.size > 0 else 0

    elif long_life_method == 'zscore':
        zscores_h1 = stats.zscore(life_time_h1)
        zscores_h2 = stats.zscore(life_time_h2) if life_time_h2.size > 0 else np.array([])

        num_long_bar_h1 = np.sum(zscores_h1 > zthresh)
        num_long_bar_h2 = np.sum(zscores_h2 > zthresh) if zscores_h2.size > 0 else 0

    print('Number of long bars in Betti 1:', num_long_bar_h1)
    print('Number of long bars in Betti 2:', num_long_bar_h2)

    return (num_long_bar_h1 == 2) and (num_long_bar_h2 == 1), dgms
