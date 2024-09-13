import numpy as np
from grid_cell.util import select_arr_by_t, get_data_filename_by_keywords, reduce_true_elements, digitize_arr
from sklearn import random_projection
import grid_cell.tuning as gct
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import os
from global_setting import *

def digitize_array_by_bin_size(array, min_val, max_val, bin_size):
    """
    Digitizes the values in an array based on the specified bounds and bin size.

    Parameters:
    array (np.ndarray): The input array to digitize.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.
    bin_size (float): The size of each bin.

    Returns:
    np.ndarray: The digitized array where each value is replaced by the corresponding bin value.
    """
    # Calculate the number of bins based on the bin size
    n_bins = int(np.ceil((max_val - min_val) / bin_size))
    
    # Compute the bin edges
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Digitize the array
    bin_indices = np.digitize(array, bin_edges) - 1
    
    # Compute the bin values (midpoints of bin edges)
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Map the bin indices to bin values
    digitized_array = bin_values[bin_indices]
    
    return digitized_array

class Grid_Cell_Processor():
    def load_data(self, mouse_name='R', day='day1', module='2', session='open_field_1', fr_smooth_sigma=10, adaptive_fr_sigma=False, base_speed=0.6, base_sigma=2, max_fr_sigma=50, min_fr_sigma=1, digitize_space=None, space_bound=[-0.75, 0.75], speed_estimation_method='finite_diff'):
        '''
        load grid cell data from the data folder

        input:
        ;;mouse_name (str): 'R' 'Q' or 'S'
        ;;day (str): 'day1' or 'day2'
        ;;module (str): '1' or '2', sometimes can be '3' depends on the mouse and day
        ;;session (str): 'open_field_1' or 'wagon_wheel_1' etc. Check the data txt file for more detail
        ;;fr_smooth_sigma (int): the sigma for smoothing the firing rate on the temporal domain. Unit is ms
        ;;digitize_space (int): the number of bins for digitizing the space. If None, the space is not digitized.

        output:
        ;;self.spike (dict): the spike time for each cell, unit is second.
        ;;self.x, self.y (np array, shape [n_time]): the x and y position of the mouse. Unit is meter
        ;;self.t (np array, shape [n_time]): the time stamp of the position data. Unit is second.
        ;;self.dt (float): the time step of the position data. Usually 0.01 s
        ;;self.fire_rate (np array, shape [n_time, n_cell]): the firing rate for each cell
        '''
        self.mouse_name = mouse_name
        self.module = module
        self.day = day
        self.session = session

        file_name = get_data_filename_by_keywords(self.mouse_name.lower(), self.day)
        cell_module_name = 'spikes_mod{}'.format(self.module)

        data_path = os.path.join(EXP_DATAROOT, file_name)
        data = np.load(data_path, allow_pickle=True)

        self.spike = data[cell_module_name].item()
        self.spike = {key: select_arr_by_t(value, value, session=self.session, file_name=file_name) for key, value in self.spike.items()} # unit is second
        self.x, self.y, self.t = (select_arr_by_t(data[key], data['t'], session=self.session, file_name=file_name) for key in ['x', 'y', 't']) # position data unit is meter
        self.dt = np.mean(np.diff(self.t)) # time step. Unit is second. The spacing is 10 ms

        self.speed = self.compute_speed(method=speed_estimation_method)
        if digitize_space is not None:
            self.x = digitize_arr(self.x, digitize_space, space_bound)
            self.y = digitize_arr(self.y, digitize_space, space_bound)

        if adaptive_fr_sigma:
            sigma_speed = gct.get_sigma_speed(self.speed, base_speed=base_speed, base_sigma=base_sigma, max_sigma=max_fr_sigma, min_sigma=min_fr_sigma)
            gaussian_window = gct.get_gaussian_window(sigma_speed)

            fire_rate = gct.compute_all_firing_rates(self.spike, self.t, None, z_score=False)
            fire_rate_smooth = np.zeros_like(fire_rate)
            for col in range(fire_rate.shape[1]):
                fire_rate_smooth[:, col] = gct.apply_gaussian_window(fire_rate[:, col], gaussian_window)
                print(f'processing neuron {col}...')
            self.fire_rate = fire_rate_smooth
        else:
            self.fire_rate = gct.compute_all_firing_rates(self.spike, self.t, fr_smooth_sigma, z_score=False)

    def compute_speed(self, method='finite_diff'):
        '''
        compute the speed of the mouse

        output:
        ;;self.speed (np array, shape [n_time]): the speed of the mouse. Unit is m/s
        '''
        if method == 'finite_diff':
            self.speed = gct.compute_speed(self.x, self.y, self.t) # speed at each time point
        elif method == 'kalman_filter':
            self.speed = gct.compute_speed_kalman(self.x, self.y, self.t)
        return self.speed

    def compute_rate_map(self, n_spatial_bin=50, spatial_sigma=2.75, visited_thre=0.01, method='rate'):
        '''
        compute the rate_map
        input:
        ;;n_spatial_bin (int): the number of spatial bins along x and y
        ;;spatial_sigma (float): the sigma of the gaussian filter for smoothing the rate map
        ;;visited_thre (float): the threshold for the staying time in each spatial bin. If the staying time is less than this threshold, the spatial bin will be masked out. Later this masked out locations will be corrected in computing the spatial smoothed rate map
        ;;method (str): 'rate' or 'spike'. 'rate' means compute the rate map by averaging the firing rate. 'spike' means compute the rate map by averaging the spike count. We recommend using 'rate' method.
        output:
        ;;rate_map (np array, shape [n_spatial_bin, n_spatial_bin, n_cell]): the rate map for each cell
        '''
        if method == 'spike': # compute the rate map by averaging the spike count
            self.rate_map = self._compute_rate_map_by_spike(n_spatial_bin, spatial_sigma, visited_thre)
        elif method == 'rate':
            self.rate_map = self._compute_rate_map_by_rate(n_spatial_bin, spatial_sigma, visited_thre)
        return self.rate_map

    def compute_autocorr_map(self, padding=10, replace=True, mode='same', boundary='fill', fillvalue=0):
        '''
        compute the autocorrelation map for each cell
        input:
        ;;padding (int): the padding for the rate map before computing the autocorrelation map
        ;;replace (bool): if True, replace the current rate map with the new rate map computed by compute_rate_map()
        ;;mode (str): the mode for computing the autocorrelation map. See scipy.signal.correlate2d for more detail
        ;;boundary (str): the boundary condition for computing the autocorrelation map. See scipy.signal.correlate2d for more detail
        ;;fillvalue (float): the fill value for computing the autocorrelation map. See scipy.signal.correlate2d for more detail
        output:
        ;;autocorr (np array, shape [n_spatial_bin, n_spatial_bin, n_cell]): the autocorrelation map for each cell
        '''
        if replace: self.rate_map = self.compute_rate_map()
        elif 'rate_map' not in self.__dict__: raise 'rate_map not computed yet'

        autocorr = []
        for i in range(self.rate_map.shape[-1]):
            rate_map_pad = np.pad(self.rate_map[:, :, i], padding, mode='linear_ramp')
            autocorr_temp = gct.compute_autocorrelation(rate_map_pad, rate_map_pad, mode=mode, boundary=boundary, fillvalue=fillvalue)
            autocorr.append(autocorr_temp)
        self.autocorr = np.moveaxis(np.array(autocorr), 0, -1)
        return self.autocorr

    def compute_gridness(self, replace=True):
        if replace: self.autocorr = self.compute_autocorr_map()
        elif 'autocorr' not in self.__dict__: raise 'autocorr (autocorrelation map) not computed yet'

        angles = [30, 60, 90, 120, 150]
        n_cell = self.autocorr.shape[-1]
        self.gridness = np.zeros(n_cell)
        for i in range(n_cell):
            correlations = gct.compute_autocorrelation_ori(self.autocorr[:, :, i], angles)
            self.gridness[i] = (correlations[1] + correlations[3]) / 2 - (correlations[0] + correlations[2] + correlations[4]) / 3
        return self.gridness

    def preprocess(self, speed_thre=-1., gridness_thre=0.3, pca_components=6, downsample_rate=1, use_rate_map=False, use_zscore=True, replace=True, return_speed=False, subset_slow_speed=False, speed_max=0.52, slow_speed_value=0.05, spatial_digitize_bin_size=None, spatial_bound=[-1, 1]):
        '''
        preprocess the data. Including 1. mask out the data with low speed and low gridness. 2. downsample the data. 3. zscore the data. 4. compute the pca components
        input:
        ;;speed_thre (float): the threshold for the speed. If the speed is less than this threshold, the data will be masked out. Unit is second
        ;;gridness_thre (float): the threshold for the gridness. If the gridness is less than this threshold, the data will be masked out.
        ;;pca_components (int): the number of pca components to keep
        ;;downsample_rate (int): the downsample rate for the data. downsample rate = 150 is a good value for umap visualization
        ;;use_rate_map (bool): if True, use the flatterned rate map is the the output feamap.
        ;;use_zscore (bool): if True, zscore the firing rate
        ;;replace (bool): if True, replace the current speed, rate map, gridness by default setting
        ;;return_speed (bool): if True, return the speed
        ;;subset_slow_speed (bool): if True, subset the data with slow speed
        ;;speed_max (float): the maximum speed for the slow speed subset
        ;;slow_speed_value (float): the value for the slow speed subset
        output:
        ;;fire_rate (np array, shape [n_time, n_cell]): the firing rate for each cell
        ;;x, y, t (np array, shape [n_time]): the x, y position and time stamp of the mouse
        '''
        if use_rate_map:
            if replace or not hasattr(self, 'rate_map'): self.compute_rate_map()
            return self.rate_map.reshape((-1, self.rate_map.shape[-1])), self.x.copy(), self.y.copy(), self.t.copy()

        if replace or not hasattr(self, 'gridness'): self.compute_gridness()
        # thresh small speed
        fire_rate = gct.filter_xyt_by_speed(self.fire_rate, self.speed, speed_thre)
        x, y, t = gct.filter_xyt_by_speed(self.x, self.speed, speed_thre), gct.filter_xyt_by_speed(self.y, self.speed, speed_thre), gct.filter_xyt_by_speed(self.t, self.speed, speed_thre)
        speed = gct.filter_xyt_by_speed(self.speed, self.speed, speed_thre)

        # thresh large speed
        speed_idx = speed < speed_max
        fire_rate, x, y, t, speed = fire_rate[speed_idx], x[speed_idx], y[speed_idx], t[speed_idx], speed[speed_idx]

        # thresh gridness
        fire_rate = fire_rate[:, self.gridness > gridness_thre]

        # downsample
        fire_rate, x, y, t, speed = fire_rate[::downsample_rate], x[::downsample_rate], y[::downsample_rate], t[::downsample_rate], speed[::downsample_rate]

        if spatial_digitize_bin_size is not None:
            x = digitize_array_by_bin_size(x, spatial_bound[0], spatial_bound[1], spatial_digitize_bin_size)
            y = digitize_array_by_bin_size(y, spatial_bound[0], spatial_bound[1], spatial_digitize_bin_size)

        if use_zscore: fire_rate = stats.zscore(fire_rate, axis=0)

        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            fire_rate = pca.fit_transform(fire_rate)

        if subset_slow_speed:
            idx = subset_speed_head(speed, slow_speed_value=slow_speed_value, speed_max=speed_max)
            fire_rate, x, y, t, speed = fire_rate[idx], x[idx], y[idx], t[idx], speed[idx]

        if return_speed: return fire_rate, x, y, t, speed
        else: return fire_rate, x, y, t

    def _compute_rate_map_by_rate(self, n_spatial_bin=50, spatial_sigma=2.75, visited_thre=0.01):
        # compute the staying time in each location
        stay_time = gct.compute_bin_times(self.x, self.y, self.dt, n_spatial_bin)
        visited_mask = stay_time >= visited_thre

        # compute the firing rate at each location
        n_cell = self.fire_rate.shape[1]
        rate_map = np.zeros([n_spatial_bin, n_spatial_bin, n_cell])
        for i in range(n_cell):
            rate_map[:, :, i] = gct.spatial_bin_fire_rate(self.fire_rate[:, i], self.x, self.y, n_bin=n_spatial_bin)
        rate_map = np.nan_to_num(rate_map) # unvisited bin will be nan, replace it with 0

        # smooth the firing rate, with visited mask correction
        rate_map = gaussian_filter(rate_map, spatial_sigma, axes=[0, 1]) # smooth the firing rate
        visited_mask = gaussian_filter(visited_mask.astype(float), sigma=spatial_sigma) # smooth the firing rate
        rate_map = rate_map / visited_mask[:, :, np.newaxis]

        return rate_map

    def _compute_rate_map_by_spike(self, n_spatial_bin, spatial_sigma, visited_thre):
        # compute the staying time in each location
        stay_time = gct.compute_bin_times(self.x, self.y, self.dt, n_spatial_bin)
        visited_mask = stay_time >= visited_thre

        # compute the spike rate at each location
        spike_count = np.zeros([n_spatial_bin, n_spatial_bin, len(self.spike)])
        for key, value in self.spike.items():
            spike_count[:, :, int(key)] = gct.spatial_bin_spikes_count(value, self.t, self.x, self.y, n_spatial_bin)

        spike_rate = spike_count / stay_time[:, :, np.newaxis]
        spike_rate = np.nan_to_num(spike_rate) # unvisited bin will be nan, replace it with 0

        # smooth the spike rate, with visited mask correction
        spike_rate = gaussian_filter(spike_rate, spatial_sigma, axes=[0, 1]) # smooth the firing rate
        visited_mask = gaussian_filter(visited_mask.astype(float), sigma=spatial_sigma) # smooth the firing rate
        spike_rate = spike_rate / visited_mask[:, :, np.newaxis]
        return spike_rate

class Data_Transformer():
    '''
    A class for transforming the firing rate data. whole data will firstly be subset into two subset size, one is fit one is test. Test data will further be transformed into transformed data. transform method can be 'data' (no transofrmation), 'pca' (pca transformation), 'pls' (pls transformation
    '''
    def load_data(self, fire_rate, label):
        '''
        load the data. Ideally these data are the partially preprocessed data from Grid_Cell_Processor()
        input:
        ;;fire_rate (np array, shape [n_sample, n_feature]): the firing rate data. n_feature is also the number of cells.
        ;;label (np array, shape [n_sample, n_label]): the label for each sample. label usually includes x, y, t, and speed
        '''
        self.fire_rate = fire_rate.copy()
        self.label = label.copy()

    def subset_fit_test(self, half_half=False, fit_subset_size=25000, test_subset_size=25000):
        '''
        subset the data into fit and test subset
        input:
        ;;half_half (bool): if True, half of the data will be used as fit subset, the other half will be used as test subset
        ;;fit_subset_size (int): the size of the fit subset. If None, the whole data set
        ;;test_subset_size (int): the size of the test subset
        output:
        ;;fire_rate_fit, label_fit, fire_rate_test, label_test (np array): the fit and test subset of the data
        '''
        n_sample = self.fire_rate.shape[0]
        if half_half:
            half = n_sample // 2
            subset_indices = np.random.choice(n_sample, (2, half), replace=False)
            fire_rate_fit, label_fit = self.fire_rate[subset_indices[0]], self.label[subset_indices[0]]
            fire_rate_test, label_test = self.fire_rate[subset_indices[1]], self.label[subset_indices[1]]
            return fire_rate_fit, label_fit, fire_rate_test, label_test

        fit_subset_indices = np.random.choice(n_sample, fit_subset_size or n_sample, replace=False) # if fit_subset_size is None, use the whole data set
        fire_rate_fit, label_fit = self.fire_rate[fit_subset_indices], self.label[fit_subset_indices]
        test_subset_indices = np.random.choice(n_sample, test_subset_size or n_sample, replace=False)
        fire_rate_test, label_test = self.fire_rate[test_subset_indices], self.label[test_subset_indices]

        return fire_rate_fit, label_fit, fire_rate_test, label_test

    def transform(self, fire_rate_fit, label_fit, fire_rate_test, label_test, transform_method='data', n_component=6):
        '''
        transform the test data into transformed data
        input:
        ;;fire_rate_fit, label_fit, fire_rate_test, label_test (np array): the fit and test subset of the data
        ;;transform_method (str): the transformation method. Can be 'data' (no transformation), 'pca' (pca transformation), 'pls' (pls transformation)
        ;;n_component (int): the number of components to keep in pca or pls transformation
        '''
        if transform_method == 'data':
            fire_rate_trans = fire_rate_test.copy()
        elif transform_method == 'pca':
            fire_rate_trans = PCA(n_components=n_component).fit(fire_rate_fit).transform(fire_rate_test)
        elif transform_method == 'pls':
            fire_rate_trans = PLSRegression(n_components=n_component).fit(fire_rate_fit, label_fit).transform(fire_rate_test)
        return fire_rate_trans

    def subset_transform(self, fit_subset_size=25000, test_subset_size=25000, n_component=6, transform_method='data', split_half_half=False, fit_label_id=None):
        '''
        subset the data into fit and test subset, and transform the test subset into transformed data
        input:
        ;;fit_subset_size (int): the size of the fit subset. If None, the whole data set
        ;;test_subset_size (int): the size of the test subset
        ;;n_component (int): the number of components to keep in pca or pls transformation
        ;;transform_method (str): the transformation method. Can be 'data' (no transformation), 'pca' (pca transformation), 'pls' (pls transformation)
        ;;split_half_half (bool): if True, half of the data will be used as fit subset, the other half will be used as test subset
        ;;fit_label_id (list): the id of the label to be used in fit subset. If None, all labels will be used
        '''
        fire_rate_fit, label_fit, fire_rate_test, label_test = self.subset_fit_test(half_half=split_half_half, fit_subset_size=fit_subset_size, test_subset_size=test_subset_size)

        if fit_label_id is not None: # in pls, we only regress on subset of id
            label_fit_trans = label_fit[:, fit_label_id]
            label_test_trans = label_test[:, fit_label_id]
        else:
            label_fit_trans = label_fit
            label_test_trans = label_test

        fire_rate_trans = self.transform(fire_rate_fit, label_fit_trans, fire_rate_test, label_test_trans, transform_method=transform_method, n_component=n_component)
        return fire_rate_trans, label_test

class Speed_Processor():
    def __init__(self):
        self.gcp = Grid_Cell_Processor() # this is used for using multiple functions in Grid_Cell_Processor class

    def load_data(self, fire_rate, label, dt=None):
        '''
        load the data
        input:
        ;;fire_rate (np array, shape [n_time, n_features]): the feature map. Feature can be cells or PC unit. The input better to be not processed by filtering or normalization etc. The time is better to be equally spaced as which it is in the raw data
        ;;label (np array, shape [n_time, n_labels]): the label for each time point. x, y, t, speed = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
        '''
        self.fire_rate = fire_rate
        self.label = label
        if dt is None:
            self.dt = np.mean(np.diff(self.label[:, 2])) # time step
        else:
            self.dt = dt

    def sample_data(self, bin_width=0.05, speed_min=0.0, speed_max=0.57, n_sample_data=2500, replace=True, n_random_projection=None):
        speed_bins = np.arange(speed_min, speed_max + bin_width / 10, bin_width) # the last element is added to include the max speed

        n_data_each_bin = []
        for i in range(speed_bins.size - 1):
            speed_win = [speed_bins[i], speed_bins[i + 1]]
            fire_rate_filt, label_filt = self._filt_data_speedwin(speed_win)
            n_data_each_bin.append(fire_rate_filt.shape[0])
        min_n_data = np.min(n_data_each_bin)
        if (n_sample_data > min_n_data) and (not replace):
            print('Warning: n_sample_data is larger than the minimum number of data across all speed bins. n_sample_data is set to be the minimum number of data: ', min_n_data)
            n_sample_data = min_n_data

        feamap_sample = []; label_sample = []
        for i in range(speed_bins.size - 1):
            speed_win = [speed_bins[i], speed_bins[i + 1]]
            fire_rate_filt, label_filt = self._filt_data_speedwin(speed_win)
            indices = np.random.choice(fire_rate_filt.shape[0], n_sample_data, replace=replace)
            feamap_sample.append(fire_rate_filt[indices])
            label_sample.append(label_filt[indices])
        feamap_sample = np.vstack(feamap_sample)
        label_sample = np.vstack(label_sample)

        if n_random_projection is not None:
            transform = random_projection.GaussianRandomProjection(n_components=n_random_projection)
            feamap_sample = transform.fit_transform(feamap_sample)

        return feamap_sample, label_sample

    def _filt_data_speedwin(self, speed_win):
        '''
        filt the data by speed window
        input:
        ;;speed_win (list of float [2]): the speed window. The rate map will be computed only from the data in this speed window. [speed_thre_low, speed_thre_up]
        '''
        speed = self.label[:, -1]
        fire_rate_filt = gct.filter_xyt_by_speed(self.fire_rate, speed, speed_thre=speed_win[0], speed_thre2=speed_win[1])
        label_filt = gct.filter_xyt_by_speed(self.label, speed, speed_thre=speed_win[0], speed_thre2=speed_win[1])
        return fire_rate_filt, label_filt

    def _compute_rate_map_speedwin(self, fire_rate_filt, label_filt):
        self.gcp.fire_rate = fire_rate_filt
        self.gcp.x, self.gcp.y, self.gcp.t = label_filt[:, 0], label_filt[:, 1], label_filt[:, 2]
        self.gcp.dt = self.dt
        rate_map = self.gcp.compute_rate_map()
        return rate_map

    def compute_rate_map_speedwin(self, speed_win):
        '''
        compute the rate map obtained only from data in a speed window
        input:
        ;;speed_win (list of float [2]): the speed window. The rate map will be computed only from the data in this speed window. [speed_thre_low, speed_thre_up]
        output:
        ;;rate_map (np array, shape [n_spatial_bin, n_spatial_bin, n_cell]): the rate map for each cell
        '''
        fire_rate_filt, label_filt = self._filt_data_speedwin(speed_win)
        rate_map = self._compute_rate_map_speedwin(fire_rate_filt, label_filt)
        return rate_map

    def compute_mean_ci_temporal(self, speed_bin_range, ds, ci=0.95, ci_method='se', n_bootstrap=1000):
        '''
        compute the mean firing rate and confidence interval for each speed bin
        input:
        ;;speed_bin_range (list of float [2]): the speed bin range. The rate map will be computed only from the data in this speed window. [speed_thre_low, speed_thre_up]
        ;;ds (float): the bin size for speed
        ;;ci (float): the confidence interval
        ;;ci_method (str): the method for computing confidence interval. 'se' for standard error, 'percentile' for percentile
        ;;n_bootstrap (int): the number of bootstrap, only when ci_method = 'bootsrap'
        output:
        ;;speed_bins (np array, shape [n_speed_bin]): the speed bin for each speed bin
        ;;mean_fr (np array, shape [n_speed_bin, n_cell]): the mean firing rate for each cell
        ;;low_ci (np array, shape [n_speed_bin, n_cell]): the lower bound of the confidence interval for each cell
        ;;up_ci (np array, shape [n_speed_bin, n_cell]): the upper bound of the confidence interval for each cell
        '''
        speed_bins = np.arange(*speed_bin_range, ds)
        speed_label = self.label[:, -1]
        speed_digitize_label = np.digitize(speed_label, speed_bins)

        mean_fr, low_ci, up_ci = [], [], []
        for i in range(1, len(speed_bins)):
            fr_within_bin = self.fire_rate[speed_digitize_label == i]
            m, mmh, mph = mean_confidence_interval(fr_within_bin, confidence=ci, ci_method=ci_method, n_bootstrap=n_bootstrap)

            mean_fr.append(m)
            low_ci.append(mmh)
            up_ci.append(mph)
        return speed_bins, np.array(mean_fr), np.array(low_ci), np.array(up_ci)

    def _compute_linear_regression_temporal(self, fire_rate, speed_label):
        n_cell = fire_rate.shape[1]

        r_values = np.zeros(n_cell)
        p_values = np.zeros(n_cell)
        slopes = np.zeros(n_cell)
        intercepts = np.zeros(n_cell)
        std_errs = np.zeros(n_cell)
        intercept_stderrs = np.zeros(n_cell)

        for i in range(n_cell):
            result = stats.linregress(speed_label, fire_rate[:,i])

            r_values[i] = result.rvalue
            slopes[i] = result.slope
            intercepts[i] = result.intercept
            std_errs[i] = result.stderr
            p_values[i] = result.pvalue
            intercept_stderrs[i] = result.intercept_stderr
        return slopes, intercepts, r_values, p_values, std_errs, intercept_stderrs

    def compute_linear_regression_temporal(self):
        speed_label = self.label[:, -1]
        return self._compute_linear_regression_temporal(self.fire_rate.copy(), speed_label)


    def compute_linear_regression_temporal_shuffle(self, n_shuffle=10):
        n_cell = self.fire_rate.shape[1]
        speed_label = self.label[:, -1]

        slope_shuf, intercept_shuf, r_value_shuf, p_value_shuf = [], [], [], []
        for _ in range(n_shuffle):
            shift_val = np.random.randint(low=0, high=len(speed_label))
            speed_label_shuf = np.roll(speed_label, shift_val)
            slopes, intercepts, r_values, p_values, _, _ = self._compute_linear_regression_temporal(self.fire_rate.copy(), speed_label_shuf)

            slope_shuf.append(slopes)
            intercept_shuf.append(intercepts)
            r_value_shuf.append(r_values)
            p_value_shuf.append(p_values)
        return np.array(slope_shuf), np.array(intercept_shuf), np.array(r_value_shuf), np.array(p_value_shuf)

    def compute_OF_cover_rate(self, speed_bins, n_spatial_bin=50, visited_thre=0.01, x_bound=[-0.75, 0.75], y_bound=[-0.75, 0.75]):
        '''
        compute how large potion of OF being covered by the data points
        input:
        ;;speed_bins: 1D array of speed bins, e.g. np.arange(0.025, 0.1, 0.025)
        '''
        x, y, speed = self.label[:, 0], self.label[:, 1], self.label[:, -1]
        speed_digitize_label = np.digitize(speed, speed_bins)
        cover_rate = np.zeros_like(speed_bins)
        for i in range(len(speed_bins)):
            x_within, y_within = x[speed_digitize_label == i], y[speed_digitize_label == i]
            if len(x_within) == 0:
                cover_rate[i] = 0
                continue

            stay_time = gct.compute_bin_times(x_within, y_within, self.dt, n_spatial_bin, x_bound=x_bound, y_bound=y_bound)
            visited_mask = stay_time >= visited_thre
            cover_rate[i] = np.sum(visited_mask) / visited_mask.size
        return cover_rate

    def compute_avg_fr_speedwin(self, speed_win):
        '''
        compute the average firing rate obtained only from data in a speed window
        input:
        ;;speed_win (list of float [2]): the speed window. The rate map will be computed only from the data in this speed window. [speed_thre_low, speed_thre_up]
        output:
        ;;mean_fr (np array, shape [n_cell]): the mean firing rate for each cell
        '''
        fire_rate_filt, label_filt = self._filt_data_speedwin(speed_win)
        rate_map = self._compute_rate_map_speedwin(fire_rate_filt, label_filt)
        n_cell = rate_map.shape[-1]
        mean_fr = np.mean(rate_map.reshape(-1, n_cell), axis=0)
        return mean_fr

    def compute_avg_fr_speedwin_all(self, speed_win_list, ci=None, n_boot=1000):
        '''
        !!! Not recommand
        compute the average firing rate obtained only from data in a speed window for multiple speed windows. This function computes fr as the mean of ratemap in each speed window, however, it is computationally expensive. Instead, we encourage to directly compute the mean firing rate from the data in each speed window, which is much faster, using compute_mean_ci_temporal. The result, as long as the data set is sufficiently large, should be the same.
        input:
        ;;speed_win_list (list of list of float [2]): the speed window. The rate map will be computed only from the data in this speed window. [speed_thre_low, speed_thre_up]
        ;;ci (float): the confidence interval for bootstrap
        ;;n_boot (int): the number of bootstrap
        output:
        ;;mean_fr (np array, shape [n_speed_win, n_cell]): the mean firing rate for each cell
        ;;lower_bound (np array, shape [n_speed_win, n_cell]): the lower bound of the confidence interval for each cell
        ;;upper_bound (np array, shape [n_speed_win, n_cell]): the upper bound of the confidence interval for each cell
        '''
        mean_fr = []
        if ci is None:
            for speed_win in speed_win_list:
                mean_fr.append(self.compute_avg_fr_speedwin(speed_win))
            return np.array(mean_fr)

        else:
            n_cell = self.fire_rate.shape[1]
            mean_fr = np.zeros((n_boot, len(speed_win_list), n_cell))
            fire_rate_temp, label_temp = self.fire_rate.copy(), self.label.copy()
            for i in range(n_boot):
                indices = np.random.randint(0, label_temp.shape[0], size=label_temp.shape[0])
                self.fire_rate, self.label = fire_rate_temp[indices, :], label_temp[indices, :]

                for j, speed_win in enumerate(speed_win_list):
                    mean_fr[i, j] = self.compute_avg_fr_speedwin(speed_win)
            self.fire_rate, self.label = fire_rate_temp, label_temp

            bootstrap_mean = np.mean(mean_fr, axis=0)
            lower_bound = np.percentile(mean_fr, (1 - ci)/2*100, axis=0)
            upper_bound = np.percentile(mean_fr, (1 + ci)/2*100, axis=0)

            return bootstrap_mean, lower_bound, upper_bound

def mean_confidence_interval(arr, confidence=0.95, ci_method='se', n_bootstrap=1000):
    m = np.mean(arr, axis=0)
    if ci_method == 'se':
        n = arr.shape[0]
        se = stats.sem(arr, axis=0)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        lower_bound, upper_bound = m - h, m + h
    elif ci_method == 'bootstrap':
        lower_bound, upper_bound = np.apply_along_axis(bootstrap_ci, 0, arr, n_bootstrap=n_bootstrap, ci=confidence)
    return m, lower_bound, upper_bound

def bootstrap_ci(arr, n_bootstrap=1000, ci=0.95):
    bootstrap_samples = np.random.choice(arr, size=(n_bootstrap, len(arr)))
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_bound = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

def subset_speed_head(speed, slow_speed_value=0.075, speed_max=0.5):
    '''
    We observed that there are more data at slow speed value which can potentially lead to highly biased fitting result. So we subset the data points with speed lower than a threshold value.
    input:
    ;;speed (np array, shape [n_data]): the speed of each data point
    ;;slow_speed_value (float): the threshold value for the speed. Data points with speed lower than this value will be subsetted
    ;;speed_max (float): the maximum speed value
    output:
    ;;idx (np array, shape [n_data]): the boolean index for the subsetted data points
    '''
    slow_speed_idx = speed < slow_speed_value
    other_speed_idx = (speed > slow_speed_value) & (speed <= speed_max)
    frac_slow_speed = slow_speed_value / (speed_max - slow_speed_value)

    if np.sum(slow_speed_idx) / np.sum(other_speed_idx) > frac_slow_speed:
        target_num_slow_speed_idx = int(np.sum(other_speed_idx) * frac_slow_speed)
        slow_speed_idx = reduce_true_elements(slow_speed_idx, target_num_slow_speed_idx)

    idx = np.logical_or(slow_speed_idx, other_speed_idx)
    return idx
