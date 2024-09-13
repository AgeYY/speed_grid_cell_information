import numpy as np
from scipy.ndimage import gaussian_filter # for smoothing grid cell tuning
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.signal import correlate2d
from scipy.ndimage import rotate, correlate
from scipy.stats import binned_statistic_2d
from scipy.stats import zscore
from filterpy.kalman import KalmanFilter
import os

def compute_bin_times(x, y, dt, n_bins, x_bound=None, y_bound=None):
    '''
    Compute the time spent in each bin
    input:
        x, y: position. Each is a 1d array. The bound of open field is about [-0.75, 0.75]
        dt: time, float. time spacing for every position.
        n_bins: number of bins in each dimension
        x_bound, y_bound: the bound of the position. If None, use the min and max of x, y
    output:
        bin_times: 2d array. The time spent in each bin
    '''
    x_bound = [min(x), max(x)] if x_bound is None else x_bound
    y_bound = [min(y), max(y)] if y_bound is None else y_bound

    EPS = 1e-8
    # Bin edges
    binx = np.linspace(x_bound[0] - EPS, x_bound[1] + EPS, n_bins+1, endpoint=True)
    biny = np.linspace(y_bound[0] - EPS, y_bound[1] + EPS, n_bins+1, endpoint=True)

    # Bin indices for each position
    bin_x_indices = np.digitize(x, binx)
    bin_y_indices = np.digitize(y, biny)
    bin_x_indices = np.clip(bin_x_indices, 0, n_bins-1)
    bin_y_indices = np.clip(bin_y_indices, 0, n_bins-1)

    # Array to store times
    bin_times = np.zeros((n_bins, n_bins))

    # Add dt to corresponding bin
    for i in range(len(x)):
        bin_times[bin_x_indices[i], bin_y_indices[i]] += dt

    return bin_times

def compute_speed(x, y, t, smooth_sigma=None):
    '''
    Compute the speed
    input:
        x, y: position. Each is a 1d array
        t: time. 1d array. Same length as x, y
    output:
        speed: 1d array. The speed at each time point
    '''
    if smooth_sigma is None:
        x_smooth, y_smooth = x, y
    else:
        # Apply a simple smoothing filter (optional)
        x_smooth = np.convolve(x, np.ones(smooth_sigma)/smooth_sigma, mode='same')
        y_smooth = np.convolve(y, np.ones(smooth_sigma)/smooth_sigma, mode='same')
    # Central differences for interior points
    dx = np.gradient(x_smooth, t)
    dy = np.gradient(y_smooth, t)

    # Calculate speed
    speed = np.sqrt(dx**2 + dy**2)
    return speed

def compute_speed_kalman(x_positions, y_positions, times, process_noise=1, measurement_noise=100):
    """
    Estimate speed using a Kalman Filter given position and time data.
    
    Parameters:
    x_positions (array): Array of x-coordinates.
    y_positions (array): Array of y-coordinates.
    times (array): Array of timestamps.
    process_noise (float): Process noise variance for the Q matrix.
    measurement_noise (float): Measurement noise variance for the R matrix.
    
    Returns:
    speeds (array): Estimated speeds at each time point.
    """
    dt = np.diff(times)
    dt = np.append(dt, dt[-1])  # Assume last interval is same as second to last

    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt[0], 0],
                     [0, 1, 0, dt[0]],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= measurement_noise
    kf.Q = np.eye(4) * process_noise
    kf.x = np.array([x_positions[0], y_positions[0], 0, 0])

    # Placeholder for estimated speeds
    speeds = np.zeros(len(x_positions))

    # Run Kalman Filter
    for i in range(1, len(x_positions)):
        kf.F = np.array([[1, 0, dt[i], 0],
                         [0, 1, 0, dt[i]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.predict()
        kf.update(np.array([x_positions[i], y_positions[i]]))
        speeds[i] = np.sqrt(kf.x[2]**2 + kf.x[3]**2)

    return speeds


def filter_spikes_by_speed(cell0, t, speed, speed_thre, speed_thre2=None):
    '''
    Filter spikes by speed. Only include spiking times when the speed is above the threshold. Speed can be computed from compute_speed
    input:
        cell0: 1d array. The spike times
        t: 1d array. The time points
        speed: 1d array. The speed at each time point
        speed_thre: float. The threshold of speed
    output:
        cell0: 1d array. The spike times after filtering
    '''
    indices = np.searchsorted(t, cell0) - 1 # output of searchsoted values from 1 to maximum index, if all values of cell0 do not exceed t
    if speed_thre2 is not None:
        valid_indices = indices[(speed[indices] > speed_thre) * (speed[indices] <= speed_thre2)]
    valid_indices = indices[speed[indices] > speed_thre]
    return t[valid_indices]

def filter_xyt_by_speed(x, speed, speed_thre, speed_thre2=None):
    if speed_thre2 is not None:
        valid_indices = (speed > speed_thre) * (speed <= speed_thre2)
    else:
        valid_indices = (speed > speed_thre)
    return x[valid_indices]

def spatial_bin_spikes_count(cell0, t, x, y, n_bin, x_bound=None, y_bound=None, verbose=False):
    '''
    Compute the number of spikes in each spatial bin
    input:
        cell0: 1d array. The spike times
        t: 1d array. The time points
        x, y: position. Each is a 1d array. Same length as t. The bound of open_field is about [-0.75, 0.75]
        n_bin: int. Number of bins in each dimension
        x_bound, y_bound: the bound of the position
        verbose: bool. If True, return the positions of each bin
    output:
        spike_count: 2d array. The number of spikes in each bin
        positions: 3d array.  The positions of each bin. Only returned when verbose=True
    '''
    x_bound = [min(x), max(x)] if x_bound is None else x_bound
    y_bound = [min(y), max(y)] if y_bound is None else y_bound

    EPS = 1e-8
    # obtain the position at each spike
    spike_idx = np.searchsorted(t, cell0) - 1
    x_spike, y_spike = x[spike_idx], y[spike_idx]

    binx = np.linspace(x_bound[0] - EPS, x_bound[1] + EPS, n_bin+1)
    biny = np.linspace(y_bound[0] - EPS, y_bound[1] + EPS, n_bin+1)
    spike_count, _, _ = np.histogram2d(x_spike, y_spike, bins=[binx, biny])
    if verbose:
        x_positions, y_positions = np.meshgrid((binx[1:] + binx[:-1]) / 2, (biny[1:] + biny[:-1]) / 2)
        positions = np.stack([x_positions, y_positions], axis=-1)
        return spike_count, positions
    else:
        return spike_count

def compute_firing_rate(cell0, t, sigma=10, z_score=True):
    '''
    Compute and return the z-scored firing rate.
    :param cell0: An array of spike times.
    :param t: An array of times.
    :param sigma: The standard deviation of the Gaussian filter, smoothing the firing rate. When converting to time, sigma time = sigma * dt
    :return:
      fire_rate: The z-scored firing rate. Unit is Hz
    '''
    # Firing count at each time step
    dt = np.mean(np.diff(t))
    indices = np.searchsorted(t, cell0, side='right')
    indices = indices[indices < len(t)] # remove spikes times exceeding the time window
    fire_rate = np.bincount(indices, minlength=len(t)).astype(float) / dt # unit is Hz

    # Gaussian smoothing
    if sigma is not None:
        fire_rate_smooth = gaussian_filter(fire_rate, sigma=sigma)
    else:
        fire_rate_smooth = fire_rate

    # Convert to z-score
    if z_score:
        fire_rate_z = zscore(fire_rate_smooth)
    else:
        fire_rate_z = fire_rate_smooth
    fire_rate_z = np.nan_to_num(fire_rate_z)
    return fire_rate_z

def compute_all_firing_rates(spikes_mod, t, sigma, z_score=True):
    n_cells = len(spikes_mod)
    n_t = len(t)
    firing_rates_matrix = np.zeros((n_t, n_cells))

    for cell, spikes in spikes_mod.items():
        firing_rates_matrix[:, cell] = compute_firing_rate(spikes, t, sigma, z_score=z_score)
    return firing_rates_matrix

def adaptive_smooth_firing_rate(firing_rate, speed, base_speed=0.6, base_sigma=1, max_sigma=20, min_sigma=1):
    '''
    Adaptive smooth the firing rate based on the speed
    :param firing_rate: 1d array. The firing rate
    :param speed: 1d array. The speed at each time point
    :param sigma_base_line: list of 2 floats. The base line sigma for low and high speed
    :return: 1d array. The smoothed firing rate
    '''
    sigma_speed = speed / base_speed * base_sigma
    sigma_speed = np.clip(sigma_speed, min_sigma, max_sigma)
    gaussian_windows = get_gaussian_window(sigma_speed)
    firing_rate_smooth = apply_gaussian_window(firing_rate, gaussian_windows)
    return firing_rate_smooth

def get_sigma_speed(speed, base_speed, base_sigma, max_sigma=20, min_sigma=1):
    sigma_speed = base_speed / speed * base_sigma
    sigma_speed = np.clip(sigma_speed, min_sigma, max_sigma)
    return sigma_speed

def get_gaussian_window(sigmas):
    gaussian_windows = []
    for i in range(len(sigmas)):
        # Generate a Gaussian window centered around the current element
        window_size = int(6 * sigmas[i])  # To cover +/- 3 sigma range
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        gauss_win = gaussian(window_size, std=sigmas[i])
        gauss_win /= gauss_win.sum()  # Normalize the window
        gaussian_windows.append(gauss_win)
    return gaussian_windows

def apply_gaussian_window(arr, gaussian_windows):
    filtered_arr = np.zeros_like(arr)
    # Calculate maximum window size for padding
    max_window_size = max(len(window) for window in gaussian_windows)

    # Pad the array once outside the loop
    padded_arr = np.pad(arr, (max_window_size // 2, max_window_size // 2), mode='reflect')

    for i, window in enumerate(gaussian_windows):
        window_size = len(window)
        # Apply the Gaussian window to the array
        filtered_arr[i] = np.dot(padded_arr[i + max_window_size//2 - window_size//2:i + max_window_size//2 + window_size//2 + 1], window)
    return filtered_arr

def spatial_bin_fire_rate(fire_rate, x, y, n_bin, x_bound=None, y_bound=None, verbose=False):
    x_bound = [min(x), max(x)] if x_bound is None else x_bound
    y_bound = [min(y), max(y)] if y_bound is None else y_bound

    EPS = 1e-8
    x_bins = np.linspace(x_bound[0] - EPS, x_bound[1] + EPS, n_bin+1)
    y_bins = np.linspace(y_bound[0] - EPS, y_bound[1] + EPS, n_bin+1)
    fire_rate_binned, _, _, _, = binned_statistic_2d(x, y, fire_rate, statistic='mean', bins=[x_bins, y_bins])
    if verbose:
        x_positions, y_positions = np.meshgrid((x_bins[1:] + x_bins[:-1]) / 2, (y_bins[1:] + y_bins[:-1]) / 2)
        positions = np.stack([x_positions, y_positions], axis=-1)
        return fire_rate_binned, positions
    else:
        return fire_rate_binned

def compute_autocorrelation(mat1, mat2, mode='same', boundary='fill', fillvalue=0):
    mat1 = (mat1 - np.mean(mat1)) / (np.std(mat1) * len(mat1))
    mat2 = (mat2 - np.mean(mat2)) / (np.std(mat2) * len(mat2))
    auto_corr = correlate2d(mat1, mat2, mode=mode, boundary=boundary, fillvalue=fillvalue)
    return auto_corr

def compute_autocorrelation_ori(matrix, angles, center_block_frac=0.15, mode='constant'):
    correlations = np.zeros_like(angles, dtype=np.float32)
    original_center = np.array([dim // 2 for dim in matrix.shape])
    radius_sq = min(original_center)**2

    for i, angle in enumerate(angles):
        rotated = rotate(matrix, angle, reshape=False, mode=mode)
        rotated_center = np.array([dim // 2 for dim in rotated.shape])
        y, x = np.ogrid[-rotated_center[0]:rotated.shape[0]-rotated_center[0], -rotated_center[1]:rotated.shape[1]-rotated_center[1]]
        region_mask = x**2 + y**2 <= radius_sq
        region_mask2 = x**2 + y**2 > radius_sq * center_block_frac
        region_mask = np.logical_and(region_mask, region_mask2)

        # # visualizaze the blocked image
        # matrix_cp = rotated.copy()
        # matrix_cp[~region_mask] = 0
        # plt.figure()
        # plt.imshow(matrix_cp)
        # plt.show()

        correlations[i] = np.corrcoef(matrix[region_mask].flatten(), rotated[region_mask].flatten())[0,1]
    
    return correlations

def gridness(rate_map, padding=10):
    rate_map_padding = np.pad(rate_map, padding, mode='linear_ramp')
    auto_corr = compute_autocorrelation(rate_map_padding, rate_map_padding)
    # plt.figure()
    # plt.imshow(auto_corr)
    # plt.show()
    angles = [30, 60, 90, 120, 150]
    correlations = compute_autocorrelation_ori(auto_corr, angles)
    score = (correlations[1] + correlations[3]) / 2 - (correlations[0] + correlations[2] + correlations[4]) / 3
    return score

def ppdata_to_avg_ratemap(n_selections, feamap, label, n_bins=50, x_bound=[-0.75, 0.75], y_bound=[-0.75, 0.75], output_pos=False):
    '''
    Compute the average ratemap from the feamap and label
    input:
        n_selections: int. Number of randomly selected samples
        feamap: 2d array. The feamap
        label: 2d array. The label
    output:
        ratemap: 3d array [n_bins, n_bins, n_pca]. The average ratemap
    '''
    if n_selections is None:
        n_selections = feamap.shape[0]
    n_pca = feamap.shape[1]
    selected_rows = np.random.choice(feamap.shape[0], n_selections, replace=True)
    feamap_boot, label_boot = feamap[selected_rows, :], label[selected_rows, :]

    x, y = label_boot[:, 0], label_boot[:, 1]
    n_cell = feamap_boot.shape[1]
    rate_map = np.zeros([n_spatial_bin, n_spatial_bin, n_cell])
    for i in range(n_cell):
        rate_map[:, :, i], positions = spatial_bin_fire_rate(feamap_boot[:, i], x, y, n_bin=n_bins, x_bound=x_bound, y_bound=y_bound, verbose=True)

    if output_pos:
        return ratemap, pos
    else:
        return ratemap
