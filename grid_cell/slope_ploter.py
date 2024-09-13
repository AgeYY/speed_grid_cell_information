import os
import pprint
import numpy as np
import hickle as hkl
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest
import scipy.stats as stats
from grid_cell.linear_regression import fit_a_line, output_params, draw_line, fit_a_line_boot, output_params_boot, draw_line_boot
from grid_cell.util import fisher_combined_p_value
from global_setting import DATAROOT

class SlopeAnalysis():
    def __init__(self, lole_data, dataset_names, x_key, y_key, shuffle_y_key, dataroot=DATAROOT):
        self.lole_data = lole_data
        self.dataset_names = dataset_names
        self.dataroot = dataroot
        self.r2_dict = {}
        self.shuffle_r2_dict = {}
        self.p_dict = {}
        self.shuffle_p_dict = {}
        self.x_key = x_key
        self.y_key = y_key
        self.shuffle_y_key = shuffle_y_key

    def compute_model_stats(self, x, y, mode='BLR'):
        if mode == 'BLR':
            x = x.flatten(); y = y.flatten();
            model = fit_a_line(x, y)
            slope, _, slope_conf_int, r2, _, p_value_slope = output_params(model)
        elif mode == 'bootstrap':
            model = fit_a_line_boot(x, y, mode=mode)
            slope, _, slope_conf_int, r2, _, p_value_slope = output_params_boot(model)
        elif mode == 'BBLR':
            model = fit_a_line_boot(x, y, mode=mode)
            slope, _, slope_conf_int, r2, _, p_value_slope = output_params_boot(model)
        yerr_up = slope_conf_int[1] - slope
        yerr_down = slope - slope_conf_int[0]
        yerr = np.array([[yerr_down, yerr_up]]).T
        return slope, yerr, r2, p_value_slope, model

    def analyze(self, speed_offset=0, y_text_offset=0.1, data_color='tab:blue', shuffle_data_color='tab:grey', x_shift=0.0, tick_label_size=16, mode='BLR', add_connecting_line=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        xlabel_list = []

        for i, dn in enumerate(self.dataset_names):
            data = self.lole_data[dn]
            x, y, shuffle_y = self._prepare_data(data, speed_offset)

            p_value_two_slope, slope, yerr, shuffle_slope, shuffle_yerr, r2, shuffle_r2, p_value_slope, shuffle_p_value_slope = self._analyze_blr_mode(x, y, shuffle_y, dn, mode=mode)
            self._plot_blr_results(ax, i, slope, yerr, shuffle_slope, shuffle_yerr, r2, shuffle_r2, p_value_slope, shuffle_p_value_slope, x_shift, y_text_offset, data_color, shuffle_data_color, add_connecting_line)

            x_sign = get_significant_text(p_value_two_slope)
            xlabel_list.append(f'{dn.upper()} \n {x_sign}')

        self._finalize_plot(ax, xlabel_list, tick_label_size)
        return fig, ax

    def _prepare_data(self, data, speed_offset):
        x = data[self.x_key][:, speed_offset:]
        y = data[self.y_key][:, speed_offset:]
        shuffle_y = data[self.shuffle_y_key][:, speed_offset:]
        return x, y, shuffle_y

    def _analyze_blr_mode(self, x, y, shuffle_y, dn, mode='BLR'):
        slope, yerr, r2, p_value_slope, model = self.compute_model_stats(x, y, mode=mode)
        self.r2_dict[dn] = r2
        self.p_dict[dn] = p_value_slope

        shuffle_slope, shuffle_yerr, shuffle_r2, shuffle_p_value_slope, shuffle_model = self.compute_model_stats(x, shuffle_y, mode=mode)
        self.shuffle_r2_dict[dn] = shuffle_r2
        self.shuffle_p_dict[dn] = shuffle_p_value_slope

        slope_mean, slope_var = model.posterior_mean[1], model.posterior_cov[1, 1]
        shuffle_slope_mean, shuffle_slope_var = shuffle_model.posterior_mean[1], shuffle_model.posterior_cov[1, 1]
        p_value_two_slope = compute_p_value_diff_two_gaussian(slope_mean, slope_var, shuffle_slope_mean, shuffle_slope_var)
        return p_value_two_slope, slope, yerr, shuffle_slope, shuffle_yerr, r2, shuffle_r2, p_value_slope, shuffle_p_value_slope

    def _plot_blr_results(self, ax, i, slope, yerr, shuffle_slope, shuffle_yerr, r2, shuffle_r2, p_value_slope, shuffle_p_value_slope, x_shift, y_text_offset, data_color, shuffle_data_color, add_connecting_line=False):
        plot_error_bars_with_text(ax, i - x_shift, slope, yerr, data_color, upper_text=f'{r2:.2f}', lower_text='', y_text_offset=y_text_offset)
        plot_error_bars_with_text(ax, i + x_shift, shuffle_slope, shuffle_yerr, shuffle_data_color, upper_text='', lower_text='', y_text_offset=y_text_offset)

        if add_connecting_line:
            ax.plot([i - x_shift, i + x_shift], [slope, shuffle_slope], color='black', linestyle='dashed')

    def _finalize_plot(self, ax, xlabel_list, tick_label_size):
        ax.set_xticks(np.arange(len(self.dataset_names)))
        ax.set_xticklabels(xlabel_list, fontsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=tick_label_size)

    def analyze_value(self, speed_offset=0, y_text_offset=0.1, data_color='tab:blue', shuffle_data_color='tab:grey', x_shift=0.0, mode='BLR', add_connecting_line=True):
        '''
        Let y = kx + b, we obtain the distribution of y_bar = k x_bar + b, where x_bar is the mean of x, and draw figures.
        '''
        fig, ax = plt.subplots(figsize=(10, 5))
        xlabel_list = []
        for i, dn in enumerate(self.dataset_names):
            data = self.lole_data[dn]
            x, y, shuffle_y = self._prepare_data(data, speed_offset)

            if mode == 'BLR':
                p_value, y_bar_mean, y_bar_err, shuffle_y_bar_mean, shuffle_y_bar_err = self._analyze_value_blr_mode(x, y, shuffle_y, dn)
            elif mode == 'bootstrap':
                y_bar_mean, y_bar_var, y_bar_err = compute_xmean_err(y)
                shuffle_y_bar_mean, shuffle_y_bar_var, shuffle_y_bar_err = compute_xmean_err(shuffle_y)
                p_value = compute_p_value_diff_two_gaussian(y_bar_mean, y_bar_var, shuffle_y_bar_mean, shuffle_y_bar_var)

            # plot
            plot_error_bars_with_text(ax, i - x_shift, y_bar_mean, y_bar_err, data_color, upper_text='', lower_text='', y_text_offset=y_text_offset)
            plot_error_bars_with_text(ax, i + x_shift, shuffle_y_bar_mean, shuffle_y_bar_err, shuffle_data_color, upper_text='', lower_text='', y_text_offset=y_text_offset)
            if add_connecting_line:
                ax.plot([i - x_shift, i + x_shift], [y_bar_mean, shuffle_y_bar_mean], color='black', linestyle='dashed')

            x_sign = get_significant_text(p_value)
            xlabel_list.append(f'{dn.upper()} \n {x_sign}')

        ax.set_xticks(np.arange(len(self.dataset_names)))
        ax.set_xticklabels(xlabel_list, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)

        return fig, ax

    def _analyze_value_blr_mode(self, x, y, shuffle_y, dn):
        x = x.flatten()
        y = y.flatten()
        shuffle_y = shuffle_y.flatten()

        x_bar = np.array([1, np.mean(x)]).reshape(1, 2)

        y_bar_mean, y_bar_var, y_bar_err = self._compute_blr_statistics(x, y, x_bar)
        y_bar_mean_shuffle, y_bar_var_shuffle, y_bar_err_shuffle = self._compute_blr_statistics(x, shuffle_y, x_bar)

        p_value = compute_p_value_diff_two_gaussian(
            y_bar_mean, y_bar_var, y_bar_mean_shuffle, y_bar_var_shuffle)
        return p_value, y_bar_mean, y_bar_err, y_bar_mean_shuffle, y_bar_err_shuffle

    def _compute_blr_statistics(self, x, y, x_bar):
        model = fit_a_line(x, y)
        y_bar_mean, y_bar_var = model.predict(x_bar)
        y_bar_mean, y_bar_var = y_bar_mean[0], y_bar_var[0]
        y_bar_ci = compute_confidence_interval(y_bar_mean, y_bar_var, alpha=0.05)
        y_bar_err = np.array([[y_bar_mean - y_bar_ci[0], y_bar_ci[1] - y_bar_mean]]).T
        return y_bar_mean, y_bar_var, y_bar_err

def get_significant_text(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return 'NS'

def plot_error_bars_with_text(ax, x_pos, slope, yerr, color, upper_text=None, lower_text=None, y_text_offset=0):
    ax.errorbar([x_pos], [slope], yerr=yerr, fmt='o', color=color, ecolor=color, capsize=10, markersize=10)
    y_text_offset = (yerr[1] + yerr[0]) * y_text_offset
    ax.text(x_pos, slope + yerr[0] + y_text_offset, upper_text, color=color, ha='center', fontsize=15)
    ax.text(x_pos, slope - yerr[1] - y_text_offset * 3, lower_text, color=color, ha='center', fontsize=15)
    return ax

def draw_two_line_data(speed_flat, y_flat, shuffle_y_flat, ax, data_label='Data', shuffle_label='Shuffled Data', color_data='tab:blue', color_shuffle='tab:grey'):
    x = speed_flat
    model = fit_a_line(x, y_flat)
    slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params(model)
    draw_line(ax, x, y_flat, model, color=color_data, line_label=None, data_label=data_label)
    print(f'R-Squared: {model.r_squared:.2f}')
    print(f'p-value: {model.p_values[0]:.5f}, {model.p_values[1]:.5f}')

    if shuffle_y_flat is not None:
        shuffle_model = fit_a_line(x, shuffle_y_flat)
        slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params(shuffle_model)
        print(f'R-Squared shuffle: {shuffle_model.r_squared:.2f}')
        print(f'p-value shuffle: {shuffle_model.p_values[0]:.5f}, {shuffle_model.p_values[1]:.5f}')
        draw_line(ax, x, shuffle_y_flat, shuffle_model, color=color_shuffle, line_label=None, data_label=shuffle_label)

    return ax

def draw_two_line_data_boot(speed, y, shuffle_y, ax, data_label='Data', shuffle_label='Shuffled Data', color_data='tab:blue', color_shuffle='tab:grey', mode='bootstrap'):
    '''
    speed: (n_boot, m)
    y: (n_boot, m)
    mode: 'bootstrap' or 'BBLR'
    '''
    x = speed
    model = fit_a_line_boot(x, y, mode=mode)
    slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params(model)

    draw_line_boot(ax, x, y, model, color=color_data, line_label=None, data_label=data_label, mode=mode)
    print(f'R-Squared: {model.r_squared:.2f}')
    print(f'p-value: {model.p_values[0]:.5f}, {model.p_values[1]:.5f}')

    if shuffle_y is not None:
        shuffle_model = fit_a_line_boot(x, shuffle_y, mode=mode)
        slope, intercept, slope_conf_int, r2, p_value_intercept, p_value_slope = output_params_boot(shuffle_model)
        print(f'R-Squared shuffle: {shuffle_model.r_squared:.2f}')
        print(f'p-value shuffle: {shuffle_model.p_values[0]:.5f}, {shuffle_model.p_values[1]:.5f}')
        draw_line_boot(ax, x, shuffle_y, shuffle_model, color=color_shuffle, line_label=None, data_label=shuffle_label, mode=mode)

    return ax

def add_small_noise(data, eps=1e-5):
    return data + np.random.normal(0, scale=eps, size=data.shape)

def compute_confidence_interval(mean, var, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    lower = mean - z * np.sqrt(var)
    upper = mean + z * np.sqrt(var)
    return lower, upper

def compute_p_value_diff_two_gaussian(y_mean_1, y_var_1, y_mean_2, y_var_2):
    mean_diff = y_mean_1 - y_mean_2
    cov_diff = y_var_1 + y_var_2
    p_value = stats.norm.cdf(0, mean_diff, np.sqrt(cov_diff))
    p_value = 2 * min(p_value, 1 - p_value)
    return p_value

def compute_xmean_err(arr):
    '''
    arr: (n_boot, n_x)
    '''
    arr_bar_boot = np.mean(arr, axis=1)
    arr_bar_mean, arr_bar_var = np.mean(arr_bar_boot), np.var(arr_bar_boot)
    arr_bar_ci = compute_confidence_interval(arr_bar_mean, arr_bar_var, alpha=0.05)
    arr_bar_err = np.array([[arr_bar_mean - arr_bar_ci[0], arr_bar_ci[1] - arr_bar_mean]]).T
    return arr_bar_mean, arr_bar_var, arr_bar_err
