import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
from grid_cell.ploter import error_bar_plot
from grid_cell.toy_locf import LOCFExperiment, run_experiments
from global_setting import *

n_boot = 5
# base_params = {'n_neuron': 5, 'n_data': 5000, 'std_frac': 0.2, 'dl': 0.1, 'box_size': [0.05], 'sigma': 0.3, 'toy_manifold_name': 'ori'}
base_params = {'n_neuron': 5, 'n_data': 10000, 'std_frac': 0.5, 'dl': 0.2, 'box_size': [0.1, 0.1], 'sigma': 0.3, 'toy_manifold_name': 'place', 'n_gkr_epoch': 20}
toy_manifold_name = base_params['toy_manifold_name']

##################################################
#################### for debugging ################
##################################################
# n_boot = 1
# n_neuron_list = [10, 20]
# accu_neuron_list, plaff_neuron_list, plaff_gkr_neuron_list = run_experiments(n_boot, 'n_neuron', n_neuron_list, base_params)
# fig, axes = plt.subplots(1, 3, figsize=(9, 3))
# cmap = plt.get_cmap('tab10')
# error_bar_plot(n_neuron_list, accu_neuron_list, fig=fig, ax=axes[0], color=cmap(0), label='LOCF accuracy', error_mode='std')
# error_bar_plot(n_neuron_list, plaff_neuron_list, fig=fig, ax=axes[0], color=cmap(1), label='Upper bound from Fisher', error_mode='std')
# error_bar_plot(n_neuron_list, plaff_gkr_neuron_list, fig=fig, ax=axes[0], color=cmap(2), label='Upper bound from GKR Fisher', error_mode='std')
# axes[0].set_xlabel('Number of neurons')
# axes[0].set_ylabel('Accuracy')
# axes[0].legend()
# plt.show()
# exit()

##################################################
#################### Compute  ################
##################################################
# Vary n_neuron
n_neuron_list = [5, 10, 20, 30, 40]
accu_neuron_list, plaff_neuron_list, plaff_gkr_neuron_list = run_experiments(n_boot, 'n_neuron', n_neuron_list, base_params)

# Vary n_data
n_data_list = [3000, 5000, 7000, 9000, 11000]
accu_data_list, plaff_data_list, plaff_gkr_data_list = run_experiments(n_boot, 'n_data', n_data_list, base_params)

# Vary std_frac
std_frac_list = [0.1, 0.2, 0.3, 0.4, 0.5]
accu_std_frac_list, plaff_std_frac_list, plaff_gkr_std_frac_list = run_experiments(n_boot, 'std_frac', std_frac_list, base_params)

data = {'n_neuron_list': n_neuron_list, 'accu_neuron_list': accu_neuron_list, 'plaff_neuron_list': plaff_neuron_list, 'plaff_gkr_neuron_list': plaff_gkr_neuron_list, 'n_data_list': n_data_list, 'accu_data_list': accu_data_list, 'plaff_data_list': plaff_data_list, 'plaff_gkr_data_list': plaff_gkr_data_list, 'std_frac_list': std_frac_list, 'accu_std_frac_list': accu_std_frac_list, 'plaff_std_frac_list': plaff_std_frac_list, 'plaff_gkr_std_frac_list': plaff_gkr_std_frac_list}
hkl.dump(data, os.path.join(DATAROOT, f'test_locf_toy_{toy_manifold_name}.hkl'))

data = hkl.load(os.path.join(DATAROOT, f'test_locf_toy_{toy_manifold_name}.hkl'))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
cmap = plt.get_cmap('tab10')
error_bar_plot(data['n_neuron_list'], data['accu_neuron_list'], fig=fig, ax=axes[0], color=cmap(0), label='LOCF accuracy', error_mode='std')
error_bar_plot(data['n_neuron_list'], data['plaff_neuron_list'], fig=fig, ax=axes[0], color=cmap(1), label='Upper bound from Fisher', error_mode='std')
error_bar_plot(data['n_neuron_list'], data['plaff_gkr_neuron_list'], fig=fig, ax=axes[0], color=cmap(2), label='Upper bound from \n GKR estimated Fisher', error_mode='std')

error_bar_plot(data['n_data_list'], data['accu_data_list'], fig=fig, ax=axes[1], color=cmap(0), label='LOCF accuracy', error_mode='std')
error_bar_plot(data['n_data_list'], data['plaff_data_list'], fig=fig, ax=axes[1], color=cmap(1), label='Upper bound from Fisher', error_mode='std')
error_bar_plot(data['n_data_list'], data['plaff_gkr_data_list'], fig=fig, ax=axes[1], color=cmap(2), label='Upper bound from \n GKR estimated Fisher', error_mode='std')

error_bar_plot(data['std_frac_list'], data['accu_std_frac_list'], fig=fig, ax=axes[2], color=cmap(0), label='LOCF accuracy', error_mode='std')
error_bar_plot(data['std_frac_list'], data['plaff_std_frac_list'], fig=fig, ax=axes[2], color=cmap(1), label='Upper bound from Fisher', error_mode='std')
error_bar_plot(data['std_frac_list'], data['plaff_gkr_std_frac_list'], fig=fig, ax=axes[2], color=cmap(2), label='Upper bound from \n GKR estimated Fisher', error_mode='std')

axes[0].set_xlabel('Number of neurons')
axes[1].set_xlabel('Number of data points')
axes[2].set_xlabel('Noise level')
[ax.legend() for ax in axes]
axes[0].set_ylabel('Accuracy')
# fig.tight_layout()
fig.savefig(os.path.join(FIGROOT, f'test_locf_toy_{toy_manifold_name}.svg'))
plt.show()
