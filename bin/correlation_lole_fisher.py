# please run lole.py and noise_fisher.py first
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import hickle as hkl
from global_setting import *

lole_path = os.path.join(DATAROOT, 'lole_speed_mse.hkl')
fisher_path = os.path.join(DATAROOT, 'noise_geo_data.hkl')

lole_speed = hkl.load(lole_path)['speed_pca_list']
lole_mse = hkl.load(lole_path)['mse_pca_list']
fisher_speed = hkl.load(fisher_path)['speed']
fisher = hkl.load(fisher_path)['fisher']

# check whether lole_speed and fisher_speed are the same
assert np.allclose(lole_speed, fisher_speed), 'lole_speed and fisher_speed are not the same'

# compute the correlation between lole_mse and fisher
# correlation = np.corrcoef(np.log(lole_mse), np.log(fisher))[0, 1]
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(lole_mse), np.log(fisher))
print(f'Correlation: {r_value}')

fitted_line = slope * np.log(lole_mse) + intercept

# print(correlation)
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(np.log(lole_mse), np.log(fisher), color='k')
ax.plot(np.log(lole_mse), fitted_line, color='k', linestyle='--')

ax.set_xlabel('log(LOLE MSE) (a.u.)')
ax.set_ylabel('log(Noise Fisher) (a.u.)')
fig.savefig(os.path.join(FIGROOT, 'correlation_lole_fisher.svg'))
plt.show()
