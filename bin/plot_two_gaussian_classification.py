import matplotlib.pyplot as plt
import numpy as np
from global_setting import *

# Parameters for the two Gaussian distributions
mean1 = [-1, -1]
mean2 = [1, 1]
cov = [[0.8, -0.5], [0.5, 0.8]]  # shared covariance matrix

# Number of data points
n_points = 50

# Generate data
data1 = np.random.multivariate_normal(mean1, cov, n_points)
data2 = np.random.multivariate_normal(mean2, cov, n_points)

# Plot the data with larger symbols, fewer points, and different symbol for the second distribution
plt.figure(figsize=(3, 3))

# Plot the data points
plt.scatter(data1[:, 0], data1[:, 1], color='black', marker='o', s=100, label='Box 1', alpha=0.5)
plt.scatter(data2[:, 0], data2[:, 1], color='black', marker='+', s=100, label='Box 2', alpha=0.5)

# Remove spines, ticks, and tick labels
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

# Keep the legend
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig(os.path.join(FIGROOT, 'two_gaussian_classification.svg'))

plt.show()
