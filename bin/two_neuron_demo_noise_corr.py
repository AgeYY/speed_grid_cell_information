import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from global_setting import *

# Generate synthetic binary classification data
np.random.seed(42)

# Define two class centers (representing two spatial locations)
class_1_center = np.array([1, 0])  # Neuron 1 active, Neuron 2 inactive
class_2_center = np.array([0, 1])  # Neuron 2 active, Neuron 1 inactive

# Generate samples for each class
n_samples = 500

# Compute Fisher Information for each noise condition
def compute_fisher_information(cov_matrix):
    """ Computes the Fisher Information for the binary classification case. """
    delta_mu = np.array(class_2_center - class_1_center).reshape(2, 1)  # Difference in means
    inv_cov = np.linalg.inv(cov_matrix)  # Inverse of covariance matrix
    FI = delta_mu.T @ inv_cov @ delta_mu  # Compute Fisher Information
    return FI.item()  # Extract scalar value

def generate_data(cov_matrix, n_samples=500):
    """ Generates class 1 and class 2 samples with given covariance structure. """
    class_1_samples = np.random.multivariate_normal(class_1_center, cov_matrix, n_samples)
    class_2_samples = np.random.multivariate_normal(class_2_center, cov_matrix, n_samples)
    
    X = np.vstack([class_1_samples, class_2_samples])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    return X, y

# Compute the scaling factor to normalize trace to 2
def normalize_cov(cov_matrix, target_trace=2.0):
    current_trace = np.trace(cov_matrix)
    return cov_matrix * (target_trace / current_trace)

# Define covariance matrices with equal trace
cov_beneficial = normalize_cov(np.array([[1.0, 0.8], [0.8, 1.0]]))
cov_detrimental = normalize_cov(np.array([[1.0, -0.8], [-0.8, 1.0]]))
cov_independent = normalize_cov(np.array([[1.0, 0.0], [0.0, 1.0]]))  # Independent noise
total_noise = np.trace(cov_independent)

# Generate datasets
X_beneficial, y_beneficial = generate_data(cov_beneficial, n_samples)
X_detrimental, y_detrimental = generate_data(cov_detrimental, n_samples)
X_independent, y_independent = generate_data(cov_independent, n_samples)

# Train classifiers
clf_beneficial = LogisticRegression().fit(X_beneficial, y_beneficial)
clf_detrimental = LogisticRegression().fit(X_detrimental, y_detrimental)
clf_independent = LogisticRegression().fit(X_independent, y_independent)

# Compute accuracy
accuracy_beneficial = accuracy_score(y_beneficial, clf_beneficial.predict(X_beneficial))
accuracy_detrimental = accuracy_score(y_detrimental, clf_detrimental.predict(X_detrimental))
accuracy_independent = accuracy_score(y_independent, clf_independent.predict(X_independent))

# Compute Fisher Information for each case
FI_beneficial = compute_fisher_information(cov_beneficial)
FI_detrimental = compute_fisher_information(cov_detrimental)
FI_independent = compute_fisher_information(cov_independent)

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(8, 3))

cases = [
    (X_beneficial, y_beneficial, clf_beneficial, "Beneficial Correlation", "blue", accuracy_beneficial, FI_beneficial),
    (X_independent, y_independent, clf_independent, "Independent Noise", "green", accuracy_independent, FI_independent),
    (X_detrimental, y_detrimental, clf_detrimental, "Detrimental Correlation", "red", accuracy_detrimental, FI_detrimental),
]

for i, (X, y, clf, title, color, accuracy, FI) in enumerate(cases):
    ax[i].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.5, marker='x')

    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[i].contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

    # Plot class centers
    ax[i].scatter([class_1_center[0]], [class_1_center[1]], color='black', marker='*', s=200, label='Class 1 Center')
    ax[i].scatter([class_2_center[0]], [class_2_center[1]], color='black', marker='*', s=200, label='Class 2 Center')

    ax[i].set_xlim(-2, 2)
    ax[i].set_ylim(-2, 2)
    ax[i].set_title(f"{title}\nClassification Accuracy: {accuracy:.2f}\nFisher Information: {FI:.2f}")
    ax[i].set_xlabel("Neuron 1 Activity")
    if i == 0:  # Only show y label for leftmost plot
        ax[i].set_ylabel("Neuron 2 Activity")
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(FIGROOT, "two_neuron_demo_noise_corr.svg"))
plt.show()