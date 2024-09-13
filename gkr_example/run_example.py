import numpy as np
import matplotlib.pyplot as plt
from gkr_example.gkr import GKR_Fitter

def generate_circular_dataset(n_data, radius=1.0, noise_factor=0.1):
    # Generate angles for circle (between 0 and 2*pi)
    angles = np.linspace(0, 2 * np.pi, n_data)
    
    # Generate circle coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Stack the x, y coordinates to form the response (n_data, 2)
    response = np.stack((x, y), axis=1)
    
    # Add noise to the response
    noise = np.random.normal(scale=noise_factor, size=response.shape)
    response_noisy = response + noise
    
    # Generate scalar labels
    labels = angles.reshape(-1, 1)  # Labels could be scalar, here using angle as label
    
    return response_noisy, labels

# Example usage
n_data = 100
response_noisy, labels = generate_circular_dataset(n_data, radius=1.0, noise_factor=0.1)

# Plot the noisy circle
plt.scatter(response_noisy[:, 0], response_noisy[:, 1], c=labels.ravel(), cmap='viridis')
plt.title("Noisy Circular Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal', adjustable='box')

gkr = GKR_Fitter(n_input=labels.shape[1], n_output=response_noisy.shape[1], circular_period=2*np.pi)

gkr.fit(response_noisy, labels)

# Generate labels prediction
label_pred = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
response_pred, response_cov = gkr.predict(label_pred)

# Plot the predicted circle
plt.plot(response_pred[:, 0], response_pred[:, 1], 'r-', label='Predicted Manifold')
plt.legend()

plt.show()

