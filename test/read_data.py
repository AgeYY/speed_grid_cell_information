import os
import numpy as np
from global_setting import *

data_path = os.path.join(DATAROOT, 'Toroidal_topology_grid_cell_data', 'rat_q_grid_modules_1_2.npz')
data = np.load(data_path, allow_pickle=True)
print(data.keys())

# Access and print the values for each key
for key in data.keys():
    print(f"\nKey: {key}")
    print(f"Type: {type(data[key])}")
    print(f"Shape: {data[key].shape}")
    
    # Print the actual data (limit output for large arrays)
    array_data = data[key]
    if array_data.size <= 10:  # For small arrays, print everything
        print(f"Values: {array_data}")
    else:  # For larger arrays, print just a sample
        print(f"First few values: {array_data.flat[:5]}...")

