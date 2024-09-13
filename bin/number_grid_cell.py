import os
import hickle as hkl
from global_setting import *

dataset_name = ['r1m1', 'r1m2', 'r1m3', 'r2m1', 'r2m2', 'r2m3', 's1m1', 'q1m1', 'q1m2']
preprocessed_file_name = 'preprocessed_data_OF.hkl'
data = hkl.load(os.path.join(DATAROOT, preprocessed_file_name))

for dn in dataset_name:
    print(dn)
    print(data[dn]['feamap'].shape)
