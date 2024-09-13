#!/bin/sh
# run by source add_python_path.sh
export PYTHONPATH=$PYTHONPATH:"/home/y.zeyuan/GridCellTorus"
export PYTHONPATH=$PYTHONPATH:"/home/y.zeyuan/disentangle"
export PYTHON_COMPUTING_DEVICE="high_performance_computer"
export PATH="/storage1/fs1/ralfwessel/Active/conda/envs/grid_cell/bin":$PATH
export LD_LIBRARY_PATH="/storage1/fs1/ralfwessel/Active/conda/envs/grid_cell/lib":$LD_LIBRARY_PATH
echo "current device is" $PYTHON_COMPUTING_DEVICE

# This file is created under windows, if you want to run code in linux, please remove this file and create an identical file in linux. Otherwise there would be strange tailing characters in the file. Please also remember to change python_computing device in global_setting.py
