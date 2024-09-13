#!/bin/sh
# run by source add_python_path.sh
export LD_LIBRARY_PATH="/data/zeyuan/miniforge3/envs/grid_cell/lib":$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:"/nfs/zeyuan/GridCellTorus"
export PYTHON_COMPUTING_DEVICE="high_performance_computer_physics"
echo "current device is" $PYTHON_COMPUTING_DEVICE
