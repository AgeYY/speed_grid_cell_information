# Speed Modulations in Grid Cell Information Geometry

## Overview
This repository contains the implementation of the Gaussian Process with Kernel Regression (GKR) method, along with the code to reproduce figures from our paper, "Speed Modulations in Grid Cell Information Geometry." (doi: 10.1101/2024.09.18.613797)

## Quick Start with GKR
For those interested in using only the GKR method, we provide a standalone `GKR_demo.ipynb` notebook that can be run on Google Colab or locally.

### Local Setup
1. Install Python 3.10 (this version has been tested).
2. Install the required dependencies:
   ```bash
   pip install -r requirements_gkr_demo.txt
   ```

### Optional GPU Support
To enable GPU support, install CUDA dependencies via conda:
```bash
conda install cudatoolkit=11.2 cudnn=8.1
```

### GKR_Fitter Class
```python
class GKR_Fitter:
    def __init__(n_input, n_output, circular_period=None, n_epochs=10, gpr_params=None):
        """
        Parameters:
        - n_input: int, number of input dimensions (labels.shape[1])
        - n_output: int, number of output dimensions/neurons (response.shape[1])
        - circular_period: None, float, or list
            - None: No periodicity, shared kernel parameters
            - float: Same period for all variables
            - list: Individual periods per dimension, e.g., [None, 2.0]
        - n_epochs: int, number of fitting iterations for noise covariance
        - gpr_params: dict, GPR configuration
            - n_inducing: int, number of inducing variables, default = None, do not use inducing
            - separate_kernel: bool, use separate kernels per neuron, default = False
            - standardize: bool, standardize input data, default = True
        """

    def fit(r, x):
        """
        Fit the model
        - r: array (n_data, n_neuron), neural responses
        - x: array (n_data, n_labels), input labels
        """

    def predict(query, return_cov=True):
        """
        Make predictions
        - query: array (n_query, n_labels), query points
        - return_cov: bool, whether to return covariance
        Returns: (predictions, covariance) or (predictions, None)
        """
```

## Reproducing Paper Figures

### Setup
1. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Optionally, install CUDA dependencies via conda:
   ```bash
   conda install cudatoolkit=11.2 cudnn=8.1
   ```

2. Download the data:
   - Obtain raw grid cell data from [Figshare](https://figshare.com/articles/dataset/Toroidal_topology_of_population_activity_in_grid_cells/16764508?file=35078602).
   - (Optional) Download pre-generated data from [WUSTL Box](https://wustl.box.com/s/9uu905omt7xf48hcepk9c7qi3hl1oyzu) to skip multiple steps in `paper.ipynb`.

3. Organize the data in the following structure:
   ```
   data/
   ├── Toroidal_topology_grid_cell_data/
   │   ├── rat_q_grid_modules_1_2.npz
   │   └── ...
   └── [pre-generated data]
   ```

4. Run `fig.ipynb` to generate the figures.

Note that by default, the data will be in `./data/`. The user can change the data path by exporting an environmental variable `GRID_CELL_DATAROOT` (see details in `global_setting.py`).

## Acknowledgments
This work builds upon grid cell spiking data from:

Gardner, R.J., Hermansen, E., Pachitariu, M. et al. Toroidal topology of population activity in grid cells. Nature 602, 123–128 (2022). https://doi.org/10.1038/s41586-021-04268-7