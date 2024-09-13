
# Speed modulations in grid cell information geometry

## Introduction
This repository contains the code for the paper "Speed modulations in grid cell information geometry".

## Main Repository Structure

- **bin/**: contains excutable files for reproducing paper's results
- **grid_cell/**: contains codes that being used by excutable files
- **global_setting.py**: Contains global hyperparameters.
- **add_python_path**: add this repo to python path
- **env.yml**: suggested environment
- **env_full.yml**: full environment details, may contains redundent packages (windows).
- **fig.ipynb**: main file reproducing figures in the paper
- **gkr_example**: all the codes your need to run gkr, includes a simple example

## Environment Setup
Key packages
```
- Python 3.9
- cudatoolkit 11.8
- cudnn 8.8
- tensorflow 2.10
- gpflow 2.9.0
- hickle
- mpi4py
- seaborn, pandas, numpy, statsmodels, etc
- giotto-ph (for persistent homology analysis)
- Jupyter Notebook (optional for running `fig.ipynb`)
```

`env.yml` provides a simple suggestion of the environment. More comprehensive environment information (exported from Windows) can be found in `env_full.yml`.

## Reproducing Paper Figures

Run `fig.ipynb`. We also provided generated data files in a [wustl_box](https://wustl.box.com/s/9uu905omt7xf48hcepk9c7qi3hl1oyzu)

## Gaussian process with Kernel Regression (GKR_Fitter)
A dataset contains response with shape `(n_data, n_neuron)` and a label with shape `(n_data, n_label)`. GKR assumes that response follows a gaussian distribution, with mean and noise covariance smoothly change with label values. Given a dataset (response and label), the goal of GKR is to fit the mean and noise covariance.

We isolated GKR codes to `gkr_example/gkr.py` and `gpr.py`. See an example usage of GKR in `gkr_example/run_example.py`. To run This example, you need to
1. add current repo to your python path, for example, by running `add_python_path.bat`
2. install python environment (gpflow package's installation instruction includes its corresponding tf installation)
```
- Python 3.9
- cudatoolkit 11.8
- cudnn 8.8
- tensorflow 2.10
- gpflow 2.9.0
- numpy
- sklearn
- scipy
```
3. run `./gkr_example/run_example.py`

Here are simple explainations to  `GKR_Fitter`. Please find the raw codes for more details.
### GKR_Fitter()
#### def __init__(...)
- `n_input`: number of input dimensions, equals to the number of labels (i.e. labels.shape[1])
- `n_output`: number of output dimensions, equals to the number of neurons (i.e. response.shape[1])
- `circular_period`: periodicity of each label variable. It can be:
  - None (default): all label variables share the same kernel parameters, no period
  - a scalar: all label variables share the same kernel parameters, with the same period
  - a list: each dimension gets its specified periodicity and kernel parameters. List can be [None, 2.0], [None, None], or [2.0, 3.0]. The length of list must match to the number of label variables.
- `n_epochs`: number of epochs for fitting noise covariance. We found in practice 10 is pretty good.
- `gpr_params`: dict. parameters for gaussian process regression. For example {'n_incuding': 200} to set number of inducing variables to 200. Some commonly used parameters are
  - `n_inducing`: int. Number of inducing variables in GPR. Default is None. Use inducing variable can boost computation speed.
  - `seperate_kernel`: bool. Whether to seperate kernel for each output dimension (i.e. neurons). If true, the computation time will approximately linearly increases with the number of outputs.
  - `standardize`: bool, whether to standardize data. Default is True.
#### def fit(r, x)
- `r`: neural responses (output) with shape (n_data, n_neuron)
- `x`: labels with shape (n_data, n_labels)

#### def predict(query, return_cov=True):
- `query`: query labels with shape (n_query, n_labels)
- `return_cov`: bool. Default is True. If False, the output is (r_pred, None). If True, the output is (r_pred, cov_pred). Setting to False make the prediction faster if you don't need covariance.

## Acknowledgement ##
This project is impossible without the grid cell spiking data from

Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., Moser, M. B. & Moser, E. I. Toroidal topology of population activity in grid cells. Nature 602, 123–128 (2022).
