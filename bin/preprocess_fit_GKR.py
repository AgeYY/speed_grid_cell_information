# preprocess and fitting GKR model
import numpy as np
import os
import hickle as hkl
from grid_cell.grid_cell_processor import Grid_Cell_Processor, Data_Transformer
from sklearn.decomposition import PCA
from grid_cell.gkr import GKR_Fitter
import argparse
from global_setting import *

parser = argparse.ArgumentParser(description='Preprocess and fit GKR model')

# Add arguments
parser.add_argument('--mouse_name', type=str, default='R', help='Rat\'s name')
parser.add_argument('--day', type=str, default='day1', help='day')
parser.add_argument('--module', type=str, default='1', help='Grid cell module id')
parser.add_argument('--session', type=str, default='open_field_1', help='task session')
parser.add_argument('--fr_smooth_sigma', type=int, default=1, help='Fire rate smoothing sigma')
parser.add_argument('--downsample_rate_gen', type=int, default=5, help='Downsample rate for data generation')
parser.add_argument('--downsample_rate_fit', type=int, default=10, help='Downsample rate for data fitting')
parser.add_argument('--pca_component', type=int, default=6, help='Number of PCA components')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs for GKR fitting')
parser.add_argument('--regenerate_data', action='store_true', help='Regenerate preprocessed data')
parser.add_argument('--adaptive_fr_sigma', action='store_true', help='Adaptive fire rate smoothing sigma. Bool')
parser.add_argument('--n_inducing', type=int, default=500, help='Number of inducing points for GKR fitting')

args = parser.parse_args()

#################### Hyperparameters ####################
preprocessed_dir = os.path.join(DATAROOT, 'preprocessed/')
preprocessed_file_name = 'preprocessed_data_{}_{}_{}_{}_{}_{}_{}.npz'.format(args.mouse_name, args.module, args.day, args.session, args.fr_smooth_sigma, args.downsample_rate_gen, args.adaptive_fr_sigma)
preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file_name)

if not os.path.exists(preprocessed_path) or args.regenerate_data:
    print('Preprocessed data does not exist, or force regeneration, generating data...')
    gcp = Grid_Cell_Processor()
    gcp.load_data(args.mouse_name, args.day, args.module, args.session, fr_smooth_sigma=args.fr_smooth_sigma, adaptive_fr_sigma=args.adaptive_fr_sigma, speed_estimation_method='kalman_filter', digitize_space=30)

    fire_rate, x, y, t, speed = gcp.preprocess(downsample_rate=args.downsample_rate_gen, pca_components=None, return_speed=True, gridness_thre=0.1, use_zscore=False, speed_thre=0.025, speed_max=0.65)

    np.savez(preprocessed_path, fire_rate=fire_rate, x=x, y=y, t=t, dt=gcp.dt, speed=speed)
else:
    print('Preprocessed data exists, loading...')
    fire_rate, x, y, t, speed = np.load(preprocessed_path)['fire_rate'], np.load(preprocessed_path)['x'], np.load(preprocessed_path)['y'], np.load(preprocessed_path)['t'], np.load(preprocessed_path)['speed']


gkr_file_name = 'gkr_{}_{}_{}_{}_pca{}_downsample{}_{}_{}_{}_{}.hkl'.format(args.mouse_name, args.module, args.day, args.session, str(args.pca_component), str(args.downsample_rate_fit), args.fr_smooth_sigma, args.downsample_rate_gen, args.adaptive_fr_sigma, args.n_inducing)
model_path = os.path.join(DATAROOT, gkr_file_name)

if os.path.exists(model_path) and (not args.regenerate_data):
    print('Model already exists, exit')
    exit()

label = np.array([x, y, speed]).T
feamap = fire_rate

label = label[::args.downsample_rate_fit]
feamap = feamap[::args.downsample_rate_fit]

pca = PCA(n_components=args.pca_component)
feamap = pca.fit_transform(feamap)
print('feamap shape is: ', feamap.shape)

model = GKR_Fitter(n_input=label.shape[1], n_output=feamap.shape[1], n_epochs=args.n_epoch, gpr_params={'n_inducing': args.n_inducing, 'standardize': True})
model.fit(feamap, label)
hkl.dump(model, model_path)
