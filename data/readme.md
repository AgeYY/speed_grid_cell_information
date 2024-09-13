file name: rat_{rat name}_day{i}_grid_modules_{module id 1}_{module id 2}. Each rat was recorded in multiple sessions. Session name is included in the similar name but txt file. Number attached to the session name indicate the recorded module of grid cells.
rat_q_grid_modules_1_2.npz:
    - spikes_mod1: dict {0: 1d array, 1: 1d array, ..., 96: 1d array}. Each 1d array has different length (typically from thousands to 10 thousands), I guess its spiking time. A few example values: [9594.8525, 9620.3956, ...] (unit is second). Ranges of every array are similar, about [9576 to 31220]. If this is spiking time, then the time should live within the time window indicated by rat_q_sessions.txt. The answer is yes, minimum time of trial is 9576 and maximum is 31223. Therefore this includes probably the spiking time of 96 grid cells, throughout different sessions.
    - x: 1d array of shape [2164699], value ranges from -1 to 1. You can get the unit of range by comparing the range of x to the range of experimental box. For example, I found across different mice, the x range in open field is always about -0.75 to 0.75. This range may match to the experimental box 1.5m. This implies the unit of x is meter. That also explains why later computing dx, the unit is multiplied by 100
    - y: shape and ranges same as above
    - t: same shape as above, ranges from [9576, 31223]. The unit is second. This is reasoned from the code. The spacing of two adjecent time is 0.01. Check the code (searching '2.5' in the git repo), we found speed is computed by:

    dx = (xxs[1:] - xxs[:-1])*100
    dy = (yys[1:] - yys[:-1])*100
    speed = np.sqrt(dx**2+ dy**2)/0.01
    xx_box = xx_box[speed>=2.5]
    
    Since the unit of 2.5 is 2.5 cm/s, so two adjcent time (the dt corresponding to dx) is 0.01 s. This exactly the same as the spacing of two adjecent time in the array t. Further, the paper mensioned (method: visualization of toroidal manifold) that they binned time on 10 ms, hence the data provided here is already binned to 10 ms.

    - z: shape same as above, ranges from 0.64 to 1.98. why z is not a constant? I check the range of z in each module, after cutoff tails, z ranges of sleep_box_1 is [0.8, 0.93], wagon_wheel [0.65, 0.87], sleep_box_2 [0.8, 1.0], open_field [0.74, 0.97]. It's OK, mostly steady except wagon wheel.
    - azimuth: same shape as above, ranges from 0 to 2 pi

# generated dataset
these datasets are generated from the experimental data.

preprocessed_data_mouse_name_module_day_session_pcaNone.npz: preprocessed data, without pca
  - fire_rate: shape [n_time, n_neuron]
  - x, y, t: shape [n_time]
preprocessed_data_mouse_name_module_day_session_pca6.npz: preprocessed data, with pca equals to 6
  - fire_rate: shape [n_time, 6]
  - x, y, t: shape [n_time]
  
gp_mouse_name_module_day_session_pca6.npz: fitted gp manifold
  - feamap: shape of [n_mesh_points, 6]
  - query_mesh: shape [n_mesh_points, 2]. Corresponding to x and y position
