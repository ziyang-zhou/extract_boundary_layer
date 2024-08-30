step_per_chunk = 50 #size of each data chunk in terms of number of timesteps
total_timesteps = 500 #last time step read
starting_timestep = 0 #Index of timestep to start from
timestep_size = 1/17863 #size of a timestep
initial_time = 0.0 #first time step condidered for the reading of the boundary layer
adjusted_fs_factor = 1 #factor to reduce sampling frequency by, can be a multiple of 5 (5,10,15...)

vtu_path = '../frame_data/' #path to read the vtu data from
vol_path = 'b_vol/' #path to write the CFD flow h5 data to
mesh_path = '/scratch/m/moreaust/zzhou/Reference_DNS/extract_boundary_profile/' #path to the grid generated by pre-processing$
project_path = '/scratch/m/moreaust/zzhou/Reference_DNS/'

#mesh path set to SLR case temporarily because the mesh in this case folder is being converted for LSB zoom

#Extraction parameter
length_extraction = 0.00678
var_detection = 'U_t'
nb_points = 500 #no. of wall normal discr points
axis = 'x'
axis_direction = 0
relative_velocity_vec = ['x_velocity', 'y_velocity', 'z_velocity']
density = '1.225'
laminar_dynamic_viscosity = '0.0000144'
density = '1.251'
U_0 = '16.38'

nb_point_chord = 107
