step_per_chunk = 50 #size of each data chunk in terms of number of timesteps
total_timesteps = 1000 #last time step read
starting_timestep = 0 #Index of timestep to start from
timestep_size = 1/17863 #size of a timestep
initial_time = 0.0 #first time step condidered for the reading of the boundary layer
adjusted_fs_factor = 1 #factor to reduce sampling frequency by, can be a multiple of 5 (5,10,15...)

vtu_path = '../frame_data/' #path to read the vtu data from
vol_path = 'b_vol/' #path to write the CFD flow h5 data to
bl_path = '/scratch/m/moreaust/zzhou/Reference_DNS/postprocessing/extract_boundary_profile/read_boundary_layer/BL_line_prof/' #path to read and write the boundary layer profiles
mesh_path = '/scratch/m/moreaust/zzhou/Reference_DNS/postprocessing/extract_boundary_profile/' #path to the grid generated by pre-processing script
project_path ='/scratch/m/moreaust/zzhou/Reference_DNS/postprocessing/'

#Boundary layer extraction parameter
length_extraction = 0.01 #slightly higher than as the delta_95
var_detection = 'U_t'
nb_points = 150 #no. of wall normal discr points
axis = 'x'
axis_direction = 0
relative_velocity_vec = ['x_velocity', 'y_velocity', 'z_velocity']
density = '1.225'
laminar_dynamic_viscosity = '0.0000144'
density = '1.251'
U_0 = '16.38'
var = 'U_n' #variable used for the cross correlation contour

#Integration
if_interpolate = True # Set as true if interpolation is needed to remove zeros in the contour
if_integrate = True # Set as true if the integral is to be calculated
troubleshoot = True # Set as true if velocity contours are to be plotted
integration_axis = 'row'
xcoor0 = -0.019227 # x location of the integration axis