step_per_chunk = 10 #size of each data chunk in terms of number of timesteps
total_timesteps = 1370 #last time step read
starting_timestep = 500 #Index of timestep to start from
timestep_size = 1/17863 #size of a timestep
initial_time = 0.0 #first time step condidered for the reading of the boundary layer
adjusted_fs_factor = 1 #factor to reduce sampling frequency by, can be a multiple of 5 (5,10,15...)

vtu_path = '../frame_data/' #path to read the vtu data from
vol_path = '../../b_vol/' #path to write the CFD flow h5 data to
bl_path = '../../BL_line_prof/' #path to read and write the boundary layer profiles
mesh_path = '../../mesh/' #path to the grid generated by pre-processing script
project_path ='../../../postprocessing/'
save_path = '/home/ziyz1701/storage/CD_airfoil/FAN_2025/Result/'

#Boundary layer extraction parameter
length_extraction = 0.01 #slightly higher than as the delta_95
var_detection = 'U_t'
nb_points = 1500 #no. of wall normal discr points
axis = 'x'
axis_direction = 0
relative_velocity_vec = ['x_velocity', 'y_velocity', 'z_velocity']
density = '1.251'
kinematic_viscosity = '0.0000144' # PowerFLOW characteristic kinematic viscosity
laminar_dynamic_viscosity = '0.000018' # PowerFLOW characteristic kinematic viscosity x density
U_0 = '16.38'
var = 'U_n' #variable used for the cross correlation contour

#Integration
if_interpolate = True # Set as true if interpolation is needed to remove zeros in the contour
if_integrate_axis = True # Set as true if the integral is to be calculated
if_integrate_field = False # Set as true if the integral length scale field is to be computed
if_read_boundary_velocity = False # Set as true for computation of mean velocity
troubleshoot = False # Set as true if velocity contours are to be plotted

#Turbulence statistic computation
wall_shear_method = 'shear_fit' #method of wall shear computation : smoothed_derivative or spline or shear_fit or legacy_spline
update_bl_var = [] # list of strings for declaring variables to be updated
update_surface_var = []
xcoor0 = -0.019227 # x location of the integration axis
probe_number = 21 # probe location at which to plot the interal length
h_0_bar = 0.2 # Dimensionless wall normal position of fixed point. Normalized by delta_95
timestep_interval = 100 # interval at which to plot contours for troubleshooting
