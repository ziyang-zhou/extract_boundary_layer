import h5py
import numpy as np
import pdb
import math
import pandas as pd

def rotate_points(x, y, x_0, y_0, angle_degrees):
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)
    
    # Translate points so that the rotation center is at the origin
    x_translated = x - x_0
    y_translated = y - y_0
    
    # Perform rotation using rotation matrix
    x_rotated = x_translated * np.cos(angle_radians) + y_translated * np.sin(angle_radians)
    y_rotated = -x_translated * np.sin(angle_radians) + y_translated * np.cos(angle_radians)
    
    # Translate points back to their original position
    x_new = x_rotated + x_0
    y_new = y_rotated + y_0
    
    return x_new, y_new

def find_closest_point(x, y, x_0, y_0):
    # Convert input arrays to numpy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the distance between each point and the target point
    distances = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    return closest_index
    
file = 'tr-meas-surface.hdf5'
fd = h5py.File(file,'r')
keylist = fd.keys()
x = fd['Geometry']['X']
y = fd['Geometry']['Y']

probe_list = ['3','5','9','21','25']
#Declare size of time series
Total_Time_Step = 17000
Chunk_Size = 500

N = int(Total_Time_Step/Chunk_Size)
rotation_center_x = -0.067
rotation_center_y = 0.0209
rotation_degree = 0 #Angle of rotation (clockwise) with respect to 8 degree airfoil.
probe_diameter = 0.005 #Probe diameter of the wall pressure probe

for probe in probe_list:

	if probe == '5':
		probe_x_position = -0.12376
		probe_y_position = 0.0212125
	elif probe == '9':
		probe_x_position = -0.0632463
		probe_y_position = 0.0199974
	elif probe == '21':
		probe_x_position = -0.019227
		probe_y_position = 0.00794334
	elif probe == '3':
		probe_x_position = -0.128516
		probe_y_position = 0.0207075
	elif probe == '25':
		probe_x_position = -0.003036
		probe_y_position = 0.00191532
	elif probe == '4':
		probe_x_position = -0.126493
		probe_y_position = 0.0180604
	elif probe == '8':
		probe_x_position = -0.0814612
		probe_y_position = 0.016288
	elif probe == '10':
		probe_x_position = -0.0637522
		probe_y_position = 0.0138381
	elif probe == '29':
		probe_x_position = -0.009613
		probe_y_position = 0.00185464
	elif probe == '24':
		probe_x_position = -0.010625
		probe_y_position = 0.0047688
	elif probe == '1':
                probe_x_position = -0.13388
                probe_y_position = 0.0194865
	elif probe == '2':
                probe_x_position = -0.131552
                probe_y_position = 0.0203201

	probe_x_position,probe_y_position = rotate_points(probe_x_position,probe_y_position,rotation_center_x,rotation_center_y,rotation_degree)
	print('probe_x_position',probe_x_position,'probe_y_position',probe_y_position,'after rotation')
	te_index = find_closest_point(x,y,probe_x_position,probe_y_position)
	x_te = x[te_index]
	y_te = y[te_index]

	te_mask = (np.sqrt((x-x_te)**2 + (y-y_te)**2) < probe_diameter) #Mask to scope all points within a spherical distance of probe size

	print('x =',x_te,'y =',y_te)

	#Declare size of pressure history
	pressure_history = np.zeros((Total_Time_Step,2))

	for n in range(1,N+1):
		Lower_Index = Chunk_Size*(n-1)
		Upper_Index = Chunk_Size*n
		timeseries = list(keylist)[Lower_Index:Upper_Index]
		pressure = np.zeros((len(timeseries),x.shape[0]))
		time = np.zeros(len(timeseries))

		for t,key in enumerate(timeseries):
			pressure[t,:] = fd[key]['pressure'][()]
			time[t] = fd[key]['time'][()]
		print('shape of time array is',time.shape)

		pressure_slice = np.zeros((time.shape[0],2))
		pressure_slice[:,0] = time[:]
		pressure_slice[:,1] = np.squeeze(pressure[:,te_index])
		pressure_history[((n-1)*Chunk_Size):n*Chunk_Size,0] = time[:]
		pressure_history[((n-1)*Chunk_Size):n*Chunk_Size,1] = np.squeeze(pressure[:,te_index])
		print('chunk',n,'read\n',N-n,'left')	

	# Convert the data to a DataFrame Create new array for pressure matrix at selected point and write it into a file
	df = pd.DataFrame(pressure_history, columns=['time', 'pressure'])	
	# Define the filename
	filename = 'probe_pressure/probe_{}_pressure.csv'.format(probe)
	np.savetxt(filename, pressure_history, fmt='%.18e', delimiter=',', newline='\n', header='time,pressure')
