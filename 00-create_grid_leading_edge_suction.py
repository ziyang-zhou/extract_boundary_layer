#!/usr/bin/env python
# coding: utf-8

# In[1]:

from antares import *

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sintp
import scipy.ndimage as sim
import h5py
import math
import os

#############################################################################################
#                                        Load Inputs                                        #
#############################################################################################
# Load the settings dataframe:
settings = pd.read_csv("setting.csv", index_col= 0)
le_cut = eval(settings.at["le_cut", settings.columns[0]])
te_cut = eval(settings.at["te_cut", settings.columns[0]])
include_pressure_side = settings.at["include_pressure_side", settings.columns[0]]
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness

file = '../mesh/tr-meas-surface_first_70511.hdf5'
b = h5py.File(file,'r')

dn0 = eval(settings.at["dn0", settings.columns[0]]) # height of first grid cell from wall
dn_max = eval(settings.at["dn_max", settings.columns[0]]) # Max allowable size of boundary layer grid
dn_q = eval(settings.at["dn_q", settings.columns[0]]) # growth rate of boundary layer grid
target_height = eval(settings.at["target_height", settings.columns[0]]) # Target wall height of boundary layer grid
Nn = eval(settings.at["Nn", settings.columns[0]]) # Min number of cells in boundary layer grid

#Investigate content of file
print('zones in base b',b.keys())
print('instants in base b',b['Geometry'].keys())

#Define the aoa
angle_of_attack = '8deg' # angle of rotation of airfoil with respect to 8 deg config
# In[5]:
nskip = 10000
finest_voxel_size=0.0000148
mid_plane_halfwidth = 3*finest_voxel_size
z_coord_array = np.array(list(b['Geometry']['Z']))
z_mask = (z_coord_array < mid_plane_halfwidth) & (z_coord_array > -mid_plane_halfwidth)

x_coord_array = np.array(list(b['Geometry']['X']))
x_coord_array = x_coord_array[z_mask]
y_coord_array = np.array(list(b['Geometry']['Y']))
y_coord_array = y_coord_array[z_mask]
# Convert lists to NumPy arrays
x_coord = np.array(x_coord_array)
y_coord = np.array(y_coord_array)
#Cut off the LE and TE radii to simplify the geometry
mask = (x_coord >= le_cut) & (x_coord <= te_cut)
x_coord = x_coord[mask]
y_coord = y_coord[mask]
print('x_coord min',min(x_coord))
print('y_coord max',max(y_coord))

#Define a piecewise mean camber line to differentiate suction points from pressure points
#8deg
def f_1(x):
    # Define the points for the first line segment
    x1 = -0.14
    y1 = 0.019
    x2 = -0.08
    y2 = 0.019
    # Define the points for the second line segment
    x3 = -0.08
    y3 = 0.019
    x4 = 0.0
    y4 = 0.0  
    # Check which line segment to use based on the value of x
    if x <= x2:
        # Equation of the first line segment (y = mx + b)
        return y2
    else:
        m=(y4-y3)/(x4-x3)
        # Handle values of x outside the defined segments
        return  m*x# You can choose to return a default value or raise an error

#15deg
def f_2(x):
    # Define the points for the first line segment
    x1 = -0.14
    y1 = 0.0276
    x2 = -0.08
    y2 = 0.0205
    # Define the points for the second line segment
    x3 = -0.08
    y3 = 0.0205
    x4 = 0.0
    y4 = -0.009  
    # Check which line segment to use based on the value of x
    if x <= x2:
        # Equation of the first line segment (y = mx + b)
        m=(y2-y1)/(x2-x1)
        return m*(x-x1) + y1
    else:
        m=(y4-y3)/(x4-x3)
        # Handle values of x outside the defined segments
        return  m*(x-x3) + y3# You can choose to return a default value or raise an error

#Sort all points into two lists based on whether they are above or below the mean camber line
# Create empty lists for the two subgroups
x_coord_suction = []
y_coord_suction = []
x_coord_pressure = []
y_coord_pressure = []
# Iterate through the coordinates and sort them into subgroups
for i in range(len(x_coord)):
    x = x_coord[i]
    y = y_coord[i]
    if angle_of_attack == '8deg':
        mean_camber = f_1(x)  # Calculate the mean camber for the current x
    elif angle_of_attack == '15deg':
        mean_camber = f_2(x)
    print('mean_camber',mean_camber)
    print('y',y)
    if y > mean_camber:
        # If y is greater than the mean camber, add to the suction subgroup
        x_coord_suction.append(x)
        y_coord_suction.append(y)
    else:
        # If y is smaller than or equal to the mean camber, add to the pressure subgroup
        x_coord_pressure.append(x)
        y_coord_pressure.append(y)

# Convert the lists to NumPy arrays if needed
x_coord_pressure = np.array(x_coord_pressure)
y_coord_pressure = np.array(y_coord_pressure)
x_coord_suction = np.array(x_coord_suction)
y_coord_suction = np.array(y_coord_suction)

#Sort all points according to order of ascending x-coordinate
sort_indices = np.argsort(x_coord_pressure)
sort_indices_descending = sort_indices[::-1]
x_coord_pressure_sorted_descending = x_coord_pressure[sort_indices_descending]
y_coord_pressure_sorted_descending = y_coord_pressure[sort_indices_descending]
# Get the indices that would sort x_coord_suction in ascending order
sort_indices_suction = np.argsort(x_coord_suction)
x_coord_suction_ascending = x_coord_suction[sort_indices_suction]
y_coord_suction_ascending = y_coord_suction[sort_indices_suction]

print('x_coord_pressure_sorted_descending min',min(x_coord_pressure_sorted_descending))
print('y_coord_pressure_sorted_descending max',max(y_coord_pressure_sorted_descending))

if include_pressure_side is True:
	x_coord = np.concatenate((x_coord_pressure_sorted_descending, x_coord_suction_ascending),axis=0)
	y_coord = np.concatenate((y_coord_pressure_sorted_descending, y_coord_suction_ascending),axis=0)
else:
	x_coord = x_coord_suction_ascending
	y_coord = y_coord_suction_ascending

# In[6]:
xmin = le_cut
xmax = te_cut

keep=(x_coord>xmin)*(x_coord<xmax)

# In[7]:

#creation of interpolation function which takes streamwise coordinate as input and outputs cartesian coordinates
xprof = x_coord[keep]
yprof = y_coord[keep]
ds = np.sqrt((xprof[1:]-xprof[:-1])**2 + (yprof[1:]-yprof[:-1])**2)
sprof = np.zeros(ds.size+1,)
sprof[1:] = np.cumsum(ds)
ls = sprof[-1]

fx = sintp.interp1d(sprof,xprof)
fy = sintp.interp1d(sprof,yprof)

# In[8]:
#declaration of dr - step size in new curvilinear array of streamwise coordinate.
npts_chord = eval(settings.at["npts_chord", settings.columns[0]]) # height of first grid cell from wall
dr = abs(le_cut - te_cut)/npts_chord

# In[9]:
#resample the curvilinear coordinates to make them equidistant
#create a corresponding set of x and y coordinates
vec_s = np.arange(0,ls,dr)
npts_prof = vec_s.size

vec_x_prof = fx(vec_s)
vec_y_prof = fy(vec_s)

vec_t_prof = np.zeros((npts_prof,2))
#create a new array of vectors vec_t_prof which contains unit vectors of the displacement between neighbouring points
for iz in range(1,npts_prof-1):
    tx_dn = vec_x_prof[iz+1]-vec_x_prof[iz]
    ty_dn = vec_y_prof[iz+1]-vec_y_prof[iz]
    tnorm = np.sqrt(tx_dn**2+ty_dn**2)
    tx_dn = tx_dn/tnorm
    ty_dn = ty_dn/tnorm

    tx_up = vec_x_prof[iz]-vec_x_prof[iz-1]
    ty_up = vec_y_prof[iz]-vec_y_prof[iz-1]
    tnorm = np.sqrt(tx_up**2+ty_up**2)
    tx_up = tx_up/tnorm
    ty_up = ty_up/tnorm
    
    vec_t_prof[iz,0] = 0.5 * (tx_up + tx_dn)
    vec_t_prof[iz,1] = 0.5 * (ty_up + ty_dn)

tx_dn = vec_x_prof[1]-vec_x_prof[0]
ty_dn = vec_y_prof[1]-vec_y_prof[0]
tnorm = np.sqrt(tx_dn**2+ty_dn**2)
vec_t_prof[0,0] = tx_dn/tnorm
vec_t_prof[0,1] = ty_dn/tnorm

tx_up = vec_x_prof[-1]-vec_x_prof[-2]
ty_up = vec_y_prof[-1]-vec_y_prof[-2]
tnorm = np.sqrt(tx_up**2+ty_up**2)
vec_t_prof[-1,0] = tx_up/tnorm
vec_t_prof[-1,1] = ty_up/tnorm

vec_n_prof = np.zeros((npts_prof,3))
vec_n_prof[:,0] = -sim.gaussian_filter1d(vec_t_prof[:,1],sigma=10, order=0, mode='nearest')
vec_n_prof[:,1] = sim.gaussian_filter1d(vec_t_prof[:,0],sigma=10, order=0, mode='nearest')

#Create the wall normal vector 
N_0 = 0
while target_height > N_0:
    Nn += 1
    dn = np.zeros(Nn,)
    for idx in range(Nn):
        dn[idx] = min(dn0*dn_q**idx,dn_max)

    vec_n = np.zeros(Nn+1,)
    vec_n[1:] = np.cumsum(dn)
    N_0 = np.max(vec_n[1:])

Xmat = np.zeros((npts_prof,Nn+1))
Ymat = np.zeros((npts_prof,Nn+1))
for idx,nv in enumerate(vec_n):
    Xmat[:,idx] = vec_x_prof + nv*vec_n_prof[:,0]
    Ymat[:,idx] = vec_y_prof + nv*vec_n_prof[:,1]

#Create a hdf5 mesh for sanjose interpolation
fout = h5py.File('interp_grid.hdf5','w')
fout.create_dataset('x', data= Xmat)
fout.create_dataset('y', data= Ymat)
fout.create_dataset('z', data= np.zeros_like(Xmat))
fout.close()

# Visualize the 2D mesh Draw horizontal lines
for i in range(Xmat.shape[0]):
    plt.plot(Xmat[i, :], Ymat[i, :], color='black', linewidth=0.5)
# Draw vertical lines
for j in range(Xmat.shape[1]):
    plt.plot(Xmat[:, j], Ymat[:, j], color='black', linewidth=0.5)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('../mesh/2D_mesh.png')