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

#############################################################################################
#                                        Load Inputs                                        #
#############################################################################################
# Load the settings dataframe:
settings = pd.read_csv("setting.csv", index_col= 0)
le_cut = eval(settings.at["le_cut", settings.columns[0]])
te_cut = eval(settings.at["te_cut", settings.columns[0]])
include_pressure_side = settings.at["include_pressure_side", settings.columns[0]]
refinement_factor = eval(settings.at["refinement_factor", settings.columns[0]])
<<<<<<< HEAD
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness

file = '../mesh/tr-meas-surface_first_70511.hdf5'
=======

file = 'tr-meas-surface_first_70511.hdf5'
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
b = h5py.File(file,'r')
#Investigate content of file
print('zones in base b',b.keys())
print('instants in base b',b['Geometry'].keys())
# In[4]:
xvals=[-0.0307167,-0.0293987,-0.000383135,-0.00220858,-0.0307167]
yvals=[0.0120501,0.0166085,0.00613008,0.00174928,0.0120501]
# In[5]:
x_coord_list = list(b['Geometry']['X'])
x_coord_list = x_coord_list[::10000]
y_coord_list = list(b['Geometry']['Y'])
y_coord_list = y_coord_list[::10000]
# Convert lists to NumPy arrays
x_coord = np.array(x_coord_list)
y_coord = np.array(y_coord_list)
#Cut off the LE and TE radii to simplify the geometry
mask = (x_coord >= le_cut) & (x_coord <= te_cut)
x_coord = x_coord[mask]
y_coord = y_coord[mask]
print('x_coord min',min(x_coord))
print('y_coord max',max(y_coord))
#Define a piecewise mean camber line to differentiate suction points from pressure points
def f(x):
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
    mean_camber = f(x)  # Calculate the mean camber for the current x
    print('mean_camber',mean_camber)
    print('y',y)
    if y > mean_camber:
        # If y is greater than the mean camber, add to the greater subgroup
        x_coord_suction.append(x)
        y_coord_suction.append(y)
    else:
        # If y is smaller than or equal to the mean camber, add to the smaller subgroup
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

plt.figure()
plt.plot(x_coord,y_coord)
plt.plot(xvals,yvals)
<<<<<<< HEAD
plt.savefig('../mesh/domain_plot.png')
=======
plt.savefig('domain_plot.png')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
print('full domain saved')
# In[6]:

xmin = le_cut
xmax = te_cut

keep=(x_coord>xmin)*(x_coord<xmax)
plt.figure()
plt.plot(xvals,yvals,linestyle='dashed')
plt.plot(x_coord[keep],y_coord[keep])
plt.axis('equal')
<<<<<<< HEAD
plt.savefig('../mesh/domain_extent.png')
=======
plt.savefig('domain_extent.png')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
print('extracted domain saved')




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
dr = 85e-6/refinement_factor
zmin = -0.0075776
zmax = 0.0075776
voxel_min = (zmax-zmin)/1024
dr = (zmax-zmin)/170

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

plt.figure(figsize=(15,10))
plt.plot(vec_x_prof,vec_n_prof[:,1])
<<<<<<< HEAD
plt.savefig('../mesh/surface_vector.png')
=======
plt.savefig('surface_vector.png')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a

# In[10]:
plt.figure()
plt.plot(vec_x_prof,vec_y_prof,'.')
plt.quiver(vec_x_prof,vec_y_prof,vec_n_prof[:,0],vec_n_prof[:,1])
plt.plot(xvals,yvals,linestyle='dashed')
plt.plot(x_coord[keep],y_coord[keep])
plt.axis('equal')
<<<<<<< HEAD
plt.savefig('../mesh/surface_vector_2.png')
=======
plt.savefig('surface_vector_2.png')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a

# In[11]:

dn0 = 20e-6
dn_max = 120e-6
dn_q = 1.03
<<<<<<< HEAD
target_height = delta_95*2 #Target height for data extraction
Nn = 73

while target_height > N_0:
    Nn += 1
    dn = np.zeros(Nn,)
    for idx in range(Nn):
        dn[idx] = min(dn0*dn_q**idx,dn_max)

    vec_n = np.zeros(Nn+1,)
    vec_n[1:] = np.cumsum(dn)
    N_0 = np.max(vec_n[1:])
    
plt.figure()
plt.plot(vec_n,'o')
plt.savefig('../mesh/normal_vector.png')
=======
N = 50
Nn = 73
dn = np.zeros(Nn,)
for idx in range(Nn):
    dn[idx] = min(dn0*dn_q**idx,dn_max)

vec_n = np.zeros(Nn+1,)
vec_n[1:] = np.cumsum(dn)
    
plt.figure()
plt.plot(vec_n,'o')
plt.savefig('normal_vector.png')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a


# In[12]:


Xmat = np.zeros((npts_prof,Nn+1))
Ymat = np.zeros((npts_prof,Nn+1))
for idx,nv in enumerate(vec_n):
    Xmat[:,idx] = vec_x_prof + nv*vec_n_prof[:,0]
    Ymat[:,idx] = vec_y_prof + nv*vec_n_prof[:,1]
    
plt.figure()
plt.contourf(Xmat,Ymat,np.ones_like(Xmat),linestyles='solid')
plt.axis('equal')
<<<<<<< HEAD
plt.savefig('../mesh/surface_grid')
=======
plt.savefig('surface_grid')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a


# In[13]:


bi = Base()
bi.init()
bi[0][0]['x']=Xmat
bi[0][0]['y']=Ymat
w=Writer('bin_tp')
w['base']=bi
<<<<<<< HEAD
w['filename']='../mesh/interpolation_grid.plt'
=======
w['filename']='interpolation_grid.plt'
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
w.dump()


# In[14]:


vec_z = np.arange(zmin+dr/2,zmax+dr/2,dr)
Nz = vec_z.size
print((zmax - zmin )/dr)
print(vec_z[-1])
print(Nz)

plt.plot([0,0],[zmin,zmin],'o')
plt.plot([Nz,Nz],[zmax,zmax],'s')
plt.plot(vec_z)
plt.show()


# In[15]:


Xvol = np.repeat(Xmat[:, :, np.newaxis], Nz, axis=2)
Yvol = np.repeat(Ymat[:, :, np.newaxis], Nz, axis=2)
Zyz = np.repeat(vec_z[np.newaxis,:], Nn+1, axis=0)
Zvol = np.repeat(Zyz[np.newaxis,:,:], npts_prof, axis=0)
Nxy = np.repeat(vec_n_prof[:,np.newaxis,:],Nn+1,axis=1)
Nvol = np.repeat(Nxy[:,:,np.newaxis,:],Nz,axis=2)


# In[16]:


bv = Base()
bv.init()
bv[0][0]['x']=Xvol
bv[0][0]['y']=Yvol
bv[0][0]['z']=Zvol
bv[0][0]['norm_x']=Nvol[:,:,:,0]
bv[0][0]['norm_y']=Nvol[:,:,:,1]
bv[0][0]['norm_z']=Nvol[:,:,:,2]
# For visualization / quality check 
w=Writer('bin_tp')
<<<<<<< HEAD
w['filename']='../mesh/interpolation_3d_grid.plt'
=======
w['filename']='interpolation_3d_grid.plt'
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
w['base']=bv
w.dump()
# For later usage
w=Writer('hdf_antares')
<<<<<<< HEAD
w['filename']='../mesh/interpolation_3d_grid'
=======
w['filename']='interpolation_3d_grid'
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
w['base']=bv
w.dump()


# In[17]:


bv.compute_cell_volume()


# In[18]:


bv.cell_to_node()
print(np.amin(bv[0][0][('cell_volume','node')]))
print(np.amax(bv[0][0][('cell_volume','node')]))


# In[19]:


<<<<<<< HEAD
fid = h5py.File('../mesh/interpolation_3d_grid.hdf5','w')
=======
fid = h5py.File('interpolation_3d_grid.hdf5','w')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
fid['/x']=Xvol.flatten()
fid['/y']=Yvol.flatten()
fid['/z']=Zvol.flatten()
fid['/volume']=bv[0][0][('cell_volume','node')].flatten()
fid.close()


# In[20]:


<<<<<<< HEAD
fid = h5py.File('../mesh/interpolation_3d_grid_dims.hdf5','w')
=======
fid = h5py.File('interpolation_3d_grid_dims.hdf5','w')
>>>>>>> 448bf0e5de47684dea93585e623a1d75623e339a
fid['/x']=Xvol
fid['/y']=Yvol
fid['/z']=Zvol
fid['/volume']=bv[0][0][('cell_volume','node')]
fid.close()


# In[21]:


print('Search distance is: {0:e} m'.format(np.amin(bv[0][0][('cell_volume','node')])**(1./3.)))


