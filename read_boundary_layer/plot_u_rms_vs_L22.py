# This script reads u_rms and v_rms and L22 and L21 and plots them on a graph at a given probe location in the wall normal direction.
from antares import *
from functions import analysis
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math
import h5py
import pandas as pd
import pdb

#Read the filepath
project_path = temporal.project_path
bl_path = temporal.bl_path
mesh_path = temporal.mesh_path
settings = pd.read_csv("../setting.csv", index_col= 0)

# Get the probe location and its equivalent index in the xcoor array.
xcoor0 = temporal.xcoor0 # x location of the selected axis of the plot
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness

#Read the geometry of the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()
xcoor = BL_line_geom[0][0]['x'][:,0,0] #BL geom array format : (streamwise,wall normal,spanwise)
x_index = analysis.find_nearest(xcoor,xcoor0) #streamwise coordinate of fixed point
print('shape of BL line geom',BL_line_geom[0][0]['x'].shape)

#Read the lengthscale
L22 = pd.read_csv(project_path + 'L22+_column')
L21 = pd.read_csv(project_path + 'L22+_row')

#Read the mesh
r=Reader('hdf_antares')
r['filename'] = bl_path + 'U_n_rms.h5'
base_un=r.read()
U_n_rms = base_un[0][0]['u_rms']

r=Reader('hdf_antares')
r['filename'] = bl_path + 'U_t_rms.h5'
base_ut=r.read()
U_t_rms = base_ut[0][0]['u_rms']

keep_h = ((base_ut[0][0]['h_coord'][:,0] <= L22['wall distance'].iloc[-1]) & (base_ut[0][0]['h_coord'][:,0] >= L22['wall distance'].iloc[0]))
hcoor = np.array(L22['wall distance'])

plt.plot(hcoor/delta_95,np.array(L22['L22+']),label='L22')
plt.plot(hcoor/delta_95,np.array(L21['L22+']),label='L21')
plt.legend()
plt.savefig(temporal.project_path + 'L22_L21')
plt.close()

plt.plot(hcoor/delta_95,U_n_rms[keep_h,x_index],label='un_rms')
plt.plot(hcoor/delta_95,U_t_rms[keep_h,x_index],label='ut_rms')
plt.legend()
plt.savefig(temporal.project_path + 'un_rms_ut_rms')
plt.close()

plt.plot(hcoor/delta_95,np.array(L22['L22+'])/np.array(L21['L22+']),label='L22/L21')
plt.plot(hcoor/delta_95,U_n_rms[keep_h,x_index]/U_t_rms[keep_h,x_index],label='un/ut rms')
plt.legend()
plt.savefig(temporal.project_path + 'L ratio to u rms ratio')
plt.close()