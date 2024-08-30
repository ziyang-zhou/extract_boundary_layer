from antares import *
import numpy as np
from antares.core.PrintUtility import progress_bar
import temporal
import math
import scipy.spatial
import scipy.integrate
import scipy.interpolate as intp

from scipy.interpolate import interp1d
from scipy.integrate import simps


def sliding_avg(data,*args, **kwargs):
    npts= kwargs.get('npts', 5)
    w = [1.0/npts]*npts
    data_filt=np.convolve(data,w,mode='valid')
    return data_filt

def find_zero_crossing(data):
    zero_crossing=np.where(np.diff(np.signbit(data)))[0]
    return zero_crossing

def find_nearest(array,value):
  idx = (np.abs(array-value)).argmin()
  return idx

def get_delta99_from_line(wall_distance,BL_detection_variable,filter_size_var=10,filter_size_der=3,npts_interp=100,delta_pct_end=0.85, nb_min_search=5):
  """
  Routine to detect the boundary layer end location based on a variable
  (typically total pressure in relative frame or rothalpy) that is constant outside the boundary layer. Call as:
  get_delta99_from_profile(wall_distance,BL_detection_variable,filter_size_var=10,filter_size_der=3,npts_interp=100,delta_pct_end=0.85)

  with :
  wall_distance The distance from the wall along the ordered (from wall to exterior) line provided
  BL_detection_variable The detection variable alond the ordered (from wall to exterior) line provided
  filter_size_var the gaussian filter size for the BL_detection_variable
  filter_size_der the gaussian filter size for the derivatives of the BL_detection_variable
  npts_interp number of points to interpolate the filtered detection variable with uniform spacing
  delta_pct_end Once the end of the boundary is found apply a safety coefficient

  """
  from scipy import ndimage
  from scipy.interpolate import interp1d
  from heapq import nsmallest

  dh=wall_distance[1:]-wall_distance[:-1]
  if np.abs(np.min(dh)-np.max(dh))>1.0e-6*np.mean(dh):
    flag_interp=True
  else:
    flag_interp=False

  # filtering
  sigma_size=10
  Rt_filt=ndimage.gaussian_filter1d(BL_detection_variable,sigma=filter_size_var, order=0, mode='nearest')
  h_filt=wall_distance

  #interpolation on a regular grid
  if flag_interp:
    h_intp=np.linspace(min(h_filt),max(h_filt),npts_interp)
    f = interp1d(h_filt, Rt_filt,kind='slinear') #spline order 1
    Rt_intp=f(h_intp)
  else:
    h_intp=h_filt
    Rt_intp=Rt_filt

  # Smooth derivative
  dRt=ndimage.gaussian_filter1d(Rt_intp,sigma=filter_size_der, order=1, mode='nearest')
  dh=h_intp[1]-h_intp[0]
  dRtdh=dRt/dh
  hd1=h_intp

  # Smooth second derivative
  dRt2dh2=ndimage.gaussian_filter1d(dRtdh,sigma=filter_size_der, order=1, mode='nearest')/dh
  hd2=hd1
  
  # Try to find zero crossing in first derivative profile
  zero_crossing=find_zero_crossing(dRtdh)

  if len(zero_crossing)>0:
    # Found location of the smallest 2nd derivative > this is the end of the BL
    idx = find_nearest(dRt2dh2[zero_crossing],0.)
    found_value = hd1[zero_crossing][idx]

  # Find minima of first derivative
  else:
    smallest_dRtdh=nsmallest(nb_min_search, dRtdh[:-filter_size_der*2])
    zero_closest=[]
    for val in smallest_dRtdh:
      idx=find_nearest(dRtdh,val)
      zero_closest.append(idx)
    # Found location of the smallest 2nd derivative > this is the end of the BL
    idx= find_nearest(dRt2dh2[zero_closest],0.)
    found_value = hd1[zero_closest][idx]

  idx_delta99=find_nearest(wall_distance,delta_pct_end*found_value) # ok it is not exactly 0.99 but we take some safety here :)
  delta_99=wall_distance[idx_delta99]
  print('Boundary layer thickness: {0:f} m'.format(delta_99))

  return idx_delta99,delta_99

def get_delta95(wall_distance,BL_detection_variable):
	max_value = np.max(BL_detection_variable)
	threshold_value = 0.99*max_value
	for i, value in enumerate(BL_detection_variable):
		if value >= threshold_value:
			delta_95 = wall_distance[i]
			indx_delta_95 = i
			return indx_delta_95,delta_95
	return None

def get_boundary_layer_thicknesses_from_line(wall_distance,relative_velocity_magnitude,density,idx_delta99):
  """
  Routine to compute the boundary layer displacement and momentum thicknesses from data provided along a sorted line from wall to exterior. Call as:
  get_boundary_layer_thicknesses_from_line(wall_distance,relative_velocity_magnitude,density,idx_delta99)

  with :
  wall_distance The distance from the wall along the ordered (from wall to exterior) line provided
  relative_velocity_magnitude the relative velocity alond the ordered (from wall to exterior) line provided
  density the density along the ordered (from wall to exterior) line provided or the single value if density is constant (incompressible)
  idx_delta99 the index of end of boundary layer obtained with the function get_delta99_from_line


  """

  from numpy import trapz

  if isinstance(density,float):
    ro=density*np.ones(relative_velocity_magnitude.shape)
  else:
    ro=density

  roe=ro[idx_delta99]
  Ue=relative_velocity_magnitude[idx_delta99]

  roU=ro*relative_velocity_magnitude

  f1=1.0-roU[:idx_delta99]/(roe*Ue)
  f1_positive=f1>0  # we want a positive results, lets force it

  f2=(1.0-relative_velocity_magnitude[:idx_delta99]/Ue)*(roU[:idx_delta99]/(roe*Ue))
  f2_positive=f2>0  # we want a positive results, lets force it

  delta_star=trapz(f1[f1_positive],wall_distance[:idx_delta99][f1_positive])
  delta_theta=trapz(f2[f2_positive],wall_distance[:idx_delta99][f2_positive])

  print('Displacement boundary layer thickness: {0:f}'.format(delta_star))
  print('Momentum boundary layer thickness: {0:f}'.format(delta_theta))
  return delta_star,delta_theta

def get_wall_shear_stress_from_line(wall_distance,relative_velocity_magnitude,density,kinematic_viscosity,filter_size_var=3,filter_size_der=3,npts_interp=100,maximum_stress=False):
  """
  Routine to compute the wall shear stress from data provided along a sorted line from wall to exterior. Call as:
  get_wall_shear_stress_from_line(wall_distance,relative_velocity_magnitude,density,kinematic_viscosity,filter_size_var=3,filter_size_der=3,npts_interp=100)

  with :
  wall_distance The distance from the wall along the ordered (from wall to exterior) line provided
  relative_velocity_magnitude the relative velocity alond the ordered (from wall to exterior) line provided
  density the density along the ordered (from wall to exterior) line provided or the single value if density is constant (incompressible)
  kinematic_viscosity the kinematic viscosity along the ordered (from wall to exterior) line provided or the single value if temperature is constant
  filter_size_var the gaussian filter size for the relative_velocity_magnitude
  filter_size_der the gaussian filter size for the derivatives of the relative_velocity_magnitude
  npts_interp number of points to interpolate the filtered relative_velocity_magnitude with uniform spacing

  """

  from scipy import ndimage
  from scipy.interpolate import interp1d

  if isinstance(density,np.ndarray):
    ro=density[1]
  else:
    ro=density
  if isinstance(kinematic_viscosity,np.ndarray):
    nu=kinematic_viscosity[0]
  else:
    nu=kinematic_viscosity

  dh=wall_distance[1:]-wall_distance[:-1]
  if np.abs(np.min(dh)-np.max(dh))>1.0e-6*np.mean(dh):
    flag_interp=True
  else:
    flag_interp=False


  #interpolation on a regular grid
  if flag_interp:
    h_intp=np.linspace(min(wall_distance),max(wall_distance),npts_interp)
    f = interp1d(wall_distance, relative_velocity_magnitude,kind='slinear') #spline order 1
    U_intp=f(h_intp)
  else:
    h_intp=wall_distance
    U_intp=relative_velocity_magnitude

  # Smooth derivative
  dU=ndimage.gaussian_filter1d(U_intp,sigma=filter_size_der, order=1, mode='nearest')
  dh=h_intp[1]-h_intp[0]

  if maximum_stress:
    viscous_stress=ro*nu*dU/dh
    max_shear_stress=np.amax(viscous_stress)
    wall_shear_stress=viscous_stress[0]
    return wall_shear_stress,max_shear_stress
  else:
    wall_shear_stress=ro*nu*dU[0]/dh
    if wall_shear_stress == 0 or math.isnan(wall_shear_stress):
      print('wall shear is zero')
      print('dU',dU[0])
      print('dh',dh)
      print('ro',ro)
      print('nu',nu)
      print('velocity profile',relative_velocity_magnitude)
      velocity_gradient = dU[0]/dh
      print('velocity gradient',velocity_gradient)
    return wall_shear_stress




def extract_BL_profiles(b_vol,BL_line_geom,length_extraction,var_detection,nb_points,axis, axis_direction,relative_velocity_vec, density=None,laminar_dynamic_viscosity=None,non_uniform=False,factor_spacing=None):

#relative_velocity_vec contains the names of the velocity components in the line base
  nb_cuts = BL_line_geom[0][0]['x'][:,0,0].size #collect the number of points which the streamwise axis has been discretized into
  print('number of points across chord',nb_cuts)
  nb_inst = len(b_vol[0].keys()) #collect the number of timesteps in bvol
  print('number of timesteps',nb_inst)
  if axis=='x':
    iaxis=0
  elif axis=='z':
    iaxis=2

  BL_line_prof=Base() #reads all the cfd domain data into a new base which contains flow data on the BL lines
  BL_line_prof.init(zones=BL_line_geom.keys(),instants=b_vol[0].keys())#does it for all the timesteps

  successful_extraction={}
  for zn in BL_line_geom.keys():
    successful_extraction[zn]=[]

  for zn in BL_line_geom.keys():

    for ihH in progress_bar(range(0,nb_cuts-1), label='extraction BL h/H on {0:s} '.format(zn)):
      print('ihH',ihH)
      z_middle = int(len(BL_line_geom[zn][0]['x'][ihH,0,:])/2) #obtain the middle index of z
      pt1=np.array([BL_line_geom[zn][0]['x'][ihH,0,z_middle],BL_line_geom[zn][0]['y'][ihH,0,z_middle],0.0])
      #print('pt1',pt1)
      n_vec=np.array([BL_line_geom[zn][0]['norm_x'][ihH,0,z_middle],BL_line_geom[zn][0]['norm_y'][ihH,0,z_middle],BL_line_geom[zn][0]['norm_z'][ihH,0,z_middle]])
      #mag_n_vec=(n_vec[0]**2+n_vec[1]**2+n_vec[2]**2)**0.5
      pt2=pt1+length_extraction*n_vec

      if non_uniform and factor_spacing is not None: #use of antares to interpolate data over the BL line

        t=Treatment('tanhline')
        t['base']=b_vol
        t['point1']=pt1
        t['point2']=pt2
        t['nbpoints']=nb_points
        t['factor']=factor_spacing
        t['mask']=False
        t['probe_tolerance']=14
        line=t.execute()

      else:

        t=Treatment('line')
        t['base']=b_vol
        t['point1']=pt1
        t['point2']=pt2
        t['nbpoints']=nb_points-1
        t['mask']=False
        t['probe_tolerance']=14
        line=t.execute()
        
      if line is not None:
        line.compute('h=((x-{0:f})**2+(y-{1:f})**2+(z-{2:f})**2)**0.5'.format(pt1[0],pt1[1],pt1[2]))
        # Compute tangential velocity
        line.compute('U_n={0:s}*{3:f}+{1:s}*{4:f}+{2:s}*{5:f}'.format(relative_velocity_vec[0],relative_velocity_vec[1],relative_velocity_vec[2],n_vec[0],n_vec[1],n_vec[2]))
        line.compute('U_tx={0:s}-U_n*{1:f}'.format(relative_velocity_vec[0],n_vec[0]))
        line.compute('U_ty={0:s}-U_n*{1:f}'.format(relative_velocity_vec[1],n_vec[1]))
        line.compute('U_tz={0:s}-U_n*{1:f}'.format(relative_velocity_vec[2],n_vec[2]))
        line.compute('U_t=(U_tx*U_tx+U_ty*U_ty+U_tz*U_tz)**0.5')

      if line is not None:
        if 'mag_velocity_rel' not in line[0][0].keys('node'):
          line.compute('mag_velocity_rel=({0:s}**2+{1:s}**2+{2:s}**2)**0.5'.format(relative_velocity_vec[0],relative_velocity_vec[1],relative_velocity_vec[2]))
        if 'nu_lam' not in line[0][0].keys('node'):
          if laminar_dynamic_viscosity is None:
            raise ValueError('Required name of field corresponding to "laminar_dynamic_viscosity" variable as input argument "laminar_dynamic_viscosity="')
          elif density is None:
            raise ValueError('Required name of field corresponding to "density" variable as input argument "density="')
          else:
            line.compute('nu_lam={0:s}/{1:s}'.format(laminar_dynamic_viscosity,density))
        if 'density' not in line[0][0].keys('node'):
          if density is None:
            raise ValueError('Required name of field corresponding to "density" variable as input argument "density="')
          elif line is not None:
            line.compute('density={0:s}'.format(density))
      if line is not None:
        current_nb_pts=len(line[0][0]['h'])
        print('current_nb_pts',current_nb_pts)
        print('nb_points',nb_points)
        if current_nb_pts != nb_points:
          print('The line extraction failed for cut {0:d} in zone {1:s}'.format(ihH,zn))
        else:
          successful_extraction[zn].append(ihH)

      if ihH==0:
        nb_vars=0
        var_list=[]
        for var in line[0][0].keys('node'):
          if var not in ['x','y','z','U_tx','U_ty','U_tz','R','hH','rt','theta']:
            nb_vars+=1
            var_list.append(var)
        data_BL=np.zeros((nb_cuts,nb_points,nb_vars,nb_inst))
      
      if line is not None:
        for it in range(nb_inst):
          for iv,var in enumerate(var_list):
            data_BL[ihH,:current_nb_pts,iv,it]=line[0][it][var]
            #print('line velocity',line[0][it]['U_t'])
      else:
        print('line is undefined for ihH',ihH)
    
    for it in range(nb_inst):
      print('instance',it,'read')
      for iv,var in enumerate(var_list):
        #print('iv',iv)
        #print('var',var)
        #print('var_list',var_list)
        #print('data_BL',data_BL[:,:,iv,it])
        BL_line_prof[zn][it][var]=data_BL[:,:,iv,it]

      # Correct wall value
      BL_line_prof[zn][it]['U_t'][:,0]=0.
      BL_line_prof[zn][it]['mag_velocity_rel'][:,0]=0.

  return BL_line_prof,successful_extraction

def compute_BL_params(BL_line_prof,var_detection,delta_pct_end=0.85,npts_interp=100,filter_size=None,filter_size_der=None,flag_compute_tau_wall=False):

  BL_param=Base()
  BL_param.init(zones=BL_line_prof.keys(),instants=BL_line_prof[0].keys())

  var_required=['mag_velocity_rel','U_t','density','nu_lam','h',var_detection]
  for var in var_required:
    if var not in BL_line_prof[0][0].keys('node'):
      raise ValueError('Required variable {0:s} not in boundary layer extractions provided'.format(var))

  if filter_size is None:
    filter_size_U=4
    filter_size_T=1
  else:
    filter_size_U=filter_size
    filter_size_T=filter_size

  if filter_size_der is None:
    filter_size_der=3
  else:
    filter_size_der=filter_size_der

  for zn in BL_line_prof.keys(): # pressure/suction sides
    nb_profs=BL_line_prof[zn][0]['h'].shape[0]
    for inst in BL_line_prof[zn].keys():
      BL_param[zn][inst]['exterior_stream_velocity']=np.zeros(nb_profs,)
      BL_param[zn][inst]['exterior_density']=np.zeros(nb_profs,)
      BL_param[zn][inst]['bl_thickness']=np.zeros(nb_profs,)
      BL_param[zn][inst]['displacement_thickness']=np.zeros(nb_profs,)
      BL_param[zn][inst]['momentum_thickness']=np.zeros(nb_profs,)
      if flag_compute_tau_wall:
        BL_param[zn][inst]['tau_wall']=np.zeros(nb_profs,)


      for il in range(nb_profs):


        idx,delta99=get_delta99_from_line(BL_line_prof[zn][0]['h'][il,:],BL_line_prof[zn][inst][var_detection][il,:],filter_size_var=filter_size_U,filter_size_der=filter_size_der,npts_interp=npts_interp,delta_pct_end=delta_pct_end)

        deltastar,deltatheta=get_boundary_layer_thicknesses_from_line(BL_line_prof[zn][0]['h'][il,:],BL_line_prof[zn][inst]['U_t'][il,:],BL_line_prof[zn][inst]['density'][il,:],idx)

        U_e=BL_line_prof[zn][inst]['mag_velocity_rel'][il,idx]
        rho_e=BL_line_prof[zn][inst]['density'][il,idx]

        if flag_compute_tau_wall:
          tau_wall=extract_BL_params.get_wall_shear_stress_from_line(BL_line_prof[zn][0]['h'][il,:],BL_line_prof[zn][inst]['U_t'][il,:],BL_line_prof[zn][inst]['density'][il,:],BL_line_prof[zn][inst]['nu_lam'][il,:],filter_size_var=filter_size_T,filter_size_der=filter_size_der,npts_interp=npts_interp)


        BL_param[zn][inst]['exterior_stream_velocity'][il]=U_e
        BL_param[zn][inst]['exterior_density'][il]=rho_e
        BL_param[zn][inst]['bl_thickness'][il]=delta99
        BL_param[zn][inst]['displacement_thickness'][il]=deltastar
        BL_param[zn][inst]['momentum_thickness'][il]=deltatheta
        if flag_compute_tau_wall:
          BL_param[zn][inst]['tau_wall'][il]=tau_wall

  return BL_param

def compute_surface_params(b_surf,BL_line_geom,BL_param, wall_shear_vec=None, pressure_gradient_vec=None,b_vol=None,pressure=None):


  flag_extract_tau_wall=False
  flag_compute_pressure_gradient=False

  if wall_shear_vec is not None:
    flag_extract_tau_wall=True

  if pressure_gradient_vec is None:
    if pressure is not None and b_vol is not None:
      flag_compute_pressure_gradient=True
      pressure_gradient_vec=('gradx_{0:s}'.format(pressure),'grady_{0:s}'.format(pressure),'gradz_{0:s}'.format(pressure))

  if flag_compute_pressure_gradient:
    t=Treatment('Gradient')
    t['base']=b_vol
    t['coordinates']=['x','y','z']
    t['variables']=[pressure,]
    tmp=t.execute()

    tmp.compute_cell_to_node(variables=pressure_gradient_vec)

    t=Treatment('interpolation')
    t['source']=tmp[:,:,['x','y','z',pressure_gradient_vec[0],pressure_gradient_vec[1],pressure_gradient_vec[2]]]
    t['target']=b_surf
    t['coordinates']=['x','y','z']
    t['nb_points']=1
    results=t.execute()

    for inst in b_surf[0].keys():
      for var in pressure_gradient_vec:
        b_surf[0][inst][var]=results[0][inst][var] #b_surf has been merged into a single zone


  for zn in BL_line_geom.keys():

    nb_profs=BL_param[zn][0].shape[0]

    for inst in BL_param[zn].keys():
      if flag_extract_tau_wall:
          BL_param[zn][inst]['tau_wall']=np.zeros(nb_profs,)
      if pressure_gradient_vec is not None:
          BL_param[zn][inst]['dpds']=np.zeros(nb_profs,)

    for ihH in progress_bar(range(nb_profs), label='getting surface params for BL on {0:s} '.format(zn)):

      pt1=np.array([BL_line_geom[zn][0]['x'][ihH],BL_line_geom[zn][0]['y'][ihH],BL_line_geom[zn][0]['z'][ihH]])
      n_vec=np.array([BL_line_geom[zn][0]['nx'][ihH],BL_line_geom[zn][0]['ny'][ihH],BL_line_geom[zn][0]['nz'][ihH]])
      chordline_dir=np.array([BL_line_geom[zn][0]['chord_dir_x'][ihH],BL_line_geom[zn][0]['chord_dir_y'][ihH],BL_line_geom[zn][0]['chord_dir_z'][ihH]])

      idx_extraction,_=b_surf[0][0].closest(pt1,['x','y','z'])


      for inst in BL_param[zn].keys():

        if pressure_gradient_vec is not None:

            gradP=np.array([b_surf[0][inst][pressure_gradient_vec[0]][idx_extraction],b_surf[0][inst][pressure_gradient_vec[1]][idx_extraction],b_surf[0][inst][pressure_gradient_vec[2]][idx_extraction]])
            gradPdn=np.dot(gradP,n_vec)
            gradP_t=gradP-gradPdn*n_vec
            BL_param[zn][inst]['dpds'][ihH]=np.sign(np.dot(gradP_t,chordline_dir))*np.linalg.norm(gradP_t)

        if flag_extract_tau_wall:
          tau=np.array([b_surf[0][inst][wall_shear_vec[0]][idx_extraction],b_surf[0][inst][wall_shear_vec[1]][idx_extraction],b_surf[0][inst][wall_shear_vec[2]][idx_extraction]])
          taun=np.dot(tau,n_vec)
          BL_param[zn][inst]['tau_wall'][ihH]=np.linalg.norm(tau-taun*n_vec)



