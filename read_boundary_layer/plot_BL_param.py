# This script plots all relevant BL parameter for PSD prediction. 
# These include Ue,Cf,delta_95,delta_star,theta,beta_c,uv+max,RT

import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math
import pandas as pd
import pdb

#Read the filepath
save_path = temporal.save_path
anna_extraction_file = '/home/ziyz1701/storage/CD_airfoil/temp/3___Boundary layer development_ch3/Result/H08_BL_param.csv'
andrea_extraction_file = '/home/ziyz1701/storage/CD_airfoil/FAN_2025/NN_simplified/01-DATABASES/database_SCONE-DNS_interpolated_NEW/H08_BLparams.csv'

anna_BL_param_df = pd.read_csv(anna_extraction_file,index_col=None)
andrea_BL_param_df = pd.read_csv(andrea_extraction_file,delimiter = ' ',index_col=None)

print(anna_BL_param_df.keys())
print(andrea_BL_param_df.keys())

#Plot delta_95
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['delta_95/c'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['delta']/andrea_BL_param_df['chord'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'delta_95.png')
plt.close()
#Plot delta_star
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['delta_star/c'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['delta_star']/andrea_BL_param_df['chord'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'delta_star.png')
plt.close()
#Plot theta
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['theta/c'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['theta']/andrea_BL_param_df['chord'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'theta.png')
plt.close()
#Plot beta c
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['Betac'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['beta_c']/andrea_BL_param_df['chord'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'betac.png')
plt.close()
#Plot RT
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['RT'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['u_tau_wall']*andrea_BL_param_df['delta']*150000*np.sqrt(abs(andrea_BL_param_df['cf(Ue)'])/2),label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'RT.png')
plt.close()
#Plot tau wall
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['utau'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['u_tau_wall'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'u_tau_wall.png')
plt.close()
#Plot Cf (normalized by Ue)
plt.plot(anna_BL_param_df['x/c'],anna_BL_param_df['Cf'],label = 'Anna')
plt.plot(andrea_BL_param_df['x']-1.0,andrea_BL_param_df['cf(Ue)'],label = 'Andrea')
plt.legend()
plt.savefig(save_path + 'cf.png')
plt.close()
