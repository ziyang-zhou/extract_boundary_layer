
from functions import extract_BL_params
import vtk
import temporal
import os
import multiprocessing
from antares import *
import pdb

#Declare read and save path
vol_path = temporal.vol_path
bl_path = temporal.bl_path
mesh_path = temporal.mesh_path

os.makedirs(bl_path, exist_ok=True)
os.makedirs(vol_path, exist_ok=True)

# Read the number of nodes from the SLURM environment variable
num_processes = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
#Extraction parameter
length_extraction = temporal.length_extraction
var_detection = temporal.var_detection
nb_points = temporal.nb_points
axis = temporal.axis
axis_direction = temporal.axis_direction
relative_velocity_vec = temporal.relative_velocity_vec
# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk
#Read the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()

def process_chunk(chunk_start, chunk_end):
    r = Reader('hdf_antares')
    r['filename'] = vol_path + 'b_vol_{}_{}.h5'.format(chunk_start, chunk_end)
    b_vol = r.read()
    print('zones in b_vol', b_vol.keys())
    print('instants in b_vol', b_vol[0].keys())
    print('variables in each instant of b_vol', b_vol[0][0].keys())
    BL_line_prof, successful_extraction = extract_BL_params.extract_BL_profiles(
        b_vol, BL_line_geom, length_extraction, var_detection, nb_points,
        axis, axis_direction, relative_velocity_vec, str(temporal.density),
        laminar_dynamic_viscosity=str(temporal.laminar_dynamic_viscosity)
    )
    writer = Writer('hdf_antares')
    writer['filename'] = bl_path + 'BL_line_prof_{}_{}'.format(chunk_start, chunk_end)
    writer['base'] = BL_line_prof
    writer.dump()

if __name__ == '__main__':
    # Initialize BL_line_geom here if needed
    pool = multiprocessing.Pool(processes=num_processes)
    # Calculate start and end indices for each chunk
    chunk_ranges = [(i * step_per_chunk + starting_timestep, (i + 1) * step_per_chunk + starting_timestep) for i in range(num_chunks)]
    # Process chunks in parallel
    pool.starmap(process_chunk, chunk_ranges)
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()


