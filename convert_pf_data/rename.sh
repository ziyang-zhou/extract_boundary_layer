#!/bin/bash

# Specify the directory where the files are located
directory="../frame_data"

# Navigate to the directory
cd "$directory" || exit 1

# Loop through the files with the pattern and rename them
for file in surface_mesh_frame_*.000000_0.vtu; do
    if [[ -f "$file" ]]; then
        # Extract the 'i' part of the filename
        i=$(echo "$file" | sed -n 's/surface_mesh_frame_\([0-9]\+\)\.000000_0/\1/p')
        
        # Rename the file
        new_name="surface_mesh_frame_$i"
        mv "$file" "$new_name"
        echo "Renamed: $file to $new_name"
    fi
done
