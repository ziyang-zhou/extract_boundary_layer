#!/bin/bash

# Define the directory where your frame_data folder is located
frame_data_dir="../frame_data"

# Find all files in subfolders of frame_data and move them to the parent directory
find "$frame_data_dir" -type f -exec mv {} "$frame_data_dir" \;

