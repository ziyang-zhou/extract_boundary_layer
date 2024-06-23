import os
import pandas as pd
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")
    else:
        print(f"Folder '{path}' already exists.")
    return None

def read_and_extract_variable(file_location, variable):
    # Read CSV file into a DataFrame
    data = pd.read_csv(file_location)

    # Iterate over all columns and check if variable is a substring
    matching_columns = [col for col in data.columns if variable in col]

    if matching_columns:
        # If there are matching columns, extract the variable from the first one
        matched_column = matching_columns[0]
        signal = np.array(data[matched_column])
        print(f"Variable '{variable}' matched with column '{matched_column}'.")
        return signal
    else:
        print(f"Variable '{variable}' not found in any column.")
        return None  # or handle the case when the variable is not found
