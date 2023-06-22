# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:37:33 2023

@author: Signalis

Script to create a lookup of already requested trips by OTP.

Sometimes, VMs shut down overnight, or large OTP runs will be split over 
    numerous VM machines, leaving us with multiple files we don't want to 
    re-request. This script creates a lookup of previously requested trips
    from multiple cost-metric output files.

When OTP filters through provided data to determine requests to send to
    the server, if OTP identifies a that a trip from the lookup that has 
    already been requested, it skips this trip.
    
Inputs: 
    - OTP cost metrics data file(s) (contains prev. requested trips)
        the metrics file must contain the following headers:
           - origin_id
           - destination_id
           - mean_duration
           - min_duration
    
Output: 
    - Lookup table containing previously requested OD pair trips within the 
      provided OTP cost metric file(s) 
      
    - Optional: If enabled, the compiled cost metric (compiled from all input
      metrics) will also be saved
    
"""
#### Imports ####
import pandas as pd
import numpy as np
import os

#### Constants ####
# Path to OTP cost metric files (supports multiple OTP outputs)
RESPONSE_PATHS = [
    r"E:\OTP_Processing\OTP outputs\First OTP run (signalis)\BUS_WALK_costs_20230608T0900_COST_METRICS-metrics.csv"
    ] 

# Output path & filename to export trips previously requested lookup
OUT_PATH = r"C:\Users\Signalis\Desktop\temp_outputs"
OUT_FILENAME = "trips_previously_requested.csv"

# Should the compiled cost matrix be saved ?
SAVE_COMPILED_COSTS = False
COMPILED_COSTS_FILENAME = "compiled_COST_METRICS-metrics.csv"
# ^ If save is True - a compiled cost matrix will be saved in the same directory 
#       as the compiled lookup of requested trips with this filename

#### Script #### 
# Load response(s) into single matrix
for i in range(len(RESPONSE_PATHS)): 
    
    if i == 0:
        # Create compiled matrix
        path = RESPONSE_PATHS[i]
        print("Reading {}".format(path))
        compiled_matrix = pd.read_csv(path)
    else:
        # Append new matrices to compiled matrix
        path = RESPONSE_PATHS[i]
        print("Reading {}".format(path))
        matrix = pd.read_csv(path)
        compiled_matrix = pd.concat([matrix, compiled_matrix], ignore_index=True)
        compiled_matrix.reset_index(inplace=True, drop=True)

# Add OD_code
compiled_matrix["od_code"] = (
    compiled_matrix["origin_id"] + "_" + compiled_matrix["destination_id"]
    )

# If only one possible route has been found (number_itineraries == 1) then JT
#  info will be stored as mean_duration and not min_duration. When this is the 
#  case, set min_duration to be mean_duration. Then, sort by values & remove 
#  duplicate trip OD_codes, keeping the first that occurs (will be the 
#  shortest JT as a result of sorting the values)

# Conditions one which selection to use
conditions = (
    np.isnan(compiled_matrix["number_itineraries"]),
    compiled_matrix["number_itineraries"] == 0,
    compiled_matrix["number_itineraries"] == 1,
    compiled_matrix["number_itineraries"] > 1
    )

# Selections to use
selections = [
    1e99,
    1e99,
    compiled_matrix["mean_duration"],
    compiled_matrix["min_duration"]
    ]

# Apply selections using conditions
compiled_matrix["min_duration"] = np.select(conditions, selections)

# Sort by values
compiled_matrix = compiled_matrix.sort_values(by="min_duration")

# Initial number of trips
initial_len = len(compiled_matrix)

# Remove duplicate trips within the compiled_matrix
compiled_matrix = compiled_matrix.drop_duplicates("od_code",
                                                   keep="first"
                                                   )

# If any of the 1e99 JTs remain, convert these back to nan so they won't
#   throw off further analysis/cost calculations.
compiled_matrix.loc[compiled_matrix["min_duration"] == 1e99, 
                    "min_duration"] = np.nan

# Print statistics on removed duplicate trips (above)
final_len = len(compiled_matrix)
print(initial_len - final_len, "duplicate trips have been removed.")
print(len(compiled_matrix), "unique trips have been found")

# Save the cost matrix??? 
if SAVE_COMPILED_COSTS:
    print("Saving compiled cost matrix to {}".format(os.path.join(OUT_PATH, COMPILED_COSTS_FILENAME)))
    # Save compiled matrix
    compiled_matrix.to_csv(os.path.join(OUT_PATH,
                                        COMPILED_COSTS_FILENAME,
                                        ))
else:
    print("Skipped saving compiled cost matrix.")

# Format required columns
compiled_matrix = compiled_matrix[["origin", "destination", "origin_id", 
                                   "destination_id", "od_code"]].copy()

# Boolean check for trip presence in dataset
compiled_matrix["check"] = True

# Format matrix columns
compiled_matrix = compiled_matrix[["od_code", "check"]].copy()

# Export data .
compiled_matrix.to_csv(os.path.join(OUT_PATH, OUT_FILENAME), 
                       index=False)

print("Data exported to:\n {}".format(os.path.join(OUT_PATH, OUT_FILENAME)))
