# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:00:34 2023

@author: Signalis

Script to complete TASK 1 from PBA TRSE Program.

Inputs: 
    OTP response data
    
Outputs: 
    Improved bus jt matrix at LSOA level for the North. 
"""


#### Imports #### 
import pandas as pd
import os
import tqdm
from math import isnan
#import geopandas as gpd

#### Data #### 
# Load response data
response_dir = r'D:\Repositories\otp4gb-py\outputs_north_run_final\costs\AM'
response_fn = 'BUS_WALK_costs_20230608T0900_METRICS_FILE.csv'

response_data = pd.read_csv(os.path.join(response_dir, 
                                         response_fn))

# Filter columns of response data
required_cols = ['origin', 'destination', 'origin_id', 'destination_id', 
                 'number_itineraries', 'mean_duration', 'min_duration']
response_data = response_data[required_cols]



# Remove NaN entries, and trips where OTP could not find appropriate routes in
#    time scale provided - mean_duration is NaN for these instances. 
initial_len = len(response_data)
response_data = response_data[response_data['mean_duration'].isna() == False].copy()
final_len = len(response_data)
print(initial_len - final_len, 'NaN trips have been removed from the dataset')

# Create matrix of LSOA --> LSOA bus_jt.
origins = list(response_data['origin_id'].unique())
destinations = list(response_data['destination_id'].unique())

# Set od_code as index for trip checking below
response_data.set_index(['origin_id', 'destination_id'], inplace=True, 
                        drop=True)

results_data = []
print('Iterating over:', len(origins), 'origin zones.')
# For every unique possible origin:
for o_id in tqdm.tqdm(origins):
    # For every unique possible destination:
        for d_id in destinations: 
            # Check if the trip exists
            
            
            try:  # Try find trip - store best JT if available
                mean_jt, min_jt = response_data.loc[(o_id, d_id)][['mean_duration', 'min_duration']].values
                
                if isnan(min_jt):
                    min_jt = mean_jt
                
                results_data.append( (o_id, d_id, min_jt/60) )
            except:
                # Trip does not exist
                continue

bus_jt_matrix = pd.DataFrame(data=results_data,
                             columns = ['o','d','jt(mins)'])

bus_jt_matrix.to_csv(r'Y:\PBA\Analysis\Data(copy)\OTP data\north_run_final(so_far)_bus_jt_matrix.csv',
                     index=False)
