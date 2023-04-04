# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:35:20 2023

@author: Signalis

Faster script for PBA TRSE Task 1

________________________________________________________________

Created on Tue Mar 28 16:00:34 2023

@author: Signalis

Script to complete TASK 1 from PBA TRSE Program.

Inputs: 
    OTP response data
    
Outputs: 
    Improved bus jt matrix at LSOA level for the North.
    
    
Before running this script, ensure the following are updated: 

    L34 - OTP response file path
    L63 - Results save path
"""


#### Imports #### 
import pandas as pd
import os
import time

save_filename = r'Y:\PBA\Analysis\Data(copy)\OTP data\final_debug_run_walk_bus_jt_matrix.csv'

#### Data #### 
# Load response data
response_dir = r'D:\Repositories\otp4gb-py\final_debug_run\costs\AM'
response_fn = 'BUS_WALK_costs_20230608T0900-metrics.csv'

response_data = pd.read_csv(os.path.join(response_dir, 
                                         response_fn))
print('Loaded responses', time.strftime("%H:%M:%S - %Y", time.localtime()))
print(os.path.join(response_dir, response_fn))

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
print('Processing responses', time.strftime("%H:%M:%S - %Y", time.localtime()))

'''
Create a column of OD pair codes (format "O_D") used as identifiers for unique
    trips. Apply a groupby on this column, selecting minimum value within 
    min_duration column. The resultant matrix is tall possible unique trips 
    from the OTP response data with minimum JTs (minutes) appended
'''
# Create OD code column
response_data['OD_code'] = response_data['origin_id'] + '_' + response_data['destination_id']

# Use mean_duration data where `min_duration` is NaN (only one trip found)
response_data.loc[response_data['min_duration'].isnull(), 'min_duration'] = response_data['mean_duration']

# Group data by OD_code, leaving the best jt for every possible OD_code trip
grouped_responses = response_data.groupby('OD_code').agg({'min_duration': 'min'})

# Re-append Origin & Destination info
grouped_responses['od_code'] = grouped_responses.index

grouped_responses[['o', 'd']] = grouped_responses['od_code'].str.split('_', 
                                                                       expand=True)

# JTs currently in seconds, convert to minutes
grouped_responses['jt(mins)'] = grouped_responses['min_duration']/60


grouped_responses = grouped_responses[['o', 'd', 'jt(mins)']]

print('Processing Finished', time.strftime("%H:%M:%S - %Y", time.localtime()))

grouped_responses.to_csv(save_filename,
                     index=False)
