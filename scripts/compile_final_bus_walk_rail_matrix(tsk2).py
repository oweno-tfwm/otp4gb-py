# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:47:06 2023

@author: Signalis

Script to take seperate PBA_TRSE_Task_2 outputs and compile into a single
    matrix. Task 2 outputs are LSOA -> NoRMS -> NoRMS -> LSOA journeys
    
Inputs:
    - TSK2 output from original low amenity run (signalis)
    - TSK2 outputs from GM_Test run
    - TSK2 outputs from 800k north run, stopped for GM_Test
    - TSK2 outputs from rest of north run
    - TSK2 outputs from 800k debug redo run

Outputs:
    - Single compiled TSK2 matrix for the north
    

Each TSK2 output file contains the following columns:
    - o
    - d
    - jt
    - bus_pct
    - rail_pct
    
o: ------ LSOA Origin ID 
d: ------ LSOA Destination ID
jt: ----- sin corrected PT JT (Walk, Bus & Rail)
bus_pct:  % of addative JT made by bus & walking
rail_pct: % of addative JT made by rail

"""
#### Imports
import pandas as pd
import time
import os

#### Methodology
'''
1 - For each TSK2 matrix, create an OD_column
2 - Concatonate all TSK2 matrices
3 - Remove duplicates based on unique OD_column
4 - Format & export compiled TSK2 matrix
'''

#### Globals
save_dir = r'Y:\PBA\Analysis\Data(copy)\OTP data\compiled_outputs'
save_name = r'001_compiled_walk_bus_rail_jt.csv'


#### Load data
print('Loading Walk, Bus & Rail Data.', time.strftime("%H:%M:%S", time.localtime()))
print('Loading low amenity run', time.strftime("%H:%M:%S - %Y", time.localtime()))
low_amenity_run_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\60_mins_max_time__low_amenity_north_(fixed_tsk2)BUS_WALK_rail_jt.csv'
low_amenity_run = pd.read_csv(low_amenity_run_path)

print('Loading GM test run', time.strftime("%H:%M:%S - %Y", time.localtime()))
GM_test_run_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\60_mins_max_time__GM_test_run(fixed_tsk2)BUS_WALK_rail_jt.csv'
GM_test_run = pd.read_csv(GM_test_run_path)

print('Loading 800k partial north run', time.strftime("%H:%M:%S - %Y", time.localtime()))
north_run_800k_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\60_mins_max_time__800k_interupted_north_run(fixed_tsk2)BUS_WALK_rail_jt.csv'
north_run_800k = pd.read_csv(north_run_800k_path)

print('Loading rest of north run', time.strftime("%H:%M:%S - %Y", time.localtime()))
rest_of_north_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\60_mins_max_time__rest_of_north(pen_run)BUS_WALK_rail_jt.csv'
rest_of_north = pd.read_csv(rest_of_north_path)

print('Loading 800k tsk2 debug run', time.strftime("%H:%M:%S - %Y", time.localtime()))
debug_run_800k_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\60_mins_max_time__final_debug_run(fixed_tsk2)BUS_WALK_rail_jt.csv'
debug_run_800k = pd.read_csv(debug_run_800k_path)


#### 1 - Create OD columns
print("Creating OD code columns", time.strftime("%H:%M:%S - %Y", time.localtime()))
low_amenity_run['OD_code'] = low_amenity_run['o'] + '_' + low_amenity_run['d']
GM_test_run['OD_code'] = GM_test_run['o'] + '_' + GM_test_run['d']
north_run_800k['OD_code'] = north_run_800k['o'] + '_' + north_run_800k['d']
rest_of_north['OD_code'] = rest_of_north['o'] + '_' + rest_of_north['d']
debug_run_800k['OD_code'] = debug_run_800k['o'] + '_' + debug_run_800k['d']


#### 2 - Concat all matrices above
print('Joining respective outputs')
compiled_matrix = pd.concat([low_amenity_run, GM_test_run, 
                            north_run_800k, rest_of_north,
                            debug_run_800k])


#### 3 - Remove duplicate trips
print("Remove duplicate trips")
initial_len = len(compiled_matrix)
compiled_matrix.drop_duplicates('OD_code',
                                inplace=True)
final_len = len(compiled_matrix)
print(initial_len - final_len, "Duplicate trips have been removed.")

#### 4 Format & export 
print("\nExporting data for", len(compiled_matrix['o'].unique()), "unique origins and", len(compiled_matrix['d'].unique()), "destinations")
print(time.strftime("%H:%M:%S - %Y", time.localtime()))
compiled_matrix = compiled_matrix[['o', 'd', 'jt', 'bus_pct', 'rail_pct']]
print(time.strftime("%H:%M:%S - %Y", time.localtime()), '\nDone')

# Save
compiled_matrix.to_csv(os.path.join(save_dir, save_name),
                       index=False)
