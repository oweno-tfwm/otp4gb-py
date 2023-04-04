# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:47:06 2023

@author: Signalis

Script to take seperate PBA_TRSE_Task_1 outputs and compile into a single
    matrix. Task 1 outputs are LSOA --> LSOA bus walk journeys
    
Inputs:
    - TSK1 output from original low amenity run (signalis)
    - TSK1 outputs from GM_Test run
    - TSK1 outputs from 800k north run, stopped for GM_Test
    - TSK1 outputs from rest of north run
    - TSK1 outputs from 800k debug redo run

Outputs:
    - Single compiled TSK1 matrix for the north
    

Each TSK1 output file contains the following columns:
    - o
    - d
    - jt
            
o: ------ LSOA Origin ID 
d: ------ LSOA Destination ID
jt: ----- sin corrected PT JT (Walk, Bus)

"""

#### Imports
import pandas as pd 
import time 
import os

#### Methodology
'''
1 - For each TSK1 matrix, create an OD_column
2 - Concatonate all TSK1 matrices
3 - Remove duplicates based on unique OD_column
4 - Format & export compiled TSK1 matrix
'''

#### Globals
save_dir = r'Y:\PBA\Analysis\Data(copy)\OTP data\compiled_outputs'
save_name  = '001_compiled_walk_bus_jt.csv'


#### Load data
print('Loading Bus & Walk Data.', time.strftime("%H:%M:%S", time.localtime()))
print('Loading low amenity run', time.strftime("%H:%M:%S - %Y", time.localtime()))
low_amenity_run_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\north_run_2(limited_amenity)_bus_jt_matrix.csv'
low_amenity_run = pd.read_csv(low_amenity_run_path)

print('Loading GM test run', time.strftime("%H:%M:%S - %Y", time.localtime()))
GM_test_run_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\GM_test_bus_jt_matrix.csv'
GM_test_run = pd.read_csv(GM_test_run_path)

print('Loading 800k partial north run', time.strftime("%H:%M:%S - %Y", time.localtime()))
north_run_800k_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\small_north_run(800k)_bus_jt_matrix.csv'
north_run_800k = pd.read_csv(north_run_800k_path)

print('Loading rest of north run', time.strftime("%H:%M:%S - %Y", time.localtime()))
rest_of_north_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\rest_of_north(pen_run)_bus_jt_matrix.csv'
rest_of_north = pd.read_csv(rest_of_north_path)

print('Loading 800k tsk2 debug run', time.strftime("%H:%M:%S - %Y", time.localtime()))
debug_run_800k_path = r'Y:\PBA\Analysis\Data(copy)\OTP data\final_debug_run_walk_bus_jt_matrix.csv'
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
compiled_matrix = compiled_matrix[['o', 'd', 'jt(mins)']]
print(time.strftime("%H:%M:%S - %Y", time.localtime()), '\nDone.')
# Save
compiled_matrix.to_csv(os.path.join(save_dir, save_name),
                       index=False)