# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:37:33 2023

@author: Signalis

Script to create a lookup of already requested trips by OTP. 
    When OTP filters through provided data to determine requests to send to
    the server, if OTP detects that a trip has already been requested, from the 
    lookup, skip this trip!
    
Inputs: 
    - OTP response data (contains requested trips)
    
Output: 
    - Lookup table containing previously requested OD pair trips
    
"""
#### Imports ####
import pandas as pd


# Load response matrices
# Load semi-complete final north run, with complete amenities list. 
#final_run_response_path = r'D:\Repositories\otp4gb-py\outputs_north_run_final\costs\AM\BUS_WALK_costs_20230608T0900_METRICS_FILE.csv'
#final_run_response_matrix = pd.read_csv(final_run_response_path)
# Load penultimate north run 
pen_run_response_path = r'D:\Repositories\otp4gb-py\outputs_rest_of_north\costs\AM\BUS_WALK_costs_20230608T0900-metrics.csv'
pen_run_response_matrix = pd.read_csv(pen_run_response_path)

final_run_response_matrix = pen_run_response_matrix

# Load complete north run 2 (over weekend on Signalis, only 3 amenities)
north_run2_response_path = r'E:\outputs_north_run_2\costs\AM\BUS_WALK_costs_20230608T0900_METRICS_FILE_north_2_signalis.csv'
north_run2_response_matrix = pd.read_csv(north_run2_response_path)
# Load complete Greater Manchester run, with complete amenities list. 
GM_authority_test_path = r'D:\Repositories\otp4gb-py\GM_test\costs\AM\BUS_WALK_costs_20230608T0900-metrics.csv'
GM_authority_test_matrix = pd.read_csv(GM_authority_test_path)


# We want to find trips already requested and make a lookup so that we can avoid
# making duplicate requests for these journeys.

# Add north_run_2 outputs into final_run_responses, then add GM_authority test trips
compiled_response_matrix = final_run_response_matrix.append(north_run2_response_matrix, ignore_index=True)
# Add GM_authority test trips
compiled_response_matrix = compiled_response_matrix.append(GM_authority_test_matrix, ignore_index=True)



# Format columns
compiled_response_matrix = compiled_response_matrix[['origin', 'destination', 'origin_id', 
                                                     'destination_id']].copy()

# Boolean check for trip presence in dataset
compiled_response_matrix['check'] = True
# Add OD_code
compiled_response_matrix['od_code'] = compiled_response_matrix['origin_id'] + '_' + compiled_response_matrix['destination_id']

initial_len = len(compiled_response_matrix)
# Remove duplicate trips within the compiled_response_matrix
compiled_response_matrix = compiled_response_matrix.drop_duplicates('od_code',
                                                               keep='first')
# Print statistics on removed duplicate trips (above)
final_len = len(compiled_response_matrix)
print(initial_len - final_len, 'duplicate trips have been removed.')
print(len(compiled_response_matrix), 'unique trips have been found')

# Format matrix columns
compiled_response_matrix = compiled_response_matrix[['od_code', 'check']].copy()
# Rename columns
compiled_response_matrix.rename(columns={'origin_id':'o',
                                         'destination_id':'d'},
                                inplace=True)
# Export data .
compiled_response_matrix.to_csv(r'D:\Repositories\otp4gb-py\final_debug_run\for_FINAL_run_run_trip_reqs.csv',
                                index = False)




# To implement the lookup, see code below:
    

# =============================================================================
# import pandas as pd
# 
# #Load file
# lookup = pd.read_csv(r'E:\OTP_Processing\OTP outputs\first_north_run\costs\AM\first_run_trip_reqs.csv')
# 
# #set index
# lookup.set_index('od_code', inplace=True, drop=True)
# 
# od_codes = ['E01028217_E01013880',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE',
#             'E01028217_E01028175',
#             'E01028217_E01027927',
#             'E043228217_E01027347',
#             'testtttttttttNNNOEOE']
# 
# import tqdm
# 
# for code in tqdm.tqdm(od_codes):
#     if code in lookup.index:
#         print(code, 'has previously been requested')
#     else:
#         print(code, 'must be requested NOW')
# =============================================================================

