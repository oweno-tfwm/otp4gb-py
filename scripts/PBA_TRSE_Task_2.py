# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:51:00 2023

@author: Signalis
"""

#### Imports ####

import pandas as pd
import math
import tqdm
import time
import os

#### Functions #### 

def path(A, B, x):
    f = A *math.sin(B*x)
    return f
    

def jt_correction(p, q, r, step_no):
    
    A = (p+q) # root or /2 or leave?
    
    if A==0:
        A=1
    
    B = math.pi / r
    
    step = r /step_no
    
    t = 0
    
    for i in range(step_no):
        
        x1 = step*i
        x2 = step*(i+1)
        y1 = path(A,B,x1)
        y2 = path(A,B,x2)
        delta_l = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        t = t + delta_l
    
    return t


#### Global Variables ####
step_size = 10  ## For pt_sin_correction
save_results_path = r'Y:\PBA\Analysis'
max_JT = 90  ## minutes

# Auto-create filename
save_results_filename = '_'.join((str(max_JT),
                                 'mins_max_time_',
                                 'BUS_WALK_rail_jt.csv'))
#### Load Data ####

## OTP Reponse data
# Manually enter path to OTP response data here:
OTP_dir = r'E:\OTP_Processing\OTP outputs\first_north_run\costs\AM'
filename = 'BUS_WALK_costs_20180608T0900-metrics.csv'

# Load morning Bus & Walk results
print('Loading  OTP results:', time.strftime("%H:%M:%S - %Y", time.localtime()))
AM_results = pd.read_csv(os.path.join(OTP_dir, filename))

print('Loaded OTP results:',  time.strftime("%H:%M:%S - %Y", time.localtime()))

## Rail data
rail_dir = r'Y:\PBA\Accessibility analysis\inputs'
base_rail_fn = 'NoRMS_JT_IGU_2018.csv'

# Load the rail JT data
rail_jt = pd.read_csv(os.path.join(rail_dir, base_rail_fn))

#### Pre-Processing ####

# Remove NaN values from the dataframe, based on `number_itineraries`
initial_len = len(AM_results)
AM_results = AM_results[AM_results['number_itineraries'].notna()]
cleaned_len = len(AM_results)
print('\n', (initial_len-cleaned_len), 'items have been removed for NaN')

# Set origin & destination id as index
AM_results.set_index(['origin_id', 'destination_id'],
                     inplace=True,
                     drop=False)

# Load LSOA to NoRMSid Lookup
LSOAid_to_NoRMSid = pd.read_csv(r'Y:\PBA\Analysis\Zones\LSOAid_to_NoRMSid_in_LSOAs.csv')


# NB: There will always be more Origins than Destinations as destinations must
#     contain a NoRMS centroid (statiom)


'''
For the results we have, we must work out the JT from every LSOA to those LSOAs
    that contain a NoRMS centroid... (a1 trips)
    
We must then also work out the JT from every relevant NoRMS containing LSOA as 
    a Origins to any LSOA that they may reach (a2 trips)
    
To clarify: a1 matrix will have every LSOA as O, but only LSOAs with NoRMS as D
            a2 matrix will have only LSOAs with NoRMS as O, but every LSOA as D
            L matrix is matrix of Rail JT
'''
origin_ids = list(AM_results['origin_id'].unique())
destination_ids = list(AM_results['destination_id'].unique())

a1_o = origin_ids
# a1 LSOA destinations must have a NoRMS centroid within 
a1_d = [d for d in LSOAid_to_NoRMSid['LSOA11CD'] if d in destination_ids]

a2_o = [o for o in LSOAid_to_NoRMSid['LSOA11CD'] if o in origin_ids]
a2_d = destination_ids


# NB: a1 & a2 JT matricies are made by Y:\PBA\Analaysis\create_a1/2_jt_martix.py
print('\nLoading a1, a2 JT matricies.')

# Load a1 & a2 JT matricies
a1_jts = pd.read_csv(r'Y:\PBA\Analysis\a1_jt.csv')
a2_jts = pd.read_csv(r'Y:\PBA\Analysis\a2_jt.csv')

# The a1 & a2 matricies will have NaN for routes where no PT could be used in 
# reasonable time period, as opposed to an OTP run error. As such, we should 
# remove them here. 
a1_jts = a1_jts[a1_jts['JT'].isna() == False]
a2_jts = a2_jts[a2_jts['JT'].isna() == False]

'''
From call with Adam (24/03) 'The TRSE analysis dataset excludes any journey if
                             a leg of that journey has a JT exceeding 60 mins'
                             
    Given this, we can remove any a1, a2 & NoRMS journeys IF the JT > 60 mins
'''

# Remove trips if the JT exceeds 60 minutes.
a1_jts = a1_jts[a1_jts['JT_mins'] <= max_JT]
a2_jts = a2_jts[a2_jts['JT_mins'] <= max_JT]
rail_jt = rail_jt[rail_jt['jt'] <= max_JT]

# Load NoRMS stations & corresponding LSOAs
LSOA_to_NoRMS_centres = pd.read_csv(r'Y:\PBA\Analysis\Zones\LSOAid_to_NoRMSid_in_LSOAs.csv')

# Join the NoRMS ids to a1 destination LSOA ids (add station IDs to LSOAs)
a1_jts = pd.merge(how='left',
                 left=a1_jts,
                 right=LSOA_to_NoRMS_centres[['LSOA11CD', 'ZoneID']],
                 right_on='LSOA11CD',
                 left_on='Destination')

# Re-formatting a1_jt.
a1_jts.drop(columns=['LSOA11CD'], inplace=True)
a1_jts.rename(columns={'ZoneID':'a1_D_NoRMS_ID'}, inplace=True)

# Set destination as index for a1_jts
a1_jts.set_index(['Origin'], drop=False,
                inplace=True)

# Join NoRMS ids to a2 Origin LSOA ids (add station IDs to LSOAs)
a2_jts = pd.merge(how='left',
                 left=a2_jts,
                 right=LSOA_to_NoRMS_centres[['LSOA11CD', 'ZoneID']],
                 right_on='LSOA11CD',
                 left_on='Origin')

# Re-formatting a2_jt
a2_jts.drop(columns=['LSOA11CD'], inplace=True)
a2_jts.rename(columns={'ZoneID':'a2_O_NoRMS_ID'}, inplace=True)


# Store data in a list, before making into a DF later
df_data = [] 

print('Analysing', len(a1_jts), 'potential trip origins', time.strftime("%H:%M:%S - %Y", time.localtime()))
for origin_id in tqdm.tqdm(list(a1_jts['Origin'].unique())):

    # Find all NoRMS LSOAs that origin_id can reach
    trips = a1_jts.loc[[origin_id]]
    
    # Add all possilbe train journeys
    trips_2 = trips.merge(how='left',
                        right=rail_jt,
                        left_on='a1_D_NoRMS_ID',
                        right_on='o')
    # Re-formatting 
    trips_2 = trips_2[['Origin', 'Destination', 'JT', 'JT_mins',
                   'a1_D_NoRMS_ID', 'd', 'jt']]
    trips_2.rename(columns={'JT':'a1_JT',
                            'JT_mins':'a1_JT_mins',
                            'Origin':'a1_Origin',
                            'Destination':'a1_Destination',
                            'd':'a2_O_NoRMS_ID',
                            'jt':'a1_a2_NoRMS_jt'},
                   inplace=True)
    
    # Add all possilbe LSOAs reachable from a2_Origin NoRMS stations
    trips_3 = trips_2.merge(how='left',
                            right=a2_jts,
                            on='a2_O_NoRMS_ID')
    
    # Re-formatting 
    trips_3.rename(columns={'Origin':'a2_Origin',
                            'Destination':'a2_Destination',
                            'JT':'a2_JT',
                            'JT_mins':'a2_JT_mins'},
                   inplace=True)
        
    # We now have a DF of complete possible trips for a given a1 Origin.
    # Set a1_Origin & a2_Destination as index to find variations for a given 
    # a1_O and a2_D journey utilising .loc[].
    trips_3.set_index(['a2_Destination'], drop=False,
                       inplace=True)
    
    # Calculate corrected JTs for potential trips
    sin_corrected_jts = [] 
    for a1_jt, a2_jt, NoRMS_jt in zip(trips_3['a1_JT_mins'], trips_3['a2_JT_mins'], trips_3['a1_a2_NoRMS_jt']):
        sin_corrected_jts.append( jt_correction(a1_jt, a2_jt, NoRMS_jt, step_size) )
    
    # Add sin corrected JTs
    trips_3['sin_corrected_jt'] = sin_corrected_jts
    
    
    # For each possible OD trip, find the fastes possible JT
    for destination_id in trips_3['a2_Destination'].unique():
        
        # Subset possible trips for given OD pair
        trips_possible = trips_3.loc[destination_id]
        
        best_jt = trips_possible['sin_corrected_jt'].min()

        # Store the best trip information
        df_data.append( (origin_id, destination_id, best_jt) )
        
        
# Create & save DataFrame of results:
results = pd.DataFrame(data = df_data,
                       columns = ['o','d','jt'])

result_origins = list(results['o'].unique())
result_destinations = list(results['d'].unique())

print('Trip information stored for', len(results), 'journeys.',
      '\nIncluding:', len(result_origins), 'origin zones.', '\nIncluding:',
      len(result_destinations), 'destination zones.')

results.to_csv(os.path.join(save_results_path,
                            save_results_filename), 
               index=False)
