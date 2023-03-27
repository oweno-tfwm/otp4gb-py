# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:40:17 2023

@author: Signalis

Using OTP metrics data, obtain JTs from every Northern LSOA that contains a 
    NoRMS centroid (train station) as Origins, to every posssible other LSOA
    (given JT, distance & amenity restrictions) within the North. This forms 
    our a2 JT matrix  used in the calculation of public transport Sin()
    corrected JTs
    
    
Inputs: 
    OTP responses
    LSOAid_to_NoRMSid (lookup of NoRMS within LSOAs)
    
Outputs:
    a2 JT matrix
    
"""

#### Imports ####

import pandas as pd
from math import isnan
import tqdm
import time
import os

#### Load Data ####

## OTP Reponse data
# Manually enter path to OTP response data below:
OTP_dir = r'C:\Users\Signalis\Desktop\Ongoing - To Complete\LSOA TRSE OTP work\OTP outputs\first_north_run\costs\AM'
filename = 'BUS_WALK_costs_20180608T0900-metrics.csv'

# Load morning Bus & Walk results
print( time.strftime("%H:%M:%S - %Y", time.localtime()))
AM_results = pd.read_csv(os.path.join(OTP_dir, filename))
print('Loaded data:',  time.strftime("%H:%M:%S - %Y", time.localtime()))

## Rail data
rail_dir = r'Y:\PBA\Accessibility analysis\inputs'
base_rail_fn = 'NoRMS_JT_IGU_2018.csv'


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


'''
Note: There will always be more Origins than destinations. We try every LSOA as
      an origin, but only select destinations that have at least one amenity, and
      are within the filter radius distance (Crow Flies)
'''

origin_ids = list(AM_results['origin_id'].unique())
destination_ids = list(AM_results['destination_id'].unique())

'''
For the results we have, we must work out the JT from every LSOA to those LSOAs
    that contain a NoRMS centroid... (a1)
    
We must then also work out the JT from every relevant NoRMS containing LSOA as 
    a Origins (a2)
    
To clarify: a1 matrix will have every LSOA as Origin, only LSOA with NoRMS as D
            a2 matrix will have only LSOAs with NoRMS as Origin, every LSOA as D
            L matrix is matrix of Rail JT
'''

# a1 LSOA origins are all LSOAs
a1_o = origin_ids
# a1 LSOA destinations must have a NoRMS centroid within 
a1_d = [d for d in LSOAid_to_NoRMSid['LSOA11CD'] if d in destination_ids]

# a2 LSOA Origins MUST have a NoRMS centroid withini
a2_o = [o for o in LSOAid_to_NoRMSid['LSOA11CD'] if o in origin_ids]
# a2 LSOA destinations are all LSOAs
a2_d = destination_ids



## Create a2 JT martix
a2_jt_os = []
a2_jt_ds  = []
a2_jts = []
a2_errors = []         

for o in tqdm.tqdm(a2_o): # Only LSOAs containing NoRMS
    for d in a2_d: # All LSOAs within range
        try:
            mean_jt, min_jt = AM_results.loc[(o, d)][['mean_duration', 'min_duration']].values
            a2_jt_os.append(o)
            a2_jt_ds.append(d)
            
            # Check if min_jt is nan, if so, use mean_jt
            if isnan(min_jt):
                min_jt = mean_jt
            
            a2_jts.append(min_jt)
            
            # If this fails, the precise od pair journey was not returned from OTP
        except:
            a2_errors.append('_'.join((o, d)))

# Store the data in a DataFrame
df_data = []       
for a, b, c in zip(a2_jt_os, a2_jt_ds, a2_jts):
    df_data.append((a, b, c))

a2_jt = pd.DataFrame(data = df_data,
                     columns = ['Origin', 'Destination', 'JT'])

a2_jt['JT_mins'] = a2_jt['JT']/60#

# Save data
a2_jt.to_csv(r'Y:\PBA\Analysis\a2_jt.csv',
             index = False)