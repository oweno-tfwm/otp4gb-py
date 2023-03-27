# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:39:28 2023

@author: Signalis

Using OTP metrics data, obtain JTs from every potential northern LSOA as an 
    origin, to every LSOA that contains a NoRMS centroid (train station) as 
    destinations. This forms the a1 JTs used in the calculation of public 
    transport Sin() corrected JTs
    
    
Inputs: 
    OTP responses
    LSOAid_to_NoRMSid (lookup of NoRMS within LSOAs)
    
Outputs:
    a1 JT matrix

"""

#### Imports ####

import pandas as pd
#import geopandas as gpd
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

# Load Rail data
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


## Create a1 JT matrix

a1_jt_os = []
a1_jt_ds  = []
a1_jts = []
a1_errors = []

for o in tqdm.tqdm(a1_o):  # All LSOAs
    for d in a1_d:  # Only LSOAs containing NoRMS centroid & within range
        try:
            mean_jt, min_jt = AM_results.loc[(o, d)][['mean_duration', 'min_duration']].values
            a1_jt_os.append(o)
            a1_jt_ds.append(d)
             
            # Check if min_jt is nan, if so, use mean_jt
            if isnan(min_jt):
                min_jt = mean_jt
             
            a1_jts.append(min_jt)
             
            # If this fails, the precise od pair journey was not returned from OTP
        except:
            a1_errors.append('_'.join((o, d)))

# Store the data in a DataFrame
df_data = []       
for a, b, c in zip(a1_jt_os, a1_jt_ds, a1_jts):
    df_data.append((a, b, c))

a1_jt = pd.DataFrame(data = df_data,
                     columns = ['Origin', 'Destination', 'JT'])

a1_jt['JT_mins'] = a1_jt['JT']/60

# Save data
a1_jt.to_csv(r'Y:\PBA\Analysis\a1_jt.csv',
             index = False)

##### testing #####
o = 'E01019827'
d = 'E01007805'
test = AM_results[ (AM_results['origin_id'] == o) & (AM_results['destination_id'] == d) ]
