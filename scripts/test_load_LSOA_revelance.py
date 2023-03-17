# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:31:18 2023

@author: Signalis
"""

import pandas as pd 
import os

LSOA_relevance = pd.read_csv(r'C:\Users\Signalis\Desktop\temp_outputs\LSOA_amenities.csv')
LSOA_relevance.set_index('LSOA11CD', inplace=True)

LSOA_id_test = 'E01000023'



total_val = LSOA_relevance.loc[LSOA_id_test]['totals']

print(total_val)


LSOA_test_id = 'E01000023'

LSOA_area_type = pd.read_csv(r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data\compiled_LSOA_area_types.csv')
LSOA_area_type.set_index('LSOA11CD', inplace=True)

area_type_string = LSOA_area_type.loc[LSOA_test_id]['RUC11NM']
area_type_code = LSOA_area_type.loc[LSOA_test_id]['RUC11CD']

print(area_type_code, area_type_string)





LSOA_area_type_weights = {'A1':1,
                          'B1':1,
                          'C1':1,
                          'C2':1,
                          'D1':1.25,
                          'D2':1.25,
                          'E1':1.5,
                          'E2':1.5}


weight = LSOA_area_type_weights[LSOA_area_type.loc[LSOA_test_id]['RUC11CD']]



for i in range(50):
    
    L_ids = LSOA_area_type.index
    L_id = L_ids[i]
    
    weight = LSOA_area_type_weights[LSOA_area_type.loc[L_id]['RUC11CD']]
    clss = LSOA_area_type.loc[L_id]['RUC11NM']
    
    print(L_id, clss, weight)
    
    
##############################################################################
''' Testing the itertools.product(zone_centroids.index) for OTP '''
import pandas as pd

zone_centroids = pd.read_csv(r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\assets\North_LSOA_centroids.csv')
zone_centroids.set_index('zone_id', inplace = True)

LSOA_relevance = pd.read_csv(r'C:/Users/Signalis/Documents/Repositories/otp4gb-py/Data/LSOA_amenities.csv')
LSOA_relevance.set_index('LSOA11CD', inplace=True)

origins = [] 
destinations = []
OD_pairs = []
counter = 0 
import itertools
from tqdm import tqdm
for o, d in tqdm(itertools.product(zone_centroids.index, zone_centroids.index)):
    counter += 1
    if counter % 250000 == 0:
        print(counter)
    if o == d:
        #print('o == d', counter)
        continue
    if LSOA_relevance.loc[d]['totals'] == 0:
        #print('irrelevant LSOA', counter)
        continue
    if LSOA_relevance.loc[d]['totals'] > 0:
        #print('Good Destination found')
        # Checks passed - append journey
        origins.append(o)
        destinations.append(d)
        OD_pairs.append('_'.join((str(o), str(d))))
    
    
print('Done, now creating DF')


n_irel = len(LSOA_relevance[LSOA_relevance]['totals'] == 0)
n_good = len(LSOA_relevance[LSOA_relevance]['totals'] > 0)

print('Total:',len(LSOA_relevance), '\nGood:', n_good, '\nBad:', n_irel)


################# Faster approach ####################
import pandas as pd
from shapely.geometry import Point
import  geopandas as gpd

#### Import data
zone_centroids = pd.read_csv(r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\assets\test_small_lsoa.csv')
zone_centroids.set_index('zone_id', inplace = True)
LSOA_relevance = pd.read_csv(r'C:/Users/Signalis/Documents/Repositories/otp4gb-py/Data/LSOA_amenities.csv')
LSOA_relevance.set_index('LSOA11CD', inplace=True)



zone_centroids_BnG = zone_centroids.copy()
zone_centroids_BnG['lat_long'] = zone_centroids_BnG[['latitude', 'longitude']].apply(Point, axis=1)
zone_centroids_BnG = gpd.GeoDataFrame(data = zone_centroids_BnG,
                                      geometry = 'lat_long',
                                      crs='EPSG:4326')

import numpy
length = 2
x = list(range(len(zone_centroids)))
mesh = numpy.meshgrid(*([x] * length))
result = numpy.vstack([y.flat for y in mesh]).T

# Create an LSOAid --> int32 range lookup
LSOA_ids = list(zone_centroids.index)


os_list = []
ds_list = []
od_pairs = []
counter = 0 
from tqdm import tqdm
import time

for i in tqdm(zip(result)):
    
    counter += 1
    if counter%250000 == 0:
         print((counter/len(result))*100, '%  @:', time.strftime("%H:%M:%S", time.localtime()))
    o = i[0][0]
    d = i[0][1]
    
    # Ignore same zone journey
    if o == d:
        continue
    
    # Check for relevance of Destination LSOA
    if LSOA_relevance.loc[LSOA_ids[d]]['totals'] > 0:
        # Destination is relevant!!
        origin = LSOA_ids[o]
        destin = LSOA_ids[d]
        
        os_list.append(origin)
        ds_list.append(destin)
        od_pairs.append('_'.join((origin, destin)))
    

OD_pairs = pd.DataFrame(data = {'Origins':os_list, 
                                'Destinations':ds_list,
                                'OD_pairs':od_pairs})


# DF of all OD pairs

#        LOG.info("Merging Origins to geometries")
# Join on zone centroid data for Origins
zone_centroids_BnG = zone_centroids_BnG.merge(how='left',
                                              right=OD_pairs,
                                              left_on=zone_centroids_BnG.index,
                                              right_on='Origins')
                                                      
# Re-classify dataframe as GeoDataFrame                                              
zone_centroids_BnG = gpd.GeoDataFrame(zone_centroids_BnG,
                                      crs='EPSG:27700')

        
# Rename Origin centroids col to Origin_centroids
zone_centroids_BnG.rename(columns={'geometry':'Origin_centroids'},
                          inplace=True)
# Re-specify geometry
zone_centroids_BnG.set_geometry('Origin_centroids',
                                 inplace=True,
                                 crs='EPSG:27700')

#LOG.info("Merging Destinations to geometries")
# Join on zone centroids data for Destinations
zone_centroids_BnG = zone_centroids_BnG.merge(how='left',
                                              right=zone_centroids_BnG_raw,
                                              left_on='Destinations',
                                              right_on=zone_centroids_BnG_raw.index)
# Re-classify dataframe as GeoDataFrame                                              
zone_centroids_BnG = gpd.GeoDataFrame(zone_centroids_BnG,
                                      crs='EPSG:27700')
        

        
        # rename centroids to destination centroids
        zone_centroids_BnG.rename(columns={'geometry':'Destination_centroids'},
                                  inplace=True)

        LOG.info("Calculating journey distances")
        # Work out distance between all OD pairs
        zone_centroids_BnG['distances'] = zone_centroids_BnG['Origin_centroids'].distance(zone_centroids_BnG['Destination_centroids'])


        # Print run stats for user
        print('\nMaximum distance possible:', str(max(zone_centroids_BnG['distances'])),
              '\nMinimum distance possible:', str(min(zone_centroids_BnG['distances'])))
        to_use = len(zone_centroids_BnG[zone_centroids_BnG['distances'] < filter_radius])
        all_rows = len(zone_centroids_BnG)
        print('\nOTP requests will be sent for:', to_use, 'zones out of a total:', all_rows, 'with a maximum standard distance of:', str(filter_radius), 'metres.\n')