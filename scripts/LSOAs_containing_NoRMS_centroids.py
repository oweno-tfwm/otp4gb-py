# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:44:02 2023

@author: Signalis

Create a lookup of LSOA's that contain a NoRMS centroid for improved JT 
    analysis. Originally going ot use JTs to the NoRMS centroids, but with 
    deadline pressures we will approximate LSOA centroids as NoRMS centroids
    if the LSOA contains a NoRMS centroid.
    
Input: 
    - List of LSOAs
    - Geospatial set of NoRMS centroids
    
Output:
    - Lookup of LSOAs that contain a NoRMS centroid
    
"""
#### Imports ####
import geopandas as gpd
import pandas as pd 
import os
#import time  # Used in points_in_area()
import tqdm

#### Functions ####

def points_in_areas(points, polygons, step):
    
    # Given a list of points, and polygons, will determine the number of 
    # polys from `polygons` that appear within each point from `points`
    
    points_in_polys = [] 
    #p_counter = 0


    for poly in tqdm.tqdm(polygons):
        
        points_in_poly=0
        for point in points:
        	if poly.contains(point):
        		points_in_poly+=1
        
        points_in_polys.append(points_in_poly)
        
        ## Un-comment below for a progress check if NOT using tqdm.tqdm()
        #p_counter+=1
        #if p_counter%step == 0:
        #	print(str(p_counter)+ '/'+str(len(polygons)), time.strftime("%H:%M:%S - %Y", time.localtime()))

    return points_in_polys

#### Load Data ####

## LSOA geometries
LSOA_dir=r'Y:\Data Strategy\GIS Shapefiles\LSOA'
LSOA_fn='Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.shp'
LSOAs=gpd.read_file(os.path.join(LSOA_dir, LSOA_fn))

# Filter only English LSOAs
LSOAs = LSOAs[LSOAs['LSOA11CD'].str.contains('E') == True]

# Get unqiue zone codes for lookup table
LSOA_codes = LSOAs['LSOA11CD'].unique()


## NoRMS geometries
NoRMS_dir=r'Y:\Data Strategy\GIS Shapefiles\NoRMS zones\_SS'
NoRMS_fn='TfN_Zones_Combined.shp'
NoRMS_zones = gpd.read_file(os.path.join(NoRMS_dir, NoRMS_fn))

# This NoRMS file is boundaries - apply .centroids() to convert polys to centroids
NoRMS_zones['centroids'] = NoRMS_zones['geometry'].centroid

# Filter columns
NoRMS_zones.drop(columns=['geometry',  'PNZone', 'PLDZone', 'PNFlag'],
                 inplace=True) 

# Convert crs to same as LSOAs GeoDataFrame (for points_in_areas())
NoRMS_zones.set_geometry('centroids', 
                         crs = 'EPSG:27700')



#### LSOAs to NoRMS station IDs Lookup ####
'''
This section makes an LSOAid--> #NoRMS_stations--> StationID lookup
    used for the creation of a1 & a2 JT matrices for TRSE LSOA analysis. 
'''

data = []
print('Now iterating over:', len(NoRMS_zones), 'items.')
# For each centroid and corresposnding centroid ID
for point, zone_id in tqdm.tqdm(zip(NoRMS_zones['centroids'], NoRMS_zones['ZoneID'])):
    
    # For each LSOA and LSOAid
    for polygon, LSOAid in zip(LSOAs['geometry'], LSOAs['LSOA11CD']):
        
        # Does LSOA contain a centroid?
        if polygon.contains(point): 
            # Current LSOA contains a NoRMS centroid
            data.append((LSOAid, zone_id))
    
# Store results in DataFrame
LSOAid_to_NoRMSid = pd.DataFrame(data = data,
                                 columns = ['LSOA11CD', 'ZoneID'])
# Formatting
LSOAid_to_NoRMSid['ZoneID'] = LSOAid_to_NoRMSid['ZoneID'].astype(str)

# Some LSOAs have more than one NoRMS station - This joins them
ids_per_LSOA = LSOAid_to_NoRMSid.groupby('LSOA11CD', as_index=False).transform(lambda x: ','.join(x))

# Adds the above line to the DataFrame
LSOAid_to_NoRMSid['ZoneIDs'] = ids_per_LSOA['ZoneID']

# Remove duplicated based on ZoneID (we capture duplicates above)
LSOAid_to_NoRMSid = LSOAid_to_NoRMSid.drop_duplicates('LSOA11CD', 
                                                      keep = 'first')

# Save the lookup
LSOAid_to_NoRMSid.to_csv(r'Y:\PBA\Analysis\Zones\LSOAid_to_NoRMSid_in_LSOAs.csv', 
                         index = False) 



#### Analysis ####
'''
run points_in_areas() using NoRMS_centroids as points and LSOA boundaries as
    polygons. Then, construct a pd.DataFrame with index: LSOA_codes and 
    data: output of point_in_areas()
'''
print('\nFinding NoRMS centroids within LSOAs')
NoRMS_station_in_LSOA = points_in_areas(points=NoRMS_zones['centroids'],
                                        polygons=LSOAs['geometry'],
                                        step=2500)
df_data = [] 
for i, j in zip(LSOA_codes, NoRMS_station_in_LSOA):
    df_data.append((i, j))


# Construct a DataFrame to store and export results
NoRMS_in_LSOAs = pd.DataFrame(data=df_data,
                              columns=['LSOA11CD', 'NoRMS_in_LSOAs'])

#### Further analysis / experiments #### 
'''
We can see that a number of NoRMS_zones cannot be found within LSOAs. Lets, 
   keep track of them & visualise in QGIS.
   
   See: https://i.imgur.com/OfOjK5s.png  for a visualisation of this data. 
   As can be seen, those NoRMS centroids not within LSOAs are not within the 
   North and do not affect our analysis.
   
   Un-comment below, run, and visualise in QGIS to check. 
'''

# =============================================================================
# empty_points = []
# i = 0
# for point in tqdm.tqdm(NoRMS_zones['centroids']):
#     i +=1 
# 
#     polys_containing_point = 0 
#     
#     for poly in LSOAs['geometry']:
#         
#         if poly.contains(point):
#             # If the centroid has been found in an LSOA, update the count
#             polys_containing_point += 1
#     
#     if polys_containing_point == 0:
#         # If the centroid cannot be found in any LSOA poly, add to list
#         empty_points.append(i)
#         
# ## Create a DataFrame of NoRMS_zone_ids that are NOT in any English LSOAs
# NoRMS_NOT_in_LSOAs = NoRMS_zones[NoRMS_zones['ZoneID'].isin(empty_points)].copy()
# NoRMS_NOT_in_LSOAs = gpd.GeoDataFrame(NoRMS_NOT_in_LSOAs,
#                                       crs = 'EPSG:27700')
# NoRMS_NOT_in_LSOAs.set_geometry('centroids',
#                                 crs = 'EPSG:27700',
#                                 inplace=True)
# 
# 
# not_empty_points = []
# i=0
# for point in tqdm.tqdm(NoRMS_zones['centroids']):
#     i +=1 
#     #print(i)
#     polys_containing_point = 0 
#     
#     for poly in LSOAs['geometry']:
#         
#         if poly.contains(point):
#             # If the centroid has been found in an LSOA, update the count
#             polys_containing_point += 1
#     
#     if polys_containing_point != 0:
#         # If the centroid cannot be found in any LSOA poly, add to list
#         not_empty_points.append(i)
#         
# NoRMS_IN_LSOAs = NoRMS_zones[NoRMS_zones['ZoneID'].isin(not_empty_points)].copy()
# NoRMS_IN_LSOAs = gpd.GeoDataFrame(NoRMS_IN_LSOAs,
#                                   crs = 'EPSG:27700')
# NoRMS_IN_LSOAs.set_geometry('centroids',
#                             crs = 'EPSG:27700',
#                             inplace=True)
# 
# #### Save Data ####
# save_dir = r'C:\Users\Signalis\Desktop\temp_outputs'
# 
# 
# NoRMS_NOT_in_LSOAs.to_file(os.path.join(save_dir, 'NoRMS_centroids_not_in_LSOAs.shp'))
# 
# NoRMS_IN_LSOAs.to_file(os.path.join(save_dir, 'NoRMS_centroids_in_LSOAs.geojson'),
#                         driver='GeoJSON')
# =============================================================================

##############################################################################
