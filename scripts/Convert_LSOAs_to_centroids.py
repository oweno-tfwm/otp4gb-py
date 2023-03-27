# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:28:05 2023

@author: Signalis
"""
#### Imports #### 
import geopandas as gpd
import os
import pandas as pd
import shapely

# Get path for data directory.. (currently within \scripts)
os.chdir('..')
data_dir = os.path.join(os.getcwd(), 'Data')


# Load LSOAs data
LSOA_dir = r'Y:\Data Strategy\GIS Shapefiles\LSOA'
LSOA_fn = 'Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.shp'
LSOAs = gpd.read_file(os.path.join(LSOA_dir, LSOA_fn))

# Convert LSOAs from BnG (27700) to WGS (4326) -- OTP takes coordinates in 
#    WGS:4326 (Lat, Long) for routing calculations. 
LSOAs = LSOAs.to_crs('EPSG:4326')

# Create centroids from boundaries file
LSOAs['centroids'] = LSOAs['geometry'].centroid
LSOAs['LAT_LONG'] = LSOAs[['LONG', 'LAT']].apply(shapely.geometry.Point, axis=1)


# File format for centroids.csv for OTP4GB-py. 
'''
Centroids CSV: CSV file containing all the zone centroids for the routing, with the following columns:
 - zone_id
 - zone_name
 - zone_system
 - latitude
 - longitude
'''
# Specify zone_system
LSOAs['zone_system'] = 'LSOA'

# Remove non-Enlgish LSOAs
LSOAs = LSOAs[LSOAs['LSOA11CD'].str.contains('E')]

# Formatting to OTP requirements
LSOAs.rename(columns = {'LSOA11CD':'zone_id',
                        'LSOA11NM':'zone_name',
                        'LAT':'latitude',
                        'LONG':'longitude'},
                        inplace = True)

# Format final centroids csv. 
required_cols = ['zone_id',
                 'zone_name',
                 'zone_system',
                 'latitude',
                 'longitude']


'''
For our analysis, we only want to consider LSOAs within the North (as 
    assessment of local travel using PT, i.e. Not MCR --> LDN).

Hence, create a bounding polygon of the North (inc. bits of midlands as DfT
    have been intrested in this, and will capture travel from North-Midlands
    boundary) and then perform a spatial check for each LSOA centroid:
        
        Does the LSOA fall within the North bounding polygon??
'''


# Create a polygon of the north: 
'''
coordinates from:
    http://bboxfinder.com/#52.988337,-4.746094,55.887635,0.933838

-4.746094,52.988337,0.933838,55.887635 (Long, Lat)

52.988337,-4.746094,55.887635,0.933838 (Lat, Long)

'''


# function to return polygon
def bbox(long0, lat0, lat1, long1):
    from shapely.geometry import Polygon
    return Polygon([[long0, lat0],
                   [long1,lat0],
                   [long1,lat1],
                   [long0, lat1]])

# Create a polygon of the north
north_poly = bbox( 0.933838, 52.988337, 55.887635, -4.746094)

# Convert to gdf to CRS used above
north_poly_gdf = gpd.GeoDataFrame(pd.DataFrame(['NorthernPoly'], columns=['geometry']),
                              geometry = [north_poly],
                              crs = 'epsg:4326')

# Save for visualisation on QGIS
# =============================================================================
# north_poly_gdf.to_file(r'C:\Users\Signalis\Desktop\temp_outputs\NorthernPolygon.geojson',
#                        driver='GeoJSON')
# =============================================================================

# Find LSOAid's for all LSOAs that do reside within the North
print('Finding Northern LSOAs')
north_LSOAs = []

non_north_LSOAs = []
counter = 0 
for LSOAid, LSOA_centroid in zip(LSOAs['zone_id'], LSOAs['centroids']):
    counter += 1
    # Check if the current LSOA centroid resides within the `north_poly`
    if north_poly.contains(LSOA_centroid):
        # The LSOA is a northern one
        north_LSOAs.append(LSOAid)
    else:
        # LSOA is outside the north
        non_north_LSOAs.append(LSOAid)
        continue
    
    if counter % 2500 == 0:
        print(str(counter) + '/' + str(len(LSOAs)))

print('Finished. ', str(len(north_LSOAs)), 'Northern LSOAs have been found')

# Filter out Non-northern LSOAs
north_LSOAs = LSOAs[LSOAs['zone_id'].isin(north_LSOAs)].copy()
out_of_north_LSOAs = LSOAs[LSOAs['zone_id'].isin(non_north_LSOAs)].copy()

# Filter required columns only
#print(type(north_LSOAs))
required_cols.append('centroids')
north_LSOAs = north_LSOAs[required_cols]
out_of_north_LSOAs = out_of_north_LSOAs[required_cols]

# Re-sepcify geometry & crs of geodataframe
north_LSOAs = gpd.GeoDataFrame(data = north_LSOAs,
                               geometry = 'centroids',
                               crs = 'EPSG:4326')

out_of_north_LSOAs = gpd.GeoDataFrame(data = out_of_north_LSOAs,
                                      geometry = 'centroids',
                                      crs = 'EPSG:4326')
required_cols.remove('centroids')

'''
To visualise the North & Non-Northern LSOAs, see:
    https://i.imgur.com/ajVe0dU.png
'''

# Save the data (for QGIS visualisation)
# =============================================================================
# north_LSOAs.to_file(r'C:\Users\Signalis\Desktop\temp_outputs\North_LSOAs.geojson',
#                     driver = 'GeoJSON')
# 
# out_of_north_LSOAs.to_file(r'C:\Users\Signalis\Desktop\temp_outputs\Out_of_North_LSOAs.geojson',
#                            driver = 'GeoJSON')
# =============================================================================

# Filter required columns only
north_LSOAs = north_LSOAs[required_cols]
out_of_north_LSOAs = out_of_north_LSOAs[required_cols]

# Export the Nortern LSOA centroids
north_LSOAs.to_csv(os.path.join(data_dir, 'North_LSOA_centroids.csv'),
                   index=False)

# Code snippets to create zone centroid files for smaller areas.
###############################################################################
# =============================================================================
# # Restrict area to Manchester for proof of concept
# manchester = LSOAs[LSOAs['zone_name'].str.contains('Manchester|Oldham')].copy()
# manchester_area = LSOAs[LSOAs['zone_name'].str.contains('|'.join(('Manchester', 
#                                                                   'Oldham',
#                                                                   'Trafford')))].copy()
# 
# # Format final columns
# manchester = manchester[required_cols]
# 
# desktop_dir = r'C:\Users\Signalis\Desktop'
# filename = 'centroids.csv'
# manchester.to_csv(os.path.join(desktop_dir, filename))
# =============================================================================

# =============================================================================
# # Restict LSOA zones for sheffield only
# sheff = LSOAs[LSOAs['zone_name'].str.contains('Sheff|Rother')].copy()
# 
# sheff = sheff.to_crs('epsg:4326')
# 
# sheff['centroids'] = sheff['geometry'].centroid
# sheff['zone_system'] = 'LSOA'
# 
# 
# sheff = sheff.to_crs('epsg:4326')
# 
# 
# sheff.rename(columns = {'LSOA11CD':'zone_id',
#                         'LSOA11NM':'zone_name',
#                         'LAT':'latitude',
#                         'LONG':'longitude'},
#                         inplace = True)
# 
# required_cols = ['zone_id',
#                  'zone_name',
#                  'zone_system',
#                  'latitude',
#                  'longitude']
# 
# sheff_LSOA = sheff[required_cols]
# 
# sheff_LSOA.to_csv(r'C:\Users\Signalis\Desktop\temp_outputs\Sheff_centroids.csv')
# =============================================================================
###############################################################################