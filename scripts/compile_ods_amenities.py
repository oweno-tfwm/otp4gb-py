# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:07:25 2023

@author: Signalis

Script to compile a lookup of # of amenities within an LSOA 

"""
####FUNCTIONS####
def points_in_areas(points, polygons, step):
    
    points_in_polys = [] 
    counter = 0
    #breakpoint()

    for poly in polygons:
        
        points_in_poly=0
        for point in points:
        	if poly.contains(point):
        		points_in_poly+=1
        
        points_in_polys.append(points_in_poly)
        counter+=1
        if counter%step == 0:
        	print(str(counter)+ '/'+str(len(LSOA_polys)), time.strftime("%H:%M:%S - %Y", time.localtime()))

    return points_in_polys

#### IMPORTS ####
import pandas as pd
import geopandas as gpd
#import openpyxl
import os
import  time
#import pdb

print('\nLoading data', time.strftime("%Y:%H:%M:%S", time.localtime()))

#### LIST OF POSTCODES TO Lat, Long COORDINATES ####
lookup_dir = r'Y:\TRSE Map Development\Postcodes to NaPTAN Points - England'
PT1 = pd.read_csv(os.path.join(lookup_dir, 'Postcodes PT 1.csv'))
PT2 = pd.read_csv(os.path.join(lookup_dir, 'Postcodes PT 2.csv'))
PT3 = pd.read_csv(os.path.join(lookup_dir, 'Postcodes PT 3.csv'))

postcode_loc_lookup = PT1.append(PT2,  ignore_index=True)
postcode_loc_lookup = postcode_loc_lookup.append(PT3, ignore_index=True)

# Postcode lookup created. Remove origional data to save space.
del(PT1)
del(PT2)
del(PT3)

# Filter columns
postcode_loc_lookup=postcode_loc_lookup[['pcds', 'lsoa11', 'lat', 'long']]

# Convert into a GeoDataFrame with relevant CRS 
postcode_loc_lookup = gpd.GeoDataFrame(data = postcode_loc_lookup,
                                       geometry=gpd.points_from_xy(x=postcode_loc_lookup['long'],
                                                                   y=postcode_loc_lookup['lat'],
                                                                   crs='EPSG:4326'))


# Lookup containing only postcode information and respective location
pcds_to_point = postcode_loc_lookup[['pcds', 'geometry']]
print('\npostcode lookups loaded')

#### LOAD LSOAs ####
# LSOA geometries
LSOA_dir=r'Y:\Data Strategy\GIS Shapefiles\LSOA'
LSOA_fn='Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.shp'
LSOAs=gpd.read_file(os.path.join(LSOA_dir, LSOA_fn))

# Select only Enlgish LSOAs
LSOAs = LSOAs[LSOAs['LSOA11CD'].str.contains('E')].copy()

print('\nLSOAs loaded')
#### Load Education Data #### 
education_path = r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data\EduBase Extract - 2016-0005414.csv'
education = pd.read_csv(education_path,
                        encoding = "ISO-8859-1")

'''
The education data comes with E&N cords (EPSG:27700) use .points_from_xy() specifying 
crs="EPSG:27700" to create point data, before then changing the crs to "EPSG:4326" to match
the crs for the rest of the analysis.
'''
education_gdf = gpd.GeoDataFrame(data = education, 
                                 geometry=gpd.points_from_xy(x=education['Easting'], 
                                                             y=education['Northing'], 
                                                             crs="EPSG:27700"))
print('\neducation_gdf loaded')
# Convert CRS to EPSG:4326 geometries to pull lat & long coords out
education_gdf = education_gdf.to_crs(crs='EPSG:4326')

# Add EPSG:4326 Lat & Long coordinates
education_gdf['lat'] = education_gdf['geometry'].x
education_gdf['long'] = education_gdf['geometry'].y

education_gdf['WGS_coords'] = gpd.points_from_xy(education_gdf['lat'],
                                                 education_gdf['long'],
                                                 crs = 'EPSG:4326')

education_gdf.set_geometry('WGS_coords',
                           inplace=True,
                           crs='EPSG:4326')

# Primary and secondary schools can be identified in the filed titled “Phase of Education”.
primary_education = education_gdf[education_gdf['PhaseOfEducation (name)'].str.contains('Primary|Middle Deemed Primary')].copy()
secondary_education = education_gdf[education_gdf['PhaseOfEducation (name)'].str.contains('Secondary|Middle Deemed Secondary')].copy()
further_education = education_gdf[education_gdf['PhaseOfEducation (name)'].str.contains('16 Plus')].copy()
print('\neducation has been split up')
# No longer need ueducation_gdf
del(education_gdf)


#### LOAD GPs INFORMATION ####
# GP surgeries
GP_path = r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data\epraccur.csv'
# Specify column headings.
colnames = ['Org_code', 'Name','National_grouping',
            'High_lvl_health_geography', 'AL1', 'AL2',
            'AL3', 'AL4', 'AL5', 'postcode', 'open_date',
            'close_date', 'status_code', 'Org_sub_type_code',
            'commissioner', 'Join_date', 'Leave_date',
            'Contact_num', 'Null1', 'Null2', 'Null3',
            'Amended Record indicator', 'Null4', 'Provider_Purchaser',
            'Null5', 'Perscribing_setting', 'Null6']

            
        
# Load GPs dataset
GPs = pd.read_csv(GP_path,
                  names = colnames)

# Only want GPs with an "Actvie" status code (opposed to Closed, Dormant, Proposed)
GPs = GPs[GPs['status_code'] == 'A']

# Using GP postcodes and our postcode_loc_lookup, we can join spatial information to the postocdes
GPs_pcds_locs = GPs[['postcode']].copy()

GPs_pcds_locs = GPs_pcds_locs.merge(how='left',
                                    right=postcode_loc_lookup[['pcds', 'geometry']],
                                    left_on = 'postcode',
                                    right_on = 'pcds')

# re-sepcify GPs_pcds_locs as a GeoDataFrame
GPs_pcds_locs = gpd.GeoDataFrame(data = GPs_pcds_locs,
                                 crs='EPSG:4326')

# The GPs_pcds_locs now contains None geometries. GPs list contains Welsh & Scottish
# GPs but our postcode_loc_lookup contains only English postcode. Thus, remove GPs
# from the list if their `pcds` values are nulls
GPs_pcds_locs = GPs_pcds_locs[GPs_pcds_locs['pcds'].isnull() == False].copy()
print('\nGPs have been loaded')

# Get a list of unique LSOA codes
LSOA_codes = list(LSOAs['LSOA11CD'].unique())
# Leave only English LSOAs remaining
LSOA_codes = [code for code in LSOA_codes if 'E' in code]
# Create empty lookup DataFrame to append amenity counts to.
LSOA_amenities = pd.DataFrame(data = LSOA_codes ,
                              columns = ['LSOA11CD'])

# Load NoRMS geometries & convert to centroids for station analysis
NoRMS_path = r'Y:\Data Strategy\GIS Shapefiles\NoRMS zones\_SS\TfN_Zones_Combined.shp'
NoRMS_zones = gpd.read_file(NoRMS_path)
print('\nNoRMS zones loaded')


# This NoRMS file is boundaries - convert polys to centroids
NoRMS_zones['centroids'] = NoRMS_zones['geometry'].centroid
# Drop origional geometry (polygon) columns
NoRMS_zones = NoRMS_zones.drop(columns='geometry')

# Format spatial data frames
NoRMS_centroids = NoRMS_zones['centroids'].copy()
del(NoRMS_zones)

LSOA_polys = LSOAs['geometry'].copy()

# Point data for amenities is Lat, Long (EPSG:4326). Thus, convert crs
LSOA_polys = LSOA_polys.to_crs('EPSG:4326')
NoRMS_centroids = NoRMS_centroids.to_crs('EPSG:4326')

# Also require information on Emplyment centres and town services access. As
# there is no spatial information on these areas (for now...) we will 
# approximate these categories from the TRSE working book. Specifically, the 
# numer of employment centres & town centre accessable within 15mins on PT.
amenities_dir = r'E:\TRSE PBA Tool (1)\TRSE Tool\lookups'
employment_fn = '501 Employment centres.ods'

# Load employment centres data
lsoa_emp_amenities = pd.read_excel(os.path.join(amenities_dir, 
                                                employment_fn),
                                   sheet_name = '2019_Revised',
                                   usecols=['LSOA_codes',
                                            '5000EmpPT15n'])
towns_fn = '508 Town centres.ods'
# Load Town centre access data
lsoa_town_amenities = pd.read_excel(os.path.join(amenities_dir,
                                                 towns_fn),
                                    sheet_name = "2019",
                                    usecols=['LSOA_codes',
                                             'TownPT15n'])


# Filter only the required data
#lsoa_emp_amenities = lsoa_emp_amenities[['LSOA_code', '5000EmpPT15n']]
#lsoa_town_amenities = lsoa_town_amenities[['LSOA_code', 'TownPT15n']]



print('\nData loading & processing complete', time.strftime("%Y:%H:%M:%S", time.localtime()), '\n\n')

###### Find NoRMS_centroids per LSOA ######
print('\nAnalysing NoRMS centroids')
NoRMS_in_LSOAs = points_in_areas(NoRMS_centroids, LSOA_polys, 2500)
# Append numer of NoRMS centroids per LSOA
LSOA_amenities['NoRMS_centroids'] = NoRMS_in_LSOAs



###### Find Supermarkets per LSOA ######
print('\nAnalysing Supermarkets')
# Load geometries 
supermarkets_path =  r'Y:\TRSE Map Development\Map Version 1\Supermarkets\SupermarketsGB.gpx'
supermarkets = gpd.read_file(supermarkets_path)
# Filter geometries
supermarket_centroids = supermarkets['geometry'].copy()
# Count points per LSOA
supermarket_in_LSOAs = points_in_areas(supermarket_centroids, LSOA_polys, 2500)
# Append numer of supermarkets per LSOA
LSOA_amenities['Supermarkets'] = supermarket_in_LSOAs



###### Find Hospitals per LSOA ######
print('\nAnalysing Hospitals')
# Load geometries
hospitals_path = r'Y:\TRSE Map Development\Map Version 1\Hospital Locations\HospitalsGB.gpx'
hospitals = gpd.read_file(hospitals_path)

# Filter out un-needed columns
hospital_centroids = hospitals['geometry']
# Count points per LSOA
hospitals_in_LSOAs = points_in_areas(hospital_centroids, LSOA_polys, 2500) 
# Add data on hospitals within LSOAs
LSOA_amenities['Hospitals'] = hospitals_in_LSOAs



###### Find Primary Schools per LSOA ######
print('\nAnalysing Primary Schools')
# Filter geometries
PS_centroids = primary_education['WGS_coords'].copy()
# Count points per LSOA
PS_in_LSOAs = points_in_areas(PS_centroids, LSOA_polys, 2500)
# Add data on primary schools within LSOAs
LSOA_amenities['PrimarySchools'] = PS_in_LSOAs


###### Find Secondary Schools per LSOA ######
print('\nAnalysing Secondary Schools')
# Filter geometries
SS_centroids = secondary_education['WGS_coords'].copy()
# Count points per LSOA
SS_in_LSOAs = points_in_areas(SS_centroids, LSOA_polys, 2500)
# Add data on secondary schools within LSOAs
LSOA_amenities['SecondarySchools'] = SS_in_LSOAs


###### Find Further Education per LSOA ######
print('\nAnalysing Further Education')
# Filter geometries
FE_centroids = further_education['WGS_coords'].copy()
# Count points per LSOA
FE_in_LSOAs = points_in_areas(FE_centroids, LSOA_polys, 2500)
# Add data on further education within LSOAs
LSOA_amenities['FurtherEducation'] = FE_in_LSOAs       


###### Find GPs per LSOA ######
print('\nAnalysing GPs')
# Filter geometries
GP_centroids = GPs_pcds_locs['geometry'].copy()

# Count GPs per LSOA
GPs_in_LSOAs = points_in_areas(GP_centroids, LSOA_polys, 2500)
# Add data on GPs within LSOAs
LSOA_amenities['GPs'] = GPs_in_LSOAs


# Add totals count for OTP filtering
LSOA_amenities['totals'] = LSOA_amenities.sum(axis = 'columns')

print('Finished - exporting data')
print(time.strftime("%Y:%H:%M:%S", time.localtime()))
LSOA_amenities.set_index('LSOA11CD', inplace=True)
LSOA_amenities.to_csv(r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data\LSOA_amenities.csv',
                      index=True)
