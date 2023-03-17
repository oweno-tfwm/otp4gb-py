# -*- coding: utf-8 -*-
"""
    Author: JamesHulse (10/3/23)

    Script to compile a LSOA .csv  lookup file that matches LSOAs by LSOA11CD (LSOA Code) to the
    number of amenities within an LSOA.

    i.e:
    LSOA11CD | Hostpials | Supermarkets | Primary Education | Secondary Education
    E061002 |    0       |     1        |      1            |         0

    Inputs:
       - List of LSOA Boundaries (Polys/MultiPolys)
       - List of NoRMS centroids (Points. approximation for stations - used for JT calcs)
       - List of Amenity locations (Points)

    Outputs:
       - Compiled LSOA lookup with number of respective amenities in each LSOA appended.

"""

#### IMPORTS ####
import pandas as pd
import geopandas as gpd
import os

#### FUNCTIONS ####

# Function that returns address strings for addresses stored over multiple 
# DataFrame columns
def rows_to_addresses(DF, addr_cols):
    if type(addr_cols) != list:
        return"Error: `addr_cols` must be  list of int"
    if type(DF) != pd.core.frame.DataFrame:
        return"Error: `DF` must be a valid Pandas DataFrame"
        
        
    # iterate over the DF
    addresses = []
    for i in range(len(DF)):
        series = DF.iloc[i, addr_cols]
        
        # obtain address from DF row using address columns
        addr = [] 
        for line in series:
            if type(line) != str:
                # No address entered
                continue
            else:
                addr.append(line)
        
        addr_line = ', '.join([AL for AL in addr])
        addresses.append(addr_line)
    
    return addresses
    

#### LOAD DATA ####

# LSOA geometries
LSOA_dir=r'Y:\Data Strategy\GIS Shapefiles\LSOA'
LSOA_fn='Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.shp'
LSOAs=gpd.read_file(os.path.join(LSOA_dir, LSOA_fn))
# Get unqiue zone codes for lookup table
LSOA_codes = LSOAs['LSOA11CD'].unique()

# NoRMS geometries
NoRMS_dir=r'Y:\Data Strategy\GIS Shapefiles\NoRMS zones\_SS'
NoRMS_fn='TfN_Zones_Combined.shp'
NoRMS_zones = gpd.read_file(os.path.join(NoRMS_dir, NoRMS_fn))

# This NoRMS file is boundaries - apply .centroids() to convert polys to centroids
NoRMS_zones['centroids'] = NoRMS_zones['geometry'].centroid

## LOAD AMENITIES ##

# Education
data_dir = r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data'
education_fn = 'EduBase Extract - 2016-0005414.csv'
education = pd.read_csv(os.path.join(data_dir, education_fn),
                        encoding = "ISO-8859-1")

# Primary and secondary schools can be identified in the filed titled “Phase of Education”.
primary_education = education[education['PhaseOfEducation (name)'].str.contains('Primary|Middle Deemed Primary')].copy()
secondary_education = education[education['PhaseOfEducation (name)'].str.contains('Secondary|Middle Deemed Secondary')].copy()
further_education = education[education['PhaseOfEducation (name)'].str.contains('16 Plus')].copy()


# Hospitals
hospitals_dir = r'Y:\TRSE Map Development\Map Version 1\Hospital Locations'
hospitals_fn = 'HospitalsGB.gpx'
hospitals = gpd.read_file(os.path.join(hospitals_dir, hospitals_fn))
# Filter out un-needed columns
hospitals = hospitals[['name', 'desc', 'link1_href', 'geometry']]

# Supermarkets 
supermarkets_dir = r'Y:\TRSE Map Development\Map Version 1\Supermarkets'
supermarkets_fn = 'SupermarketsGB.gpx'
supermarkets = gpd.read_file(os.path.join(supermarkets_dir, supermarkets_fn))
# Filter out un-needed columns
supermarkets = supermarkets[['name', 'desc', 'link1_href', 'geometry']]

# GP surgeries
GPs_dir = r'Y:\TRSE Map Development\Map Version 1\GP Surgery locations'
GPs_fn = 'GP surgery locations - original data.csv'

GP_path = r'C:\Users\Signalis\Downloads\epraccur.csv'
# Specify column headings.
colnames = ['Org_code',
            'Name',
            'National_grouping',
            'High_lvl_health_geography',
            'AL1',
            'AL2',
            'AL3',
            'AL4',
            'AL5',
            'postcode',
            'open_date',
            'close_date',
            'status_code',
            'Org_sub_type_code',
            'commissioner',
            'Join_date',
            'Leave_date',
            'Contact_num',
            'Null1',
            'Null2',
            'Null3',
            'Amended Record indicator',
            'Null4',
            'Provider_Purchaser',
            'Null5',
            'Perscribing_setting',
            'Null6']

            
        
# Load GPs dataset
GPs = pd.read_csv(GP_path,
                  names = colnames)

# Import Nominatim geolocator
from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")


# Get address lines
GP_addresses = rows_to_addresses(GPs, [1, 4, 5, 6, 7, 8, 9])
GP_postcodes = rows_to_addresses(GPs, [9])

# Obtain Lat, Long coordinates for addresses/postcodes captured
counter = 0
locations = []
bad_locations = []
print('Geo-coding postcode location information for GPs')

# Nominatim API has usage limit of 1 Request/Second. Import time for time.sleep
import time

for location, postcode in zip(GP_addresses, GP_postcodes):
    
    if len(postcode) < 3: 
        # if postcode len < 3, a postcode has not been input and cannot be
        # geo-coded
        continue
    
    loc = geolocator.geocode(postcode)
    time.sleep(1)
    try:
        locations.append((loc.latitude, loc.longitude))

        
    except:
        #print(postcode, 'Could not be geo-coded. Skipping this location.')
        bad_locations.append((postcode, location))
        continue
    
    # Progress checker.
    counter += 1
    if counter%250 == 0: # No remainder
        print(str(counter)+ '  / ' + str(len(GP_addresses)))

    

education={'PS':[],
           'SS':[],
           'FE':[]}

edu_types = [primary_education,
             secondary_education,
             further_education]
edu_type_str = ['PS', 'SS', 'FE']
locations_d={'PS':[],
           'SS':[],
           'FE':[]}
bad_locations_d={'PS':[],
           'SS':[],
           'FE':[]}
counter= 0

print('Geo-coding postcode location information for Education')

for edu_type in edu_types:
    edu_str = edu_type_str[counter]
    print('\n',  edu_str, 'containing:', len(edu_type), 'items.')
    counter += 1
    type_addresses = rows_to_addresses(edu_type, [12, 13, 14, 15, 16, 17])
    type_postcodes = rows_to_addresses(edu_type, [17])
    
    i = 0 
    for location, postcode in zip(type_addresses, type_postcodes):
        
        if len(postcode) < 3:
            # if postcode len < 3, a postcode has not been input and cannot be
            # geo-coded
            continue
        
        loc = geolocator.geocode(postcode)
        time.sleep(1)
        
        try:
            locations_d[edu_type_str].append((loc.latitude, loc.longitude))

            
        except:
            print(postcode, 'Could not be geo-coded. Skipping this location.')
            bad_locations[edu_type_str].append((postcode, location))
            continue
        
        i += 1
        if i%250 == 0:
            print(edu_type_str+ ' : '+ str(i) + ' / ' +str(len(edu_type)))
      
    
    
    