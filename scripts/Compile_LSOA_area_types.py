# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:05:08 2023

Script that takes to Govt. LSOA --> Urban/Rural classifications and creates an
area type lookup for the OTP tool to assess an appropriate maximum radial filter 
distance

@author: Signalis


Classification                                  | Classification Code
Urban major conurbation                         | A1
Urban minor conurbation                         | B1
Urban city and town                             | C1
Urban city and town in a sparse setting         | C2
Rural town and fringe                           | D1
Rural town and fringe in a sparse setting       | D2
Rural village and dispersed                     | E1
Rural village and dispersed in a sparse setting | E2

In General though:
    Urban -- [A1, B1, C1, C2]
    Rural -- [D1, D2, E1, E2]


"""

## Generic Imports ##
#import geopandas as gpd
import pandas as pd 
import os 


data_dir = r'C:\Users\Signalis\Documents\Repositories\otp4gb-py\Data'
# Move on dir up
os.chdir('..')

# Load LSOA data: 
LSOA_area_types = pd.read_excel(os.path.join(os.getcwd(), 'Data\Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.ods'),
                                sheet_name = 'LSOA11',
                                skiprows=[0,1])

'''
Thoughts: We don't need to be implementing a check for Urban LSOAs as they will
          use the specified radius. Rural LSOAs should have the maximum_radius
          increased. Say D1 & D2 add 25% to radius, E1 & E2 add 50% to max distance?
'''
# Rename columns
LSOA_area_types.rename(columns = {'Lower Super Output Area 2011 Code':'LSOA11CD',
                                  'Rural Urban Classification 2011 code':'RUC11CD',
                                  'Rural Urban Classification 2011 (10 fold)':'RUC11NM',
                                  'Rural Urban Classification 2011 (2 fold)':'RUC'},
                       inplace = True)

unique_RUC_codes = LSOA_area_types['RUC11CD'].unique()
unique_RUC_names = LSOA_area_types['RUC11NM'].unique()
print(unique_RUC_codes)
print(unique_RUC_names)
LSOA_area_types = LSOA_area_types.replace('\xa0', '', regex=True)
unique_RUC_codes = LSOA_area_types['RUC11CD'].unique()
unique_RUC_names = LSOA_area_types['RUC11NM'].unique()
print(unique_RUC_codes)
print(unique_RUC_names)

# Format required columns
LSOA_area_types = LSOA_area_types[['LSOA11CD', 'RUC11CD', 'RUC11NM', 'RUC']]

# Export the data.
LSOA_area_types.to_csv(os.path.join(os.getcwd(), 'Data', 'compiled_LSOA_area_types.csv'),
                       index = False)


