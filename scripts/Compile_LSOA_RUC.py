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

In general though:
    Urban -- [A1, B1, C1, C2]
    Rural -- [D1, D2, E1, E2]


"""

#### IMPORTS ####
# System imports
import os 

# 3rd party imports
import pandas as pd 

#### CONSTANTS ####
# Specify data directory & filename for Rural Urban Classifications (RUC) file
DATA_DIR = r"E:\otp4gb-py\Data"
DATA_FILENAME = "Rural_Urban_Classification_2011_lookup_tables_for_small_area_geographies.xlsx"
# Excel doc sheet name
SHEET_NAME = "LSOA11"

# Filename for compiled RUC lookup file
SAVE_FILENAME = "compiled_LSOA_area_types.csv"

#### SCRIPT ####
# Load LSOA data: 
lsoa_area_types = pd.read_excel(os.path.join(DATA_DIR, DATA_FILENAME),
                                sheet_name = SHEET_NAME,
                                skiprows=[0,1])

# Rename columns for lookup format
lsoa_area_types.rename(columns = {'Lower Super Output Area 2011 Code':'LSOA11CD',
                                  'Rural Urban Classification 2011 code':'RUC11CD',
                                  'Rural Urban Classification 2011 (10 fold)':'RUC11NM',
                                  'Rural Urban Classification 2011 (2 fold)':'ruc'},
                       inplace = True)

unique_ruc_codes = lsoa_area_types['RUC11CD'].unique()
unique_ruc_names = lsoa_area_types['RUC11NM'].unique()
print(unique_ruc_codes)
print(unique_ruc_names)

# As seen above, sheet contains newlines \xa0 characters which affect parsing, 
#  change these using regex
lsoa_area_types = lsoa_area_types.replace('\xa0', '', regex=True)
unique_ruc_codes = lsoa_area_types['RUC11CD'].unique()
unique_ruc_names = lsoa_area_types['RUC11NM'].unique()
print(unique_ruc_codes)
print(unique_ruc_names)

# Format required columns
lsoa_area_types = lsoa_area_types[['LSOA11CD', 'RUC11CD', 'RUC11NM', 'ruc']]

# Export the data.
lsoa_area_types.to_csv(os.path.join(DATA_DIR, SAVE_FILENAME),
                       index = False)
