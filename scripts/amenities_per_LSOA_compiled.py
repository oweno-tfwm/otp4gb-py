# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:44:41 2023

@author: Signalis
"""

#### IMPORTS ####
import pandas as pd
#import geopandas as gpd
import openpyxl
import os


amenities_dir = r'E:\TRSE PBA Tool (1)\TRSE Tool\lookups'
amenities_fn = 'Amenity_compiled.xlsx'

EMP_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                             sheet_name='EMP')

PS_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='PS')

SS_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='SS')

FE_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='FE')

GPs_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='GPs')

Hosp_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='Hosp')

Town_compiled = pd.read_excel(os.path.join(amenities_dir, amenities_fn),
                            sheet_name='Town')



# List of column names that contain "n" as the last character of string
EMP_cols = [title for title in EMP_compiled.columns if ('n' in title[-1])]
PS_cols = [title for title in PS_compiled.columns if ('n' in title[-1])]
SS_cols = [title for title in SS_compiled.columns if ('n' in title[-1])]
FE_cols = [title for title in FE_compiled.columns if ('n' in title[-1])]
GPs_cols = [title for title in GPs_compiled.columns if ('n' in title[-1])]
Hosp_cols = [title for title in Hosp_compiled.columns if ('n' in title[-1])]
Town_cols = [title for title in Town_compiled.columns if ('n' in title[-1])]
