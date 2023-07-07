"""
Created on Tue Mar 28 10:37:33 2023

@author: Signalis

Script to create a lookup of already requested trips by OTP.

Sometimes, VMs shut down overnight, or large OTP runs will be split over
    numerous VM machines, leaving us with multiple response files we don't
    want to re-request.

    This script creates a single output cost-metrics matrix from a list
    of OTP responses files as pathlib.Path objects

Inputs:
    - OTP responses data (.jsonl)

Output:
    - Compiled cost-metrics matrix from all OTP responses provided
"""

# IMPORTS
# System imports
import os
import sys
from datetime import datetime
from pathlib import Path

# Add otp4gb dir to sys.path for local imports
sys.path.append(str(Path(os.getcwd()).parents[0]))
# Local imports
from otp4gb.cost import cost_matrix_from_responses, AggregationMethod

# CONSTANTS
# response paths should be a pathlib.Path object
RESPONSES_PATHS = [
    Path(r"E:\OTP_Processing\OTP outputs\TRSE OTP Related runs\First OTP run (signalis)\BUS_WALK_costs_20230608T0900.csv-response_data.jsonl"),
    Path(r"E:\OTP_Processing\OTP outputs\TNE_bus_test\costs\AM\BUS_WALK_costs_20230608T0900.csv-response_data.jsonl"),
]
# Output path for the compiled responses matrix
MATRIX_PATH = Path(r"C:\Users\Signalis\Desktop\temp_outputs\test_output_matrix.csv")

# SCRIPT
print("Script commenced: {}\nCompiling responses {}".format(
    datetime.now().strftime("%H:%M:%S"),
    RESPONSES_PATHS,
))

cost_matrix_from_responses(responses_files=RESPONSES_PATHS,
                           matrix_file=MATRIX_PATH,
                           aggregation_method=AggregationMethod)

print("Script finished: {}".format(datetime.now().strftime("%H:%M:%S")))
