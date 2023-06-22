# -*- coding: utf-8 -*-
"""
Created on Mon June 19 11:14:17 2023

@author: Signalis

Script to export a cost matrix from a partial json lines response file.

Typically, OTP creates this once all requests have been sent. However, some times
    a machine may crash mid run, or be turned off overnight, meaning any progress
    would be lost.

This script can make use of the these partial response files, exporting a cost matrix
    for trips requested to far. Using this, one could determine the trips already
    requested by this run, and add them as `irrelevant_destination`'s when restarting
    the run.

"""

#### IMPORTS ####
import enum
import pathlib
import pandas as pd
import tqdm
import pydantic
import numpy as np
from pathlib import Path
import mmap
import datetime as datetime

from typing import Optional
from otp4gb import routing

#### USER PARAMS ####

# (Partially) complete json lines response file path
response_path = r"E:\otp4gb-py\trse_walk_test\costs\PM\WALK_costs_20230608T1900.csv-response_data.jsonl"

# File path for extracted cost matrix
output_path = r"E:\otp4gb-py\trse_walk_test\costs\PM\test_metrics.csv"


#### Functions
class AggregationMethod(enum.StrEnum):
    """Method to use when aggregating itineraries."""

    MEAN = enum.auto()
    MEDIAN = enum.auto()


class CostResults(pydantic.BaseModel):
    """Cost results from OTP saved to responses JSON file."""

    origin: routing.Place
    destination: routing.Place
    plan: Optional[routing.Plan]
    error: Optional[routing.RoutePlanError]
    request_url: str


def _matrix_costs(result: CostResults) -> dict:
    matrix_values: dict[str, str | int | float | datetime.datetime] = {
        "origin": result.origin.name,
        "destination": result.destination.name,
        "origin_id": result.origin.id,
        "destination_id": result.destination.id,
        "origin_zone_system": result.origin.zone_system,
        "destination_zone_system": result.destination.zone_system,
    }

    if result.plan is None:
        return matrix_values

    matrix_values["number_itineraries"] = len(result.plan.itineraries)
    if matrix_values["number_itineraries"] == 0:
        return matrix_values

    stats = [
        "duration",
        "walkTime",
        "transitTime",
        "waitingTime",
        "walkDistance",
        "otp_generalised_cost",
        "transfers",
        "generalised_cost",
    ]
    for s in stats:
        values = []
        for it in result.plan.itineraries:
            val = getattr(it, s)
            # Set value to NaN if it doesn't exist or isn't set
            if val is None:
                val = np.nan
            values.append(val)

        matrix_values[f"mean_{s}"] = np.nanmean(values)

        if matrix_values["number_itineraries"] > 1:
            matrix_values[f"median_{s}"] = np.nanmedian(values)
            matrix_values[f"min_{s}"] = np.nanmin(values)
            matrix_values[f"max_{s}"] = np.nanmax(values)
            matrix_values[f"num_nans_{s}"] = np.sum(np.isnan(values))

    matrix_values["min_startTime"] = min(i.startTime for i in result.plan.itineraries)
    matrix_values["max_startTime"] = max(i.startTime for i in result.plan.itineraries)
    matrix_values["min_endTime"] = min(i.endTime for i in result.plan.itineraries)
    matrix_values["max_endTime"] = max(i.endTime for i in result.plan.itineraries)

    return matrix_values


def _write_matrix_files(
        data: list[dict[str, float]],
        matrix_file: pathlib.Path,
) -> None:
    matrix = pd.DataFrame(data)

    if type(matrix_file) == str:
        matrix_file = Path(matrix_file)

    metrics_file = matrix_file.with_name(matrix_file.stem + "-metrics.csv")
    matrix.to_csv(metrics_file, index=False)
    print("Written cost metrics to {}\nThis can be used to create a lookup of trips requested".format(metrics_file))


def mapcount(filename):
    """
    Function to quickly determine response file length
    https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    """
    with open(filename, "r+") as f:
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline
        while readline():
            lines += 1
        return lines


def cost_matrix_from_responses(
        responses_file: pathlib.Path,
        matrix_file: pathlib.Path,
) -> None:
    """Create cost matrix CSV from responses JSON lines file.

    Parameters
    ----------
    responses_file : pathlib.Path
        Path to JSON lines file containing `CostResults`.
    matrix_file : pathlib.Path
        Path to CSV file to output cost metrics to.
    """
    matrix_data = []
    with open(responses_file, "rt") as responses:

        print("Determining response file length")
        # Get length of response file & report to user
        response_length = mapcount(responses_file)
        print("Evaluating {} response lines. This may take some time...".format(response_length))

        for line in tqdm.tqdm(
                responses, desc="Calculating cost matrix for", dynamic_ncols=True
        ):
            results = CostResults.parse_raw(line)
            # TODO(MB) Recalculate generalised cost if new parameters are provided
            matrix_data.append(_matrix_costs(results))

    _write_matrix_files(matrix_data, matrix_file)


# Export cost metrics from partial response file
cost_matrix_from_responses(Path(response_path), Path(output_path))
