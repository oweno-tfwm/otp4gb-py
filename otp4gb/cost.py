# -*- coding: utf-8 -*-
"""
    Module for calculating the costs between OD pairs.
"""

##### IMPORTS #####
# Standard imports
import datetime
import io
import itertools
import logging
import pathlib
import threading
from typing import Any, NamedTuple, Optional

# Third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pydantic
import tqdm

# Local imports
from otp4gb import routing, util, centroids

##### CONSTANTS #####
LOG = logging.getLogger(__name__)

##### CLASSES #####
class CostSettings(NamedTuple):
    """Settings for the OTP routing."""

    server_url: str
    modes: list[str]
    datetime: datetime.datetime
    arrive_by: bool = False
    wheelchair: bool = False
    max_walk_distance: float = 1000


class CalculationParameters(NamedTuple):
    """Parameters for `calculate_costs` function."""

    server_url: str
    modes: list[str]
    datetime: datetime.datetime
    origin: routing.Place
    destination: routing.Place
    arrive_by: bool = False
    wheelchair: bool = False
    max_walk_distance: float = 1000


class CostResults(pydantic.BaseModel):
    """Cost results from OTP saved to responses JSON file."""

    origin: routing.Place
    destination: routing.Place
    plan: Optional[routing.Plan]
    error: Optional[routing.RoutePlanError]
    request_url: str


##### FUNCTIONS #####
def build_calculation_parameters(
    zone_centroids: gpd.GeoDataFrame,
    centroids_columns: centroids.ZoneCentroidColumns,
    settings: CostSettings,
) -> list[CalculationParameters]:
    """Build a list of parameters for running `calculate_costs`.

    Parameters
    ----------
    zone_centroids : gpd.GeoDataFrame
        Positions of zones to calculate costs between.
    centroids_columns : centroids.ZoneCentroidColumns
        Names of the columns in `zone_centroids`.
    settings : CostSettings
        Additional settings for calculating the costs.

    Returns
    -------
    list[CalculationParameters]
        Parameters to run `calculate_costs`.
    """

    def row_to_place(row: pd.Series) -> routing.Place:
        return routing.Place(
            name=row[centroids_columns.name],
            id=row.name,
            zone_system=row[centroids_columns.system],
            lon=row["geometry"].y,
            lat=row["geometry"].x,
        )

    LOG.info("Building cost calculation parameters")
    zone_centroids = zone_centroids.set_index(centroids_columns.id)[
        [centroids_columns.name, centroids_columns.system, "geometry"]
    ]

    params = []
    for o, d in itertools.product(zone_centroids.index, zone_centroids.index):
        params.append(
            CalculationParameters(
                server_url=settings.server_url,
                modes=settings.modes,
                datetime=settings.datetime,
                origin=row_to_place(zone_centroids.loc[o]),
                destination=row_to_place(zone_centroids.loc[d]),
                arrive_by=settings.arrive_by,
                wheelchair=settings.wheelchair,
                max_walk_distance=settings.max_walk_distance,
            )
        )

    return params


def calculate_costs(
    parameters: CalculationParameters,
    response_file: io.TextIOWrapper,
    lock: threading.Lock,
) -> dict[str, Any]:
    """Calculate cost between 2 zones using OTP.

    Parameters
    ----------
    parameters : CalculationParameters
        Parameters for OTP.
    response_file : io.TextIOWrapper
        File to save the JSON response too.
    lock : threading.Lock
        Lock object to avoid race conditions when writing
        to `response_file`.

    Returns
    -------
    dict[str, Any]
        Cost statistics for the cost matrix.
    """
    rp_params = routing.RoutePlanParameters(
        fromPlace=f"{parameters.origin.lon}, {parameters.origin.lat}",
        toPlace=f"{parameters.destination.lon}, {parameters.destination.lat}",
        date=parameters.datetime.date(),
        time=parameters.datetime.time(),
        mode=parameters.modes,
        arriveBy=parameters.arrive_by,
        wheelchair=parameters.wheelchair,
        maxWalkDistance=parameters.max_walk_distance,
    )
    url, result = routing.get_route_itineraries(parameters.server_url, rp_params)

    cost_res = CostResults(
        origin=parameters.origin,
        destination=parameters.destination,
        plan=result.plan,
        error=result.error,
        request_url=url,
    )
    with lock:
        response_file.write(cost_res.json() + "\n")

    return _matrix_costs(cost_res)


# TODO(MB) Additional parameter for generalised cost calculation parameters
# and return generalised cost as a separate float
def _matrix_costs(result: CostResults) -> dict:
    matrix_values = {
        "origin": result.origin.name,
        "destination": result.destination.name,
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
        "generalizedCost",
        "transfers",
    ]
    for s in stats:
        values = []
        for it in result.plan.itineraries:
            values.append(getattr(it, s, np.nan))

        matrix_values[f"mean_{s}"] = np.nanmean(values)

        if matrix_values["number_itineraries"] > 1:
            matrix_values[f"median_{s}"] = np.nanmedian(values)
            matrix_values[f"min_{s}"] = np.nanmin(values)
            matrix_values[f"max_{s}"] = np.nanmax(values)
            matrix_values[f"num_nans_{s}"] = np.sum(np.isnan(values))

    return matrix_values


def build_cost_matrix(
    zone_centroids: gpd.GeoDataFrame,
    centroids_columns: centroids.ZoneCentroidColumns,
    settings: CostSettings,
    matrix_file: pathlib.Path,
    workers: int = 0,
) -> None:
    """Create cost matrix for all zone to zone pairs.

    Parameters
    ----------
    zone_centroids : gpd.GeoDataFrame
        Zones for the cost matrix.
    centroids_columns : centroids.ZoneCentroidColumns
        Names of the columns in `zone_centroids`.
    settings : CostSettings
        Settings for calculating the costs.
    matrix_file : pathlib.Path
        Path to save the cost matrix to.
    workers : int, default 0
        Number of threads to create during calculations.
    """
    LOG.info("Calculating costs for %s with settings\n%s", matrix_file.name, settings)
    jobs = build_calculation_parameters(zone_centroids, centroids_columns, settings)

    lock = threading.Lock()
    response_file = matrix_file.with_name(matrix_file.name + "-response_data.jsonl")
    with open(response_file, "wt") as responses:
        iterator = util.multithread_function(
            workers, calculate_costs, jobs, dict(response_file=responses, lock=lock)
        )

        matrix_data = []
        for res in tqdm.tqdm(
            iterator,
            total=len(jobs),
            desc="Calculating costs",
            dynamic_ncols=True,
            smoothing=0,
        ):
            matrix_data.append(res)

    LOG.info("Written responses to %s", response_file)

    matrix = pd.DataFrame(matrix_data)
    matrix.to_csv(matrix_file, index=False)
    LOG.info("Written cost matrix %s", matrix_file)


def cost_matrix_from_responses(
    responses_file: pathlib.Path, matrix_file: pathlib.Path
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
        for line in tqdm.tqdm(
            responses, desc="Calculating cost matrix", dynamic_ncols=True
        ):
            results = CostResults.parse_raw(line)
            matrix_data.append(_matrix_costs(results))

    matrix = pd.DataFrame(matrix_data)
    matrix.to_csv(matrix_file, index=False)
    LOG.info("Written cost matrix to %s", matrix_file)
