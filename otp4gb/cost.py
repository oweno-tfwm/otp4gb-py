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
from otp4gb import routing, util

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
    zone_centroids: gpd.GeoDataFrame, zone_column: str, settings: CostSettings
) -> list[CalculationParameters]:
    """Build a list of parameters for running `calculate_costs`.

    Parameters
    ----------
    zone_centroids : gpd.GeoDataFrame
        Positions of zones to calculate costs between.
    zone_column : str
        Name of the column containing the zone ID.
    settings : CostSettings
        Additional settings for calculating the costs.

    Returns
    -------
    list[CalculationParameters]
        Parameters to run `calculate_costs`.
    """
    LOG.info("Building cost calculation parameters")
    zone_centroids = zone_centroids.set_index(zone_column)["geometry"]

    params = []
    for o, d in itertools.product(zone_centroids.index, zone_centroids.index):
        params.append(
            CalculationParameters(
                server_url=settings.server_url,
                modes=settings.modes,
                datetime=settings.datetime,
                origin=routing.Place(
                    name=o, lon=zone_centroids.loc[o].y, lat=zone_centroids.loc[o].x
                ),
                destination=routing.Place(
                    name=d, lon=zone_centroids.loc[d].y, lat=zone_centroids.loc[d].x
                ),
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
    zone_column: str,
    settings: CostSettings,
    matrix_file: pathlib.Path,
    workers: int = 0,
) -> None:
    """Create cost matrix for all zone to zone pairs.

    Parameters
    ----------
    zone_centroids : gpd.GeoDataFrame
        Zones for the cost matrix.
    zone_column : str
        Name of the column containing the zone ID.
    settings : CostSettings
        Settings for calculating the costs.
    matrix_file : pathlib.Path
        Path to save the cost matrix to.
    workers : int, default 0
        Number of threads to create during calculations.
    """
    LOG.info("Calculating costs for %s with settings\n%s", matrix_file.name, settings)
    jobs = build_calculation_parameters(zone_centroids, zone_column, settings)

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


def cost_matrix_from_responses():
    # TODO Read responses jsonl file and build cost matrix without starting a new OTP server
    raise NotImplementedError("WIP: cost_matrix_from_responses")
