# -*- coding: utf-8 -*-
"""
    Module for calculating the costs between OD pairs.
"""

##### IMPORTS #####
# Standard imports
import datetime
import enum
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
from otp4gb import routing, util, centroids#, config
#For now just specify maximum radius here
filter_radius = 1000  #metres - (Set to 0 if not needed)


##### CONSTANTS #####
LOG = logging.getLogger(__name__)

##### CLASSES #####
class CostSettings(NamedTuple):
    """Settings for the OTP routing."""

    server_url: str
    modes: list[routing.Mode]
    datetime: datetime.datetime
    arrive_by: bool = False
    search_window_seconds: Optional[int] = None
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
    searchWindow: Optional[int] = None
    wheelchair: bool = False
    max_walk_distance: float = 1000


class CostResults(pydantic.BaseModel):
    """Cost results from OTP saved to responses JSON file."""

    origin: routing.Place
    destination: routing.Place
    plan: Optional[routing.Plan]
    error: Optional[routing.RoutePlanError]
    request_url: str


class AggregationMethod(enum.StrEnum):
    """Method to use when aggregating itineraries."""

    MEAN = enum.auto()
    MEDIAN = enum.auto()


class GeneralisedCostFactors(pydantic.BaseModel):
    """Parameters for calculating the generalised cost."""

    wait_time: float
    transfer_number: float
    walk_time: float
    transit_time: float
    walk_distance: float
    transit_distance: float


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
    
    
    # Create GDF of all OD pairs & work out distance between them.
    if filter_radius != 0:

        origins = []
        destinations = []
        OD_pairs = []
        for o, d in itertools.product(zone_centroids.index, zone_centroids.index):
            if o == d:
                continue
            origins.append(o)
            destinations.append(d)
            OD_pairs.append('_'.join((str(o), str(d))))
        
        # DF of all OD pairs
        OD_pairs = gpd.GeoDataFrame(data={"Origins": origins,
                                          "Destinations": destinations,
                                          "OD_pairs": OD_pairs})
        
        # Test print statements - checking that zone_centroids indeed does contain 'geometry'
        print("\nPrint statement of zone_centroids data types:\n", zone_centroids.dtypes)
        print("\n","geometry" in zone_centroids, "Boolean check - Is 'geometry' %in% zone_centroids?\n\n")
        
        # Join on zone centroid data - Origins
        OD_pairs.merge(how="left",
                       right=zone_centroids.geometry,
                       left_on="Origins",
                       right_on=zone_centroids.index)

        OD_pairs.rename(columns={"geometry":"Origin_centroids"},
                        inplace=True)

        # Join on zone centroids - Destinations
        OD_pairs.merge(how="left",
                       right=zone_centroids.geometry,
                       left_on="Destinations",
                       right_on=zone_centroids.index)

        OD_pairs.rename(columns={"geometry":"Destination_centroids"},
                        inplace=True)

        # Work out distance between all OD pairs
        OD_pairs['distances'] = OD_pairs.apply(
            lambda row: row["Origin_centroids"].distance(row["Destination_centroids"]),
            axis=1)

    params = []
    for o, d in itertools.product(zone_centroids.index, zone_centroids.index):
        if o == d:
            continue

        # Check if od journey exceeds maximum distance
        if filter_radius != 0:
            od_code = '_'.join(str(o), str(d))
            od_distance = OD_pairs[OD_pairs["OD_pairs"] == od_code]["distances"].values[0]
            if od_distance > filter_radius:
                continue

        params.append(
            CalculationParameters(
                server_url=settings.server_url,
                modes=[str(m) for m in settings.modes],
                datetime=settings.datetime,
                origin=row_to_place(zone_centroids.loc[o]),
                destination=row_to_place(zone_centroids.loc[d]),
                arrive_by=settings.arrive_by,
                searchWindow=settings.search_window_seconds,
                wheelchair=settings.wheelchair,
                max_walk_distance=settings.max_walk_distance,
            )
        )

    return params


def calculate_costs(
    parameters: CalculationParameters,
    response_file: io.TextIOWrapper,
    lock: threading.Lock,
    generalised_cost_parameters: GeneralisedCostFactors,
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
    generalised_cost_parameters : GeneralisedCostParameters
        Factors and other parameters for calculating the
        generalised cost.

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
        searchWindow=parameters.searchWindow,
        wheelchair=parameters.wheelchair,
        maxWalkDistance=parameters.max_walk_distance,
    )
    url, result = routing.get_route_itineraries(parameters.server_url, rp_params)

    if result.plan is not None:
        result.plan.date = result.plan.date.astimezone(parameters.datetime.tzinfo)

        for itinerary in result.plan.itineraries:
            itinerary.startTime = itinerary.startTime.astimezone(
                parameters.datetime.tzinfo
            )
            itinerary.endTime = itinerary.endTime.astimezone(parameters.datetime.tzinfo)

            # Update itineraries with generalised cost
            generalised_cost(itinerary, generalised_cost_parameters)

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


def generalised_cost(
    itinerary: routing.Itinerary, factors: GeneralisedCostFactors
) -> None:
    """Calculate the generalised cost and update value in `itinerary`.

    Times given in `itinerary` shoul be in seconds and are converted
    to minutes for the calculation.
    Distances given in `itinerary` should be in metres and are converted
    to km for the calculation.

    Parameters
    ----------
    itinerary : routing.Itinerary
        Route itinerary.
    factors : GeneralisedCostParameters
        Factors for generalised cost calculation.
    """
    wait_time = itinerary.waitingTime * factors.wait_time
    transfer_penalty = itinerary.transfers * factors.transfer_number
    walk_time = itinerary.walkTime / 60 * factors.walk_time
    transit_time = itinerary.transitTime / 60 * factors.transit_time
    walk_distance = itinerary.walkDistance / 1000 * factors.walk_distance

    transit_distance = sum(
        l.distance for l in itinerary.legs if l.mode in routing.Mode.transit_modes()
    )
    transit_distance = transit_distance / 1000 * factors.transit_distance

    itinerary.generalised_cost = (
        wait_time
        + transfer_penalty
        + walk_time
        + transit_time
        + walk_distance
        + transit_distance
    )


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
    aggregation: AggregationMethod,
) -> None:
    matrix = pd.DataFrame(data)

    metrics_file = matrix_file.with_name(matrix_file.stem + "-metrics.csv")
    matrix.to_csv(metrics_file, index=False)
    LOG.info("Written cost metrics to %s", metrics_file)

    try:
        LOG.info(
            "Minimum departure time from all responses {:%x %X %Z}".format(
                matrix["min_startTime"].min()
            )
        )
        LOG.info(
            "Maximum arrival time from all responses {:%x %X %Z}".format(
                matrix["max_endTime"].max()
            )
        )
    except KeyError as error:
        LOG.warning("Start / end times unavailable in matrix: %s", error)

    gen_cost_column = f"{aggregation}_generalised_cost"

    if gen_cost_column not in matrix.columns:
        LOG.error(
            "Generalised cost column (%s) not found in "
            "matrix metrics, so skipping matrix creation",
            gen_cost_column,
        )
        return

    # Pivot generalised cost
    gen_cost = matrix.pivot(
        index="origin_id", columns="destination_id", values=gen_cost_column
    )
    gen_cost.to_csv(matrix_file)
    LOG.info("Written %s generalised cost matrix to %s", aggregation, matrix_file)


def build_cost_matrix(
    zone_centroids: gpd.GeoDataFrame,
    centroids_columns: centroids.ZoneCentroidColumns,
    settings: CostSettings,
    matrix_file: pathlib.Path,
    generalised_cost_parameters: GeneralisedCostFactors,
    aggregation_method: AggregationMethod,
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
    generalised_cost_parameters : GeneralisedCostParameters
        Factors and other parameters for calculating the
        generalised cost.
    aggregation_method : AggregationMethod
        Aggregation method used for generalised cost matrix.
    workers : int, default 0
        Number of threads to create during calculations.
    """
    LOG.info("Calculating costs for %s with settings\n%s", matrix_file.name, settings)
    jobs = build_calculation_parameters(zone_centroids, centroids_columns, settings)

    lock = threading.Lock()
    response_file = matrix_file.with_name(matrix_file.name + "-response_data.jsonl")
    with open(response_file, "wt") as responses:
        iterator = util.multithread_function(
            workers,
            calculate_costs,
            jobs,
            dict(
                response_file=responses,
                lock=lock,
                generalised_cost_parameters=generalised_cost_parameters.copy(),
            ),
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
    _write_matrix_files(matrix_data, matrix_file, aggregation_method)


def cost_matrix_from_responses(
    responses_file: pathlib.Path,
    matrix_file: pathlib.Path,
    aggregation_method: AggregationMethod,
) -> None:
    """Create cost matrix CSV from responses JSON lines file.

    Parameters
    ----------
    responses_file : pathlib.Path
        Path to JSON lines file containing `CostResults`.
    matrix_file : pathlib.Path
        Path to CSV file to output cost metrics to.
    aggregation_method : AggregationMethod
        Aggregation method used for generalised cost matrix.
    """
    matrix_data = []
    with open(responses_file, "rt") as responses:
        for line in tqdm.tqdm(
            responses, desc="Calculating cost matrix", dynamic_ncols=True
        ):
            results = CostResults.parse_raw(line)
            # TODO(MB) Recalculate generalised cost if new parameters are provided
            matrix_data.append(_matrix_costs(results))

    _write_matrix_files(matrix_data, matrix_file, aggregation_method)
