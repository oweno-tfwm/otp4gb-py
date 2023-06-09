# -*- coding: utf-8 -*-
"""
    Module for calculating the costs between OD pairs.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
import datetime
import enum
import io
import itertools
import logging
import pathlib
import threading
import os
from typing import Any, Iterator, NamedTuple, Optional, Sequence

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
CROWFLY_DISTANCE_CRS = "EPSG:27700"
RUC_WEIGHTS = {
    "A1": 1,
    "B1": 1,
    "C1": 1,
    "C2": 1,
    "D1": 1.25,
    "D2": 1.25,
    "E1": 1.5,
    "E2": 1.5,
}


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
    crowfly_max_distance: Optional[float] = None


class CalculationParameters(pydantic.BaseModel):
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
    crowfly_max_distance: Optional[float] = None


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


@dataclasses.dataclass
class RUCLookup:
    """Path to RUC lookup file and names of columns."""

    path: pathlib.Path
    id_column: str = "zone_id"
    ruc_column: str = "ruc"


@dataclasses.dataclass
class IrrelevantDestinations:
    """Path to file containing irrelevant destinations, and column name."""

    path: pathlib.Path
    zone_column: str = "zone_id"


##### FUNCTIONS #####
def _to_crs(data: gpd.GeoDataFrame, crs: str, name: str) -> gpd.GeoDataFrame:
    """Convert `data` to `crs` and raise error for invalid geometries."""
    original_crs = data.crs
    invalid_before: pd.Series = ~data.is_valid

    if invalid_before.any():
        LOG.warning(
            "%s (%s%%) invalid features in %s data before CRS conversion",
            invalid_before.sum(),
            invalid_before.sum() / len(data) * 100,
            name,
        )

    data = data.to_crs(crs)

    invalid_after: pd.Series = ~data.is_valid
    if not invalid_before.equals(invalid_after):
        raise ValueError(
            f"{invalid_after.sum()} ({invalid_after.sum()/len(data):.0%}) "
            f"invalid features after converting {name} from {original_crs} to {crs} "
        )

    return data


def _calculate_distance_matrix(
    origins: gpd.GeoDataFrame, destinations: gpd.GeoDataFrame, crs: str
) -> pd.DataFrame:
    """Calculate distances between all `origins` and `destinations`.

    Geometries are converted to `crs` before calculating distance.

    Parameters
    ----------
    origins, destinations : gpd.GeoDataFrame
        Points for calculating distances between, should
        have the same index of zone IDs.
    crs: str
        Name of CRS to convert to before calculating
        distances e.g. 'EPSG:27700'.

    Raises
    ------
    ValueError
        If the CRS conversion causes invalid features.
    """
    distances = pd.DataFrame(
        {
            "origin": np.repeat(origins.index, len(destinations)),
            "destination": np.tile(destinations.index, len(origins)),
        }
    )

    for name, data in (("origin", origins), ("destination", destinations)):
        data = _to_crs(data, crs, f"{name} centroids")

        distances = distances.merge(
            data["geometry"],
            left_on=name,
            how="left",
            validate="m:1",
            right_index=True,
        )
        distances.rename(columns={"geometry": f"{name}_centroid"}, inplace=True)

    distances.loc[:, "distance"] = gpd.GeoSeries(distances["origin_centroid"]).distance(
        gpd.GeoSeries(distances["destination_centroid"])
    )

    return distances.set_index(["origin", "destination"])["distance"]


def _summarise_list(values: Sequence | np.ndarray, max_values: int = 10) -> str:
    """Create string of first few values."""
    message = ", ".join(str(i) for i in values[:max_values])

    if len(values) > max_values:
        message += "..."

    return message


def _load_ruc_lookup(data: RUCLookup, zones: np.ndarray) -> pd.Series:
    """Load RUC lookup data into Series of weights to apply."""
    LOG.info("Loading RUC lookup from %s", data.path.name)
    lookup_data = pd.read_csv(data.path, usecols=[data.id_column, data.ruc_column])
    lookup: pd.Series = lookup_data.set_index(data.id_column, verify_integrity=True)[
        data.ruc_column
    ]

    unknown_zones = lookup.index[~lookup.index.isin(zones)]
    if len(unknown_zones) > 0:
        raise ValueError(
            f"{len(unknown_zones)} values in RUC lookup zones not "
            f"found in centroids data: {_summarise_list(unknown_zones)}"
        )

    lookup = lookup.astype(str).str.upper()
    invalid_ruc = lookup[~lookup.isin(RUC_WEIGHTS)].unique().tolist()
    if len(invalid_ruc) > 0:
        raise ValueError(
            f"{len(invalid_ruc)} values in RUC classifications not "
            f"found in centroids data: {_summarise_list(invalid_ruc)}"
        )

    missing_zones = zones[~np.isin(zones, lookup.index)]
    if len(missing_zones) > 0:
        LOG.warning(
            "RUC distance factor defaulting to 1 for %s zones: %s",
            len(missing_zones),
            _summarise_list(missing_zones),
        )
        lookup = pd.concat([lookup, pd.Series(data=1, index=missing_zones)])

    lookup = lookup.replace(RUC_WEIGHTS).astype(float)

    return lookup


def _load_irrelevant_destinations(
    data: IrrelevantDestinations, zones: np.ndarray
) -> np.ndarray | None:
    """Load array of destinations to exclude, return None if file is empty."""
    LOG.info("Loading irrelevant destinations from %s", data.path.name)
    irrelevant_data = pd.read_csv(data.path, usecols=[data.zone_column])
    irrelevant = irrelevant_data[data.zone_column].unique()

    unknown_zones = irrelevant[~np.isin(irrelevant, zones)]
    if len(unknown_zones) > 0:
        raise ValueError(
            f"{len(unknown_zones)} zones in irrelevant destinations not "
            f"found in centroids data: {_summarise_list(unknown_zones)}"
        )

    LOG.info("Given %s unique destinations to exclude", len(irrelevant))

    if len(irrelevant) == 0:
        return None

    return irrelevant


def _build_calculation_parameters_iter(
    settings: CostSettings,
    columns: centroids.ZoneCentroidColumns,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame | None = None,
    irrelevant_destinations: np.ndarray | None = None,
    distances: pd.DataFrame | None = None,
    distance_factors: pd.Series | None = None,
    progress_bar: bool = True,
) -> Iterator[CalculationParameters]:
    """Generate calculation parameters, internal function for `build_calculation_parameters`."""

    def row_to_place(row: pd.Series) -> routing.Place:
        point = row.at["geometry"]

        return routing.Place(
            name=str(row.at[columns.name]),
            id=str(row.name),
            zone_system=str(row.at[columns.system]),
            lon=point.x,
            lat=point.y,
        )

    od_pairs = None

    # Filter OD pairs based on distance before looping
    if settings.crowfly_max_distance is not None:
        if distance_factors is not None and distances is not None:
            # Factor distances down to account for RUC factor
            # increasing the max distance filter, factor applied
            # to use origin RUC
            distance_factors.index.name = "origin"
            distances = distances.divide(distance_factors, axis=0)
            LOG.info(
                "Applied RUC distance factors to %s OD pairs",
                (distance_factors != 1).sum(),
            )
        else:
            LOG.info("No RUC distance factors applied")

        if distances is not None:
            od_pairs = distances.loc[
                distances <= settings.crowfly_max_distance
            ].index.to_frame(index=False)
            LOG.info(
                "Dropped %s requests with crow-fly distance > %s (%s remaining)",
                f"{len(distances) - len(od_pairs):,}",
                f"{settings.crowfly_max_distance:,}",
                f"{len(od_pairs):,}",
            )
    else:
        LOG.info("No crow-fly distance filtering applied")

    if od_pairs is None:
        od_pairs = pd.DataFrame(
            itertools.product(origins.index, origins.index),
            columns=["origin", "destination"],
        )

    if irrelevant_destinations is not None:
        exclude = od_pairs["destination"].isin(irrelevant_destinations)
        od_pairs = od_pairs.loc[~exclude]
        LOG.info(
            "Dropped %s OD pairs with irrelevant destinations, %s remaining",
            f"{exclude.sum():,}",
            f"{len(od_pairs):,}",
        )
    else:
        LOG.info("No irrelevant destinations excluded")

    if progress_bar:
        iterator = tqdm.tqdm(
            od_pairs.itertuples(index=False, name=None),
            desc="Building parameters",
            dynamic_ncols=True,
            total=len(od_pairs),
        )
    else:
        iterator = od_pairs.itertuples(index=False, name=None)

    for origin, destination in iterator:
        yield CalculationParameters(
            server_url=settings.server_url,
            modes=[str(m) for m in settings.modes],
            datetime=settings.datetime,
            origin=row_to_place(origins.loc[origin]),
            destination=row_to_place(
                origins.loc[destination]
                if destinations is None
                else destinations.loc[destination]
            ),
            arrive_by=settings.arrive_by,
            searchWindow=settings.search_window_seconds,
            wheelchair=settings.wheelchair,
            max_walk_distance=settings.max_walk_distance,
            crowfly_max_distance=settings.crowfly_max_distance,
        )


def build_calculation_parameters(
    zones: centroids.ZoneCentroids,
    settings: CostSettings,
    ruc_lookup: RUCLookup | None = None,
    irrelevant_destinations: IrrelevantDestinations | None = None,
    crowfly_distance_crs: str = CROWFLY_DISTANCE_CRS,
) -> list[CalculationParameters]:
    """Build a list of parameters for running `calculate_costs`.

    Parameters
    ----------
    zone_centroids :ZoneCentroids
        Positions of zones to calculate costs between.
    settings : CostSettings
        Additional settings for calculating the costs.
    ruc_lookup : RUCLookup, optional
        File containing zone lookup for rural urban classification,
        used for factoring max crow-fly distance filter.
    irrelevant_destinations: IrrelevantDestinations, optional
        File containing list of destinations to exclude from
        calculation parameters.
    crowfly_distance_crs : str, default `CROWFLY_DISTANCE_CRS`
        Coordinate reference system to convert geometries to
        when calculating crow-fly distances.

    Returns
    -------
    list[CalculationParameters]
        Parameters to run `calculate_costs`.
    """
    LOG.info("Building cost calculation parameters")

    zone_columns = [zones.columns.name, zones.columns.system, "geometry"]
    origins = zones.origins.set_index(zones.columns.id)[zone_columns]

    if zones.destinations is None:
        destinations = None
    else:
        destinations = zones.destinations.set_index(zones.columns.id)[zone_columns]

    if settings.crowfly_max_distance is not None and settings.crowfly_max_distance > 0:
        distances = _calculate_distance_matrix(
            origins,
            origins if destinations is None else destinations,
            crowfly_distance_crs,
        )
    else:
        distances = None

    if ruc_lookup is not None and distances is not None:
        distance_factors = _load_ruc_lookup(ruc_lookup, origins.index.values)
    else:
        distance_factors = None

    if irrelevant_destinations is not None:
        irrelevant = _load_irrelevant_destinations(
            irrelevant_destinations, origins.index.values
        )
    else:
        irrelevant = None

    parameters: list[CalculationParameters] = []
    iterator = _build_calculation_parameters_iter(
        settings,
        zones.columns,
        origins=origins,
        destinations=destinations,
        irrelevant_destinations=irrelevant,
        distances=distances,
        distance_factors=distance_factors,
    )

    for params in iterator:
        if isinstance(params, CalculationParameters):
            parameters.append(params)

        else:
            raise TypeError(
                f"unexpected type ({type(params)}) returned "
                "by `_build_calculation_parameters_iter`"
            )

    LOG.info("Built parameters for %s requests", f"{len(parameters):,}")
    return parameters


def save_calculation_parameters(
    zones: centroids.ZoneCentroids,
    settings: CostSettings,
    output_file: pathlib.Path,
    **kwargs,
) -> pathlib.Path:
    """Build calibration parameters and save to JSON lines file.

    Parameters
    ----------
    zone_centroids :ZoneCentroids
        Positions of zones to calculate costs between.
    settings : CostSettings
        Additional settings for calculating the costs.
    output_file : pathlib.Path
        File to save, suffix will be changed to '.jsonl'
        if it isn't already.

    Returns
    -------
    pathlib.Path
        Path to JSON lines file created.
    """
    parameters = build_calculation_parameters(zones, settings, **kwargs)
    pbar = tqdm.tqdm(parameters, desc="Saving parameters", dynamic_ncols=True)

    output_file = output_file.with_suffix(".jsonl")
    with open(output_file, "wt", encoding="utf-8") as file:
        for params in pbar:
            file.write(params.json() + "\n")

    LOG.info(
        "Written %s calculation parameters to JSON lines file: %s",
        len(parameters),
        output_file,
    )
    return output_file


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
        fromPlace=f"{parameters.origin.lat}, {parameters.origin.lon}",
        toPlace=f"{parameters.destination.lat}, {parameters.destination.lon}",
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
            # Set value to NaN if it doesn"t exist or isn"t set
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
    zone_centroids: centroids.ZoneCentroids,
    settings: CostSettings,
    matrix_file: pathlib.Path,
    generalised_cost_parameters: GeneralisedCostFactors,
    aggregation_method: AggregationMethod,
    ruc_lookup: RUCLookup | None = None,
    irrelevant_destinations: IrrelevantDestinations | None = None,
    workers: int = 0,
) -> None:
    """Create cost matrix for all zone to zone pairs.

    Parameters
    ----------
    zone_centroids : ZoneCentroids
        Zones for the cost matrix.
    settings : CostSettings
        Settings for calculating the costs.
    matrix_file : pathlib.Path
        Path to save the cost matrix to.
    generalised_cost_parameters : GeneralisedCostParameters
        Factors and other parameters for calculating the
        generalised cost.
    aggregation_method : AggregationMethod
        Aggregation method used for generalised cost matrix.
    ruc_lookup : RUCLookup, optional
        File containing zone lookup for rural urban classification,
        used for factoring max crow-fly distance filter.
    irrelevant_destinations: IrrelevantDestinations, optional
        File containing list of destinations to exclude from
        calculation parameters.
    workers : int, default 0
        Number of threads to create during calculations.
    """
    LOG.info("Calculating costs for %s with settings\n%s", matrix_file.name, settings)
    jobs = build_calculation_parameters(
        zones=zone_centroids,
        settings=settings,
        ruc_lookup=ruc_lookup,
        irrelevant_destinations=irrelevant_destinations,
    )

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


def iterate_responses(response_file: io.TextIOWrapper) -> Iterator[CostResults]:
    """Iterate through and parse responses JSON lines file.

    Parameters
    ----------
    response_file : io.TextIOWrapper
        JSON lines file to read from.

    Yields
    ------
    CostResults
        Cost results for a single OD pair.
    """
    for line in response_file:
        yield CostResults.parse_raw(line)


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
