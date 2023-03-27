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
from otp4gb import routing, util, centroids


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
    global zone_centroids_BnG
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
    #### LSOA TRSE ####
    # TODO: Move this to a config file as optional parameter
    filter_radius = 24150  # metres (0 if not required) (15 miles)

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
    #print('zone_centroids:', zone_centroids)
    print("\n LSOA analysis maximum radius:", str(filter_radius))
    print(len(zone_centroids))

    # Load LSOA relevance and area type lookup file
    import os
    LSOA_relevance_path = os.path.join(os.getcwd(), 'Data', 'LSOA_amenities.csv')
    LSOA_area_path = os.path.join(os.getcwd(), 'Data', 'compiled_LSOA_area_types.csv')
    LSOA_relevance = pd.read_csv(LSOA_relevance_path)
    LSOA_area_types = pd.read_csv(LSOA_area_path)

    # Set zone id's as index, matching `zone_centroids`
    LSOA_relevance.set_index('LSOA11CD', inplace=True)
    LSOA_area_types.set_index('LSOA11CD', inplace=True)

    compiled_centroids_path = (os.path.join(os.getcwd(), 'Data', 'compiled_zone_centroids_with_filter.csv'))

    #TODO: find "area_type" and convert to "RUC" (Rural Urban Classification) as these are not area_types.
    # Rural weighting for Origin LSOAs
    LSOA_area_type_weights = {'A1': 1,
                              'B1': 1,
                              'C1': 1,
                              'C2': 1,
                              'D1': 1.25,
                              'D2': 1.25,
                              'E1': 1.5,
                              'E2': 1.5}

    # Create GeoDataFrame of all OD pairs & work out distance between them.
    if (filter_radius != 0) & (os.path.isfile(compiled_centroids_path) is False):

        print('\nA zone centroids file named "compiled_zone_centroids_with_filter.csv" could not be found.')
        print('Creating `compiled_zone_centroids_with_filter.csv` now.')

        # Create copy of zone_centroids to manipulate CRS (filter distance is metres (EPSG:27700))
        zone_centroids_BnG = zone_centroids.copy()
        zone_centroids_BnG = gpd.GeoDataFrame(zone_centroids_BnG)
        zone_centroids_BnG.to_crs('EPSG:27700', inplace=True)
        
        # For joining destination centroids later 
        zone_centroids_BnG_raw = zone_centroids_BnG.copy()

        # Set crs to BnG [EPSG:27700] so distance calc is in metres.
        zone_centroids_BnG.to_crs('EPSG:27700')

        zone_centroids_BnG.drop(columns=['zone_system', 'zone_name'],
                                inplace=True)

        # Create a DataFrame of all possible OD combinations:

        origins = []
        destinations = []
        OD_pairs = []
        LOG.info("Constructing distance GDF and assessing relevant Destination zones")
        LSOA_ids = list(zone_centroids.index)

        # Create possible combinations
        length = 2  # An Origin & Destination
        x = list(range(len(zone_centroids)))  # Possible Origins or Destinations
        mesh = np.meshgrid(*([x] * length))
        result = np.vstack([y.flat for y in mesh]).T

        print('\nAnalysing', len(result), 'initial trips for destination relevance & same zone journeys.\n')

        for i in tqdm.tqdm(zip(result)):
            o = i[0][0]
            d = i[0][1]

            # Ignore same zone journeys
            if o == d:
                continue

            # Check for relevance of Destination LSOA
            if LSOA_relevance.loc[LSOA_ids[d]]['totals'] > 0:
                origin = LSOA_ids[o]
                destin = LSOA_ids[d]

                origins.append(origin)
                destinations.append(destin)
                OD_pairs.append('_'.join((origin, destin)))
            else:
                # Destination LSOA is not relevant
                continue
        LOG.info("Removing: "+str((len(result) - len(origins)))+" trips, leaving: "+str(len(origins))+' trips remaining.\n')

        LOG.info("Constructing Data Frame")
        # DF of all OD pairs
        OD_pairs = pd.DataFrame(data={'Origins': origins,
                                      'Destinations': destinations,
                                      'OD_pairs': OD_pairs})

        LOG.info("Merging Origins to geometries")
        # Join on zone centroid data for Origins
        zone_centroids_BnG = zone_centroids_BnG.merge(how='left',
                                                      right=OD_pairs,
                                                      left_on=zone_centroids_BnG.index,
                                                      right_on='Origins')
                                                      
        # Re-classify dataframe as GeoDataFrame                                              
        zone_centroids_BnG = gpd.GeoDataFrame(zone_centroids_BnG,
                                              crs='EPSG:27700')

        # Rename Origin centroids col to Origin_centroids
        zone_centroids_BnG.rename(columns={'geometry': 'Origin_centroids'},
                                  inplace=True)
        # Re-specify geometry
        zone_centroids_BnG.set_geometry('Origin_centroids',
                                         inplace=True,
                                         crs='EPSG:27700')

        LOG.info("Merging Destinations to geometries")
        # Join on zone centroids data for Destinations
        zone_centroids_BnG = zone_centroids_BnG.merge(how='left',
                                                      right=zone_centroids_BnG_raw,
                                                      left_on='Destinations',
                                                      right_on=zone_centroids_BnG_raw.index)
        # Re-classify dataframe as GeoDataFrame                                              
        zone_centroids_BnG = gpd.GeoDataFrame(zone_centroids_BnG,
                                              crs='EPSG:27700')

        # Rename centroids to destination centroids
        zone_centroids_BnG.rename(columns={'geometry': 'Destination_centroids'},
                                  inplace=True)

        LOG.info("Calculating journey distances")
        # Work out distance between all OD pairs
        zone_centroids_BnG['distances'] = zone_centroids_BnG['Origin_centroids'].distance(zone_centroids_BnG['Destination_centroids'])

        # Print run stats for user
        print('\nMaximum trip distance:', str(max(zone_centroids_BnG['distances'])),
              '\nMinimum trip distance:', str(min(zone_centroids_BnG['distances'])), '\n')

        # Short enough trips are the number of trips within the specified filter radius with Rural weighting applied
        # to the filter distance. Any trip distance greater than this can now be removed.
        short_enough_trips = len(zone_centroids_BnG[zone_centroids_BnG['distances'] < (filter_radius*LSOA_area_type_weights['E2'])])
        all_trips = len(zone_centroids_BnG)
        diff = str(all_trips - short_enough_trips)


        LOG.info("Removing "+str(diff)+" trips for being too far.\n\nLeaving: "+str(short_enough_trips)+' trips remaining.')
        zone_centroids_BnG = zone_centroids_BnG[zone_centroids_BnG['distances'] < filter_radius*LSOA_area_type_weights['E2']]

        # At this point we have likely spent a long time compiling and computing the above distances. Let's save it in
        # case of crashes etc, and then we can re-load the above if it has already been compiled.

        # Save this as a csv. Tho it is a GeoDataFrame we no longer need spatial information here, only trip distances.
        compiled_zone_centroids_path = os.path.join(os.getcwd(), 'Data', 'compiled_zone_centroids_with_filter.csv')

        print('\nSaving compiled zone_centroids_BnG file as csv to:\n\n', compiled_zone_centroids_path, '\n')

        zone_centroids_BnG = zone_centroids_BnG[["Origins", "Destinations", "OD_pairs", centroids_columns.name,
                                                 centroids_columns.system, 'distances']]

        zone_centroids_BnG.to_csv(compiled_zone_centroids_path)

    elif (filter_radius != 0) & (os.path.isfile(compiled_centroids_path) is True):  # compiled centroids has been found
        # File already exists, so load
        compiled_centroids_path = (os.path.join(os.getcwd(), 'Data', 'compiled_zone_centroids_with_filter.csv'))
        zone_centroids_BnG = pd.read_csv(compiled_centroids_path)
        print('Existing compiled zone_centroids file found. Loading the \n', compiled_centroids_path)
        print('Existing compiled zone_centroids file found. Loading the DataFrame\n', compiled_centroids_path)

    params = []
    if filter_radius == 0:
        # No filter radius applied - carry on as normal.
        for o, d in itertools.product(zone_centroids.index, zone_centroids.index):
            if o == d:
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

    LOG.info("Assessing if trips are too far based on Rural Urban Classifications.")

    too_far_destinations = 0
    if filter_radius != 0:
        # Filter radius has been applied - use compiled GDF from above
        #TODO: replace itertools.product with zip BnG ids
        compiled_centroids_path = (os.path.join(os.getcwd(), 'Data', 'compiled_zone_centroids_with_filter.csv'))
        zone_centroids_BnG = pd.read_csv(compiled_centroids_path)
        print('\nTrips to assess:', len(zone_centroids_BnG))

        zone_distances = zone_centroids_BnG.copy()
        # Create DF of journey distances with OD_pairs as index to apply .loc[] with OD_pairs
        zone_distances.set_index('OD_pairs', inplace=True)

        for o, d in tqdm.tqdm(zip(zone_centroids_BnG['Origins'], zone_centroids_BnG['Destinations'])):

            # Check if OD journey exceeds maximum distance with RUC weightings applied
            if filter_radius != 0:
                od_code = '_'.join((str(o), str(d)))

                od_distance = zone_distances.loc[od_code]['distances']
                # Check area type of origin zone - apply extra radius weighting if origin is rural
                radius_weight = LSOA_area_type_weights[LSOA_area_types.loc[o]['RUC11CD']]

                if od_distance > (filter_radius * radius_weight):
                    too_far_destinations += 1
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

        LOG.info('Additional: '+str(too_far_destinations)+' journeys removed for being too far based on RUCs.')
        LOG.info('Requests will now be sent for '+str(len(params))+' journeys.')
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
