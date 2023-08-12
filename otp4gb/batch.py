import logging
import operator
import os
import pandas as pd
import threading

from otp4gb.geo import (
    buffer_geometry,
    get_valid_json,
    parse_to_geo,
    sort_by_descending_time,
)
from otp4gb.net import api_call
from otp4gb.centroids import ZoneCentroids

logger = logging.getLogger(__name__)


#### OTP config
#
#// otp-config.json
#{
#    "otpFeatures" : {
#        "SandboxAPITravelTime" : true
#    }
#}
#
#### API parameters
#
#- `location` Origin of the search, can be either `latitude,longitude` or a stop id
#- `time` Departure time as a ISO-8601 time and date (example `2023-04-24T15:40:12+02:00`). The default value is the current time.
#- `cutoff` The maximum travel duration as a ISO-8601 duration. The `PT` can be dropped to simplify the value. 
#  This parameter can be given multiple times to include multiple isochrones in a single request.
#  The default value is one hour.
#- `modes` A list of travel modes. WALK is not implemented, use `WALK, TRANSIT` instead.
#- `arriveBy` Set to `false` when searching from the location and `true` when searching to the 
#  location
#
#### Isochrone API
#    /otp/traveltime/isochrone
#
#Results is the travel time boundaries at the `cutoff` travel time.
#
#### Travel time surface API
#    /otp/traveltime/surface
#
#The travel time as a GeoTIFF raster file. The file has a single 32-bit int band, which contains the 
#travel time in seconds.
#
#### Example Request
#http://localhost:8080/otp/traveltime/isochrone?batch=true&location=52.499959,13.388803&time=2023-04-12T10:19:03%2B02:00&modes=WALK,TRANSIT&arriveBy=false&cutoff=30M17S



def build_run_spec(
    name_key,
    modes,
    centroids : ZoneCentroids,
    arrive_by,
    travel_time_max,
    travel_time_step,
    max_walk_distance,
    server,
    arrive,
):
    items = []
    locations: ZoneCentroids = centroids.destinations
    if locations is None:
        locations = centroids.origins

    for _, destination in locations.iterrows():
        name = destination[name_key]
        location = [destination.geometry.y, destination.geometry.x]
        for mode in modes:
            modeText = ",".join(mode)
            cutoffs = [
                ("cutoff", str(c) + "M")
                for c in range(travel_time_step, int(travel_time_max) + 1, travel_time_step)
            ]
            query = [
                ("location", ",".join([str(x) for x in location])),
                ("modes", modeText),
                ("time", arrive_by.isoformat()),
                ("arriveby", "true" if arrive else "false"),
            ] + cutoffs
            url = server.get_root_url("traveltime/isochrone", query=query)
            batch_spec = {
                "name": name,
                "travel_time": arrive_by.isoformat(),
                "url": url,
                "mode": modeText,
                "destination": destination,
                "arrive": arrive,
            }
            items.append(batch_spec)
    return items


def setup_worker(config):
    global _output_dir, _centroids, _buffer_size, _FILENAME_PATTERN, _name_key
    _output_dir = config.get("output_dir")
    _centroids = config.get("centroids")
    _buffer_size = config.get("buffer_size")
    _FILENAME_PATTERN = config.get("FILENAME_PATTERN")
    _name_key = config.get("name_key")


def run_batch(batch_args: dict) -> list[dict]:
    logger.debug("args = %s", batch_args)
    url, name, mode, travel_time, destination, arrive = operator.itemgetter(
        "url", "name", "mode", "travel_time", "destination", "arrive"
    )(batch_args)

    threadId = threading.get_ident()

    logger.info("T%d:Processing %s for %s", threadId, mode, name)

    #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
    logger.debug("T%d:Getting URL %s", threadId, url)
    data = api_call(url)

    data = parse_to_geo(data)

    data = buffer_geometry(data, buffer_size=_buffer_size)

    data = sort_by_descending_time(data)
    largest = data.loc[[0]]

    origins = _centroids.origins.clip(largest)
    origins = origins.assign(travel_time="")

    # Calculate all possible origins within travel time by minutes
    for i in range(data.shape[0]):
        row = data.iloc[[i]]
        journey_time = int(row.time.iloc[0])
        logger.debug("T%d:Journey time %s", threadId, journey_time)
        geojson_file = _FILENAME_PATTERN.format(
            location_name=name,
            mode=mode,
            buffer_size=_buffer_size,
            arrival_time=travel_time,
            journey_time=journey_time / 60,
            arrive_depart= "Arrive" if arrive else "Depart",
        ) 
        geojson_file = geojson_file.replace(',', '_')
        geojson_file = geojson_file.replace(':', '-')
        geojson_file = geojson_file.replace(' ', '_')

        # Write isochrone
        with open(os.path.join(_output_dir, geojson_file), "w") as f:
            f.write(get_valid_json(row))

        covered_indexes = origins.clip(row).index
        logger.debug(
            "T%d:Mode %s for %s covers %s centroids in %s seconds",
            threadId,
            mode,
            name,
            len(covered_indexes),
            int(row.time.iloc[0]),
        )
        updated_times = pd.DataFrame(
            {"travel_time": journey_time}, index=covered_indexes
        )
        origins.update(updated_times)

    travel_time_matrix = pd.DataFrame(
        {
            "OriginName": origins[_name_key],
            "OriginLatitude": origins.geometry.y,
            "OriginLongitude": origins.geometry.x,
            "DestinationName": name,
            "DestinationLatitude": destination.geometry.y,
            "DestinationLongitide": destination.geometry.x,
            "Mode": mode,
            "Minutes": origins.travel_time / 60,
        }
    )

    # Drop duplicate source / destination
    travel_time_matrix = travel_time_matrix[
        ~(travel_time_matrix["OriginName"] == travel_time_matrix["DestinationName"])
    ]

    logger.debug("T%d:Travel Matrix ==>\n%s", threadId, travel_time_matrix)

    # Convert to native python list
    travel_time_matrix = travel_time_matrix.to_dict(orient="records")

    logger.info("T%d:Completing %s for %s", threadId, mode, name)

    return travel_time_matrix


def run_batch_catch_errors(batch_args: dict) -> list[dict]:
    """Wraps `run_batch` catches any exceptions and returns dict of arguments and error."""
    try:
        return run_batch(batch_args)
    except Exception as e:
        # Get destination name instead of whole Series
        args = batch_args.copy()
        dest_name = args.pop("destination").loc[_name_key]
        args["destination"] = dest_name

        err_msg = f"{e.__class__.__name__}: {e}"
        logger.error("T%d:Error in run_batch(%s) - %s", threading.get_ident(), args, err_msg)

        return [{**args, "error": err_msg}]
