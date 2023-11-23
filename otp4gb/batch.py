import datetime
import logging
import operator
import os
import geopandas as gpd
import pandas as pd
import threading

from shapely import envelope


from otp4gb.geo import (
    buffer_geometry,
    get_valid_json,
    parse_to_geo,
    sort_by_descending_time,
)
from otp4gb.net import api_call
from otp4gb.centroids import ZoneCentroids, _CENTROIDS_CRS



_TIME_BY_STRING : str = "By"
_TIME_BETWEEN_STRING : str = "Between"


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
                for c in range(travel_time_step, int(travel_time_max), travel_time_step)
            ] + [("cutoff", str(travel_time_max) + "M")]
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



# performance - .buffer is massively expensive on complex geometries (e.g. car accessibility isochrones) 
# taking a comparable amount of time / cpu as the entire call to the OTP server. 
# 
# To put it in context if .to_crs takes 1 unit of time, .simplify takes 10 units of time, .make_valid takes 50 units of time
# .buffer takes 100 units of time, the entire call to the OTP server takes 200 units of elapsed time.
#
# so we can put in a lot of logic/ exception handlers and still have much better performance if we avoid calls to .buffer, and call buffer after .simplify
#
def run_batch(batch_args: list[dict]) -> dict:
    #if supplied with more than one request we union all the isochrones together grouped by the 'time' attribute in the responses.
    #used when testing public transport accessibilty. Testing for just one time when service frequencies are low can give misleading
    #results, so (dependent on config file setting) we test for multiple arrival times and create a union of the best accessibility.
    
    threadId = threading.get_ident()

    batchResponses = gpd.GeoDataFrame()
    
    travelTimes = set()

    for request_args in batch_args:
        
        logger.debug("args = %s", request_args)
        url, name, mode, travel_time, destination, arrive = operator.itemgetter(
            "url", "name", "mode", "travel_time", "destination", "arrive"
        )(request_args)

        logger.info("T%d:Processing %s for %s at time %s", threadId, mode, name, travel_time)
        
        travelTimes.add(datetime.datetime.fromisoformat(travel_time))

        #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
        logger.debug("T%d:Getting URL %s", threadId, url)
        requestData = api_call(url)

        requestData = parse_to_geo(requestData)

        batchResponses = pd.concat([batchResponses, requestData])


    # Remove rows with no geometry
    batchResponses = batchResponses.dropna(subset=["geometry"])
    # and remove a different kind of empty
    batchResponses = batchResponses[~batchResponses.geometry.is_empty]
    
    if len(batchResponses) > 0:
    
        batchResponses.geometry = batchResponses.geometry.to_crs("EPSG:27700") #transform to GB national grid so that we can work in metre units.
            # discuss if this is better worse or indifferent to EPSG:23030") which is what the original code used
        try:
            #boundingBox =  batchData.geometry.envelope
            batchResponses = doGeometryProcessing( batchResponses, travelTimes )

        except Exception as e:
            logger.warn("T%d:Exception during geometry processing %s for %s (%s) - trying again...", threadId, mode, name, e)
            
            batchResponses.geometry = batchResponses.geometry.make_valid()
            
            # simplifying the geometry reduces some to being empty
            batchResponses = batchResponses.dropna(subset=["geometry"])
            batchResponses = batchResponses[~batchResponses.geometry.is_empty]
            
            if len(batchResponses) > 0:
                batchResponses = doGeometryProcessing( batchResponses, travelTimes )
            
            logger.info("T%d:Geometry processing successful after make_valid() %s for %s", threadId, mode, name)
            
        
        if len(batchResponses) > 0:
            batchResponses.geometry = batchResponses.geometry.to_crs(_CENTROIDS_CRS) #transform back to lat/long.
            
            travel_time_matrix = saveIsochronesAndGenerateMatrix( batchResponses, travelTimes, destination, name, mode, arrive )
        else:  
            logger.warn("T%d: no valid geometry returned for %s for %s", threadId, mode, name)
            travel_time_matrix = dict()        
    else:  
        logger.warn("T%d: no valid geometry returned for %s for %s", threadId, mode, name)
        travel_time_matrix = dict()

    logger.info("T%d:Completing %s for %s", threadId, mode, name)
        
    return travel_time_matrix



def doGeometryProcessing( batchResponses: gpd.GeoDataFrame, travelTimes: set[datetime.datetime]
                                  ) -> gpd.GeoDataFrame:
    
    #union all the isochrones for the same travel time together.
    if len(travelTimes) > 1:
        batchResponses = batchResponses.dissolve("time", as_index=False)
        #if the geometry is invalid this will blow up with GEOSException: TopologyException: unable to assign free hole to a shell at.....   
        #fail fast - fail early, will get called again after .make_valid has been called. 

        # simplifying/ buffering the geometry reduces some to being empty
        batchResponses = batchResponses.dropna(subset=["geometry"])
        batchResponses = batchResponses[~batchResponses.geometry.is_empty]
    
    if len(batchResponses) > 0:
        batchResponses = batchResponses.apply(processOneGeometry, axis=1)          
        
        # simplifying/ buffering the geometry reduces some to being empty
        batchResponses = batchResponses.dropna(subset=["geometry"])
        batchResponses = batchResponses[~batchResponses.geometry.is_empty]
                
    return batchResponses

def processOneGeometry(row):
    
    if row.geometry and 0 != _buffer_size:
        
        boundsBox = row.geometry.envelope.bounds
        width = boundsBox[2] - boundsBox[0]
        height = boundsBox[3] - boundsBox[1]
    
        #don't buffer or simplify very small shapes
        if width > 500 and height > 500:
            
            #half the buffer size for small shapes
            bufferSizeToUse =  _buffer_size if width > 1000 and height > 1000 else _buffer_size/2            

            if _buffer_size > 0:
                row.geometry = row.geometry.simplify( tolerance=10, preserve_topology=False )

                # resolution of 2 is an octogon shape
                row.geometry = row.geometry.buffer(distance = bufferSizeToUse, resolution=2)
            elif _buffer_size < 0:
                row.geometry = row.geometry.simplify( tolerance= -bufferSizeToUse, preserve_topology=True )

    return row



def saveIsochronesAndGenerateMatrix( batchResponses: gpd.GeoDataFrame, 
                                   travelTimes: set[datetime.datetime],
                                  destination :str,
                                  name :str,
                                  mode :str,
                                  arrive :bool
                                  ) -> dict:
    
    threadId = threading.get_ident()

    batchResponses = sort_by_descending_time(batchResponses) #bug: times were being treated as strings, so 900 came before 1800
    largest = batchResponses.iloc[[0]] #bug: this was calling loc, not iloc, so the sorting had no effect.

    #despite the buffer - the isochrone generated can have 'noise' in the form of islands of inaccessiblity - be cautious and clip to bounding box not isochrone itself
    origins = _centroids.origins.clip(largest.envelope)
    origins = origins.assign(travel_time="")

    if len(travelTimes) > 1:
        travelTimeForFilename = min(travelTimes).isoformat() + "_and_" + max(travelTimes).isoformat()
        byString = _TIME_BETWEEN_STRING
    elif travelTimes:
        travelTimeForFilename =  next(iter(travelTimes))
        byString = _TIME_BY_STRING

    # Calculate all possible origins within travel time by minutes
    for i in range(batchResponses.shape[0]):
        
        row = batchResponses.iloc[[i]]
        
        journey_time = int(row.time.iloc[0])
        
        logger.debug("T%d:Journey time %i", threadId, journey_time)
        
        filename = _FILENAME_PATTERN.format(
            location_name=name,
            mode=mode,
            buffer_size=_buffer_size,
            by_or_between = byString, 
            arrival_time=travelTimeForFilename,
            journey_time=journey_time / 60,
            arrive_depart= "Arrive" if arrive else "Depart",
        ) 
        filename = filename.replace(',', '_')
        filename = filename.replace(':', '-')
        filename = filename.replace(' ', '_')

        # Write isochrone
        with open(os.path.join(_output_dir, filename), "w") as f:
            f.write(get_valid_json(row))

        #despite all the make_valid etc, we very occasionally get a TopologyException: side location conflict       
        try:
            covered_indexes = origins.clip(row).index
        except Exception as e:
            logger.warn("T%d: exception while clipping %s for %s (%s) trying again... ", threadId, mode, name, e )
            batchResponses.at[row.index[0], 'geometry'] = row.geometry.make_valid().iloc[0]
            row = batchResponses.iloc[[i]]
            covered_indexes = origins.clip(row).index
            
        logger.debug(
            "T%d:Mode %s for %s covers %i centroids in %i seconds",
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


    #since we clipped to bounding box we may have some inaccessible origins - remove them.
    origins = origins[origins.travel_time != ""]

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

    # Convert to native python dict
    travel_time_matrix = travel_time_matrix.to_dict(orient="records")

    return travel_time_matrix



def run_batch_catch_errors(batch_args: list[dict]) -> dict:
    """Wraps `run_batch` catches any exceptions and returns dict of arguments and error."""
    try:
        return run_batch(batch_args)
    except Exception as e:
        
        #just return the first one if we have more than one job in the batch at this level
        for job in batch_args:
            args = job.copy()
            
            # Get destination name instead of whole Series
            dest_name = args.pop("destination").loc[_name_key]
            args["destination"] = dest_name

            err_msg = f"{e.__class__.__name__}: {e}"
            logger.error("T%d:Error in run_batch(%s) - %s", threading.get_ident(), args, err_msg)
            args["error"] = err_msg
            return args
            
 