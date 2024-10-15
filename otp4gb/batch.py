import datetime
import logging
import operator
import os
import geopandas as gpd
import pandas as pd
import re
import threading
import uuid

from shapely import envelope


from otp4gb.geo import (
    buffer_geometry,
    get_valid_json,
    write_valid_json,
    parse_to_geo,
    sort_by_descending_time,
)
from otp4gb.net import api_call_retry
from otp4gb.centroids import ZoneCentroids, _CENTROIDS_CRS
from otp4gb.geo import _PROCESSING_CRS


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
    travel_time_min,
    travel_time_max,
    travel_time_step,
    server,
    arrive,
):
    items = []
    locations: ZoneCentroids = centroids.destinations
    if locations is None:
        locations = centroids.origins

    if travel_time_min is None:
        travel_time_min = travel_time_step

    for _, destination in locations.iterrows():
        name = destination[name_key]
        location = [destination.geometry.y, destination.geometry.x]
        for mode in modes:
            modeText = ",".join(mode)
            cutoffs = [
                ("cutoff", str(c) + "M")
                for c in range(int(travel_time_min), int(travel_time_max), travel_time_step)
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
    global _output_dir, _buffer_size, _FILENAME_PATTERN, _name_key, _centroids, _test_origin_areas, _centroids_origin_sindex, _fanout_directory
    _output_dir = config.get("output_dir")
    _buffer_size = config.get("buffer_size")
    _FILENAME_PATTERN = config.get("FILENAME_PATTERN")
    _name_key = config.get("name_key")
    
    _centroids = config.get("centroids")    
    _test_origin_areas = 'polygon' in _centroids.origins.columns
        
    #reproject so that area calulations work OK
    if _test_origin_areas:
        _centroids.origins.set_geometry(col='polygon',drop=True,inplace=True)
        _centroids.origins.to_crs(crs=_PROCESSING_CRS, inplace=True)
        _centroids.origins['originArea'] = _centroids.origins.geometry.area
    else:
        _centroids.origins.to_crs(crs=_PROCESSING_CRS, inplace=True)
        
    _centroids_origin_sindex = _centroids.origins.sindex
    _fanout_directory = config.get("fanout_directory")




_isochroneVariables = ['buffer', 'mode', 'zone', 'arriveDepart', 'ByBetween', 'fromDateTime', 'toDateTime', 'travelTime']

'''
def _getValuesFromFilename( filename: str) -> dict:
        
    regex_pattern = r"buffered(\d+).*isochroneby_(.+)_tozone_(.+)_to(.+?)_(.+?)_(.+?)(_and_(.+))?_within.*?(\d+).*mins"
        
    match = re.match(regex_pattern, filename, re.IGNORECASE)
    if match:
        groups = match.groups()
        # Define keys for the dictionary
        keys = ['buffer', 'mode', 'zone', 'arriveDepart', 'ByBetween', 'fromDateTime', 'optional', 'toDateTime', 'travelTime']
        # Construct dictionary with keys and extracted values
        extracted_values = dict(zip(keys, groups))
        del extracted_values['optional']
        return extracted_values
    else:
        return None  


def _add_attributes_to_geojson(geojson_data, dict):
    # Modify the GeoJSON data here to add attributes
    for feature in geojson_data['features']:
        for attribute, value in dict.items():
            feature['properties'][attribute] = value
    return geojson_data
'''

def _mkdir( path ):
    # Create the directory if it doesn't exist
    directory = os.path.dirname( path )
    if not os.path.exists( directory ):
        os.makedirs( directory, exist_ok=True )        



# performance - .buffer is massively expensive on complex geometries (e.g. car accessibility isochrones) 
# taking a comparable amount of time / cpu as the entire call to the OTP server. 
# 
# To put it in context if .to_crs takes 1 unit of time, .simplify takes 10 units of time, .make_valid takes 50 units of time
# .buffer takes 100 units of time, the entire call to the OTP server takes 200 units of elapsed time.
#
# so we can put in a lot of logic/ exception handlers and still have much better performance if we avoid calls to .buffer, and call buffer after .simplify
#
def run_batch(batch_args: list[dict]) -> dict:
    return run_batchInternal( 1, batch_args )[1]
    
def run_batchInternal(stackDepth: int, batch_args: list[dict]) -> tuple[int, dict]:

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

        logger.info("T%d:Processing %s for destination %s at time %s", threadId, mode, name, travel_time)
        
        travelTimes.add(datetime.datetime.fromisoformat(travel_time))

        #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
        logger.debug("T%d:Getting URL %s", threadId, url)
        requestData = api_call_retry( url, 3, 10 )

        requestData = parse_to_geo(requestData)

        batchResponses = pd.concat([batchResponses, requestData])

    for var in _isochroneVariables:
        batchResponses[var] = None

    # Remove rows with no geometry
    batchResponses = batchResponses.dropna(subset=["geometry"])
    # and remove a different kind of empty
    batchResponses = batchResponses[~batchResponses.geometry.is_empty]
    
    if len(batchResponses) <= 0:
        logger.warn("T%d: no valid geometry returned for %s for destination %s", threadId, mode, name)
        
        if stackDepth<=1:
            #if we snap a location to the middle of a highway / major roundabout, and then ask OTP to route from there, it may decide it's
            #and inaccessible location for pedestrians. If this happens we need to slightly adjust the location and try again.
            logger.warn("T%d: adjusting location and trying again for %s for destination %s", threadId, mode, name)           
            travel_time_matrix = run_batchAdjustLocationInternal( stackDepth, batch_args )
        else:
            travel_time_matrix = dict()
    else:    
        batchResponses.geometry = batchResponses.geometry.to_crs(_PROCESSING_CRS) #transform to GB national grid so that we can work in metre units.
            # discuss if this is better worse or indifferent to EPSG:23030") which is what the original code used
        try:
            #boundingBox =  batchData.geometry.envelope
            batchResponses = doGeometryProcessing( batchResponses, travelTimes )

        except Exception as e:
            logger.warn("T%d:Exception during geometry processing %s for destination %s (%s) - trying again...", threadId, mode, name, e)
            
            batchResponses.geometry = batchResponses.geometry.make_valid()
            
            # simplifying the geometry reduces some to being empty
            batchResponses = batchResponses.dropna(subset=["geometry"])
            batchResponses = batchResponses[~batchResponses.geometry.is_empty]
            
            if len(batchResponses) > 0:
                batchResponses = doGeometryProcessing( batchResponses, travelTimes )
            
            logger.info("T%d:Geometry processing successful after make_valid() %s for destination %s", threadId, mode, name)
            
        
        if len(batchResponses) > 0:
            #batchResponses.geometry = batchResponses.geometry.to_crs(_CENTROIDS_CRS) #transform back to lat/long.
            
            travel_time_matrix = saveIsochronesAndGenerateMatrix( batchResponses, travelTimes, destination, name, mode, arrive )
        else:  
            logger.warn("T%d: no valid geometry returned for %s for destination %s", threadId, mode, name)
            travel_time_matrix = dict()        
    
    logger.info("T%d:Completing %s for destination %s", threadId, mode, name)
        
    return len(batchResponses), travel_time_matrix



_locationAdjustmentAmount: float = 0.0003

_boxSearchPattern: list[tuple[int, int]] = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1)]

def run_batchAdjustLocationInternal(stackDepth: int, batch_args: list[dict]) -> dict:

    #http://laptop11266:8888/otp/traveltime/isochrone?location=52.40411109957709,-1.508189400000908&modes=WALK,TRANSIT&time=2023-11-20T08:45:00%2B00:00&arriveby=true&cutoff=45M&cutoff=90M

    threadId = threading.get_ident()
   
    stackDepth = stackDepth + 1

    for boxSize in range(1, 3):
        
        for search in _boxSearchPattern:

            batch_argsCopy = list()

            for request_args in batch_args:
        
                url = operator.itemgetter("url")(request_args)

                match = re.search("location=(.*?),(.*?)&", url)

                if match:
                    lat = float(match.group(1))
                    long = float(match.group(2))
                else:
                    logger.error("T%d: Pattern not found in url. (%s)", threadId, url)
                    break
            
                long = long + ( _locationAdjustmentAmount * boxSize * search[0] )
                lat = lat + ( _locationAdjustmentAmount * boxSize * search[1] )

                updatedUrl = url.replace("=" + match.group(1) + ",", "=" + str(lat) + ",")
                updatedUrl = updatedUrl.replace("," + match.group(2) + "&", "," + str(long) + "&")

                request_argsCopy = request_args.copy()
                request_argsCopy['url'] = updatedUrl
            
                batch_argsCopy.append( request_argsCopy )

            numGeometryResults, travel_time_matrix = run_batchInternal( stackDepth, batch_argsCopy )
        
            if numGeometryResults > 0:
                logger.info("T%d: created non-null geometry after adjusting location. depth(%i) direction(%i,%i)", threadId, boxSize, search[0], search[1] )
                return travel_time_matrix

    logger.info("T%d: failed to create non-null geometry after adjusting location.", threadId )

    return dict()



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

    if len(_centroids.origins) > 0 :
        #despite the buffer - the isochrone generated can have 'noise' in the form of islands of inaccessibility - be cautious and clip to bounding box not isochrone itself
        possible_matches_index = list(_centroids_origin_sindex.query(largest.envelope, predicate="intersects")[1])   
        origins = _centroids.origins.iloc[possible_matches_index]
        origins = origins[ origins.intersects( largest.envelope.unary_union ) ]
        origins = origins.assign(travel_time="")

    if len(travelTimes) > 1:
        #travelTimeForFilename = min(travelTimes).isoformat() + "_and_" + max(travelTimes).isoformat()
        byString = _TIME_BETWEEN_STRING
        fromDateTime = min(travelTimes).isoformat()
        toDateTime = max(travelTimes).isoformat()
    elif travelTimes:
        #travelTimeForFilename =  next(iter(travelTimes))
        byString = _TIME_BY_STRING
        fromDateTime = next(iter(travelTimes)).isoformat()
        toDateTime = None

    travel_time_matrix = pd.DataFrame()

    # Calculate all possible origins within travel time by minutes
    for i in range(batchResponses.shape[0]):
        
        row = batchResponses.iloc[[i]]
        
        journey_time = int(row.time.iloc[0])
        
        logger.debug("T%d:Journey time %i", threadId, journey_time)
        
        '''
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
        '''

        row['buffer']=_buffer_size
        row['mode']=mode
        row['zone']=name
        row['arriveDepart']="Arrive" if arrive else "Depart"
        row['ByBetween']=byString
        row['fromDateTime']=fromDateTime
        row['toDateTime']=toDateTime
        row['travelTime']=journey_time / 60

        modename_safe_for_filename = mode.replace(",","_")

        filename = str(uuid.uuid4()).replace('-', '') + '-' + modename_safe_for_filename + '-' + str(int(journey_time / 60)) + '.geojson'                            

        # Write isochrone
        if (_fanout_directory):
            directory = os.path.join(_output_dir, "mode="+modename_safe_for_filename, "traveltime="+str(int(row['travelTime'].iloc[0])) )
        else:
            directory = _output_dir

        with open(os.path.join(directory, filename), "w") as f:
            write_valid_json(row,f)

        if len(_centroids.origins) > 0 :
            #despite all the make_valid etc, we very occasionally get a TopologyException: side location conflict       
            try:
                uu = row.unary_union
                covered = origins[ origins.intersects( uu.envelope ) ]
                covered = covered.clip( uu )
                
            except Exception as e:
                logger.warn("T%d: exception while clipping %s for %s (%s) trying again... ", threadId, mode, name, e )
                batchResponses.at[row.index[0], 'geometry'] = row.geometry.make_valid().iloc[0]
                row = batchResponses.iloc[[i]]
            
                covered = origins.clip(row)
            
            
            logger.debug(
                "T%d:Mode %s for %s covers %i centroids in %i seconds",
                threadId,
                mode,
                name,
                len(covered),
                int(row.time.iloc[0]),
            )
        
            if _test_origin_areas:
                travel_time_matrix = pd.concat([travel_time_matrix, pd.DataFrame({
                        "OriginName": covered[_name_key],
                        "DestinationName": name,
                        "Mode": mode,
                        "Minutes": journey_time / 60,
                        "OriginCoverFraction": round( covered.geometry.area / covered['originArea'], 4 )
                    })],
                ignore_index=True)
            
            else:
                updated_times = pd.DataFrame( {"travel_time": journey_time}, index=covered.index )
                origins.update(updated_times)


    if not _test_origin_areas and len(_centroids.origins) > 0 :
        #since we clipped to bounding box we may have some inaccessible origins - remove them.
        origins = origins[origins.travel_time != ""]

        travel_time_matrix = pd.DataFrame(
            {
                "OriginName": origins[_name_key],
                "DestinationName": name,
                "Mode": mode,
                "Minutes": origins.travel_time / 60
            }
        )


    # Drop duplicate source / destination
#    travel_time_matrix = travel_time_matrix[
#        ~(travel_time_matrix["OriginName"] == travel_time_matrix["DestinationName"])
#    ]

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
            
 