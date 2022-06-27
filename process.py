import datetime
import geopandas as gpd
import json
import logging
import os
import sys
from otp4gb.config import ASSET_DIR, load_config
from otp4gb.otp import Server

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def main():
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')

    config = load_config(opt_base_folder)

    opt_travel_time = config.get('travel_time')
    opt_buffer_size = config.get('buffer_size', 100)
    opt_modes = config.get('modes')
    opt_no_server = True

    # Start OTP Server
    server = Server(opt_base_folder)
    if not opt_no_server:
        server.start()

    # Load Northern MSOAs

    # Filter MSOAs by bounding box

    # For each filtered MSOA add to Origins and Destinations

    # Create output directory
    isochrones_dir = os.path.join(opt_base_folder, 'isochrones')
    if not os.path.exists(isochrones_dir):
        os.makedirs(isochrones_dir)

    def parse_to_geo(text):
        data = json.loads(text)
        crs = data['crs']['properties']['name']
        data = gpd.GeoDataFrame.from_features(data, crs=crs)
        return data

    def buffer_geometry(data, buffer_size=100):
        new_geom = data.geometry.to_crs('EPSG:23030')
        buffered_geom = new_geom.buffer(buffer_size)
        data.geometry = buffered_geom.to_crs(data.crs).simplify(
            tolerance=0.0001, preserve_topology=True)
        return data

    def process_location(location, location_name, filename_pattern="Buffered{buffer_size}m_IsochroneBy_{mode}_ToWorkplaceZone_{location_name}_ToArriveBy_{arrival_time}_within_{journey_time}minutes.geojson"):
        for mode in opt_modes:
            result = get_isochrone(location,
                                   date=opt_travel_time.date(),
                                   time=opt_travel_time.time(),
                                   mode=mode,
                                   max_travel_time=180
                                   )

            data = parse_to_geo(result)
            data = buffer_geometry(data)

            for i in range(data.shape[0]):
                row = data.iloc[[i]]
                journey_time = row['time']
                geojson_file = os.path.join(
                    isochrones_dir,
                    filename_pattern.format(
                        location_name=location_name,
                        mode=mode,
                        buffer_size=opt_buffer_size,
                        arrival_time=opt_travel_time.isoformat(),
                        journey_time=str(int(journey_time/60)).rjust(4, '_'),
                    )
                )
                with open(geojson_file, 'w') as f:
                    f.write(row.to_json())

    def get_isochrone(location, date, time, mode, max_walk_distance=2500, max_travel_time=90, isochrone_step=15):
        cutoffs = [('cutoffSec', str(c*60))
                   for c in range(isochrone_step, max_travel_time+1, isochrone_step)]
        query = [
            ('fromPlace', ','.join([str(x) for x in location])),
            ('mode', mode),
            ('date', date),
            ('time', time),
            ('maxWalkDistance', str(max_walk_distance)),
            ('arriveby', 'false'),
        ] + cutoffs
        return server.send_request(path='isochrone', query=query)

    # Calculate ISOChrones from Bradford (as a debug)
    process_location([53.79456, -1.75197], location_name='TEST-BRADFORD')

    # For each destination
    #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
    #   Write isochrone
    #   Calculate all possible origins within travel time by minutes
    #

    # Stop OTP Server
    server.stop()


if __name__ == '__main__':
    main()
