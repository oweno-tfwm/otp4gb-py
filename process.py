import logging
import os
import sys
from otp4gb.otp import Server

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')

    opt_date = '09-10-2019'
    opt_time = '8:00am'
    opt_no_server = True

    # Start OTP Server
    server = Server(opt_base_folder)
    if not opt_no_server:
        server.start()

    # Load Northern MSOAs

    # Filter MSOAs by bounding box

    # For each filtered MSOA add to Origins and Destinations

    # Define acceptable modes
    modes = [
        'WALK',
        'CAR',
        'TRANSIT,WALK',
        'BUS,WALK',
        'BICYCLE'
    ]

    # Create output directory
    isochrones_dir = os.path.join(opt_base_folder, 'isochrones')
    if not os.path.exists(isochrones_dir):
        os.makedirs(isochrones_dir)

    def process_location(location, filename_pattern):
        for mode in modes:
            geojson_file = os.path.join(
                isochrones_dir, filename_pattern.format(mode=mode))
            result = get_isochrone(location,
                                   date=opt_date,
                                   time=opt_time,
                                   mode=mode,
                                   max_travel_time=180
                                   )
            logger.debug("Writing to %s", geojson_file)
            with open(geojson_file, 'w') as f:
                f.write(result)

    def get_isochrone(location, date, time, mode, max_travel_time=90, isochrone_step=15):
        cutoffs = [('cutoffSec', str(c*60))
                   for c in range(isochrone_step, max_travel_time+1, isochrone_step)]
        query = [
            ('fromPlace', ','.join([str(x) for x in location])),
            ('mode', mode),
            ('date', date),
            ('time', time),
            ('maxWalkDistance', '25000'),
            ('arriveby', 'false'),
        ] + cutoffs
        return server.send_request(path='isochrone', query=query)

    # Calculate ISOChrones from Bradford (as a debug)
    process_location([53.79456, -1.75197], "BradfordIsochones_{mode}.geojson")

    # For each destination
    #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
    #   Write isochrone
    #   Calculate all possible origins within travel time by minutes
    #

    # Stop OTP Server
    server.stop()


if __name__ == '__main__':
    main()
