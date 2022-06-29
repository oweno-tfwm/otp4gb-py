
import datetime
import multiprocessing
import operator
import os
import time
import pandas as pd
import sys
from otp4gb.batch import build_run_spec, run_batch, setup_worker
from otp4gb.centroids import load_centroids
from otp4gb.config import ASSET_DIR, load_config
from otp4gb.logging import get_logger
from otp4gb.otp import Server
from otp4gb.util import Timer


logger = get_logger()

FILENAME_PATTERN = "Buffered{buffer_size}m_IsochroneBy_{mode}_ToWorkplaceZone_{location_name}_ToArriveBy_{arrival_time:%Y%m%d_%H%M}_within_{journey_time:_>4n}_mins.geojson"


def main():
    _process_timer = Timer()
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')

    config = load_config(opt_base_folder)

    opt_travel_time = config.get('travel_time')
    opt_buffer_size = config.get('buffer_size', 100)
    opt_modes = config.get('modes')
    opt_centroids_path = os.path.join(ASSET_DIR, config.get('centroids'))
    opt_clip_box = operator.itemgetter(
        'min_lon', 'min_lat', 'max_lon', 'max_lat')(config.get('extents'))

    opt_isochrone_step = 15
    opt_max_travel_time = 90
    opt_max_walk_distance = 2500
    opt_no_server = False
    opt_num_workers = 8
    opt_name_key = 'msoa11nm'

    # Start OTP Server
    server = Server(opt_base_folder)
    if not opt_no_server:
        logger.info('Starting server')
        server.start()

    # Load Northern MSOAs
    logger.info('Loading centroids')
    centroids = load_centroids(opt_centroids_path)

    # Filter MSOAs by bounding box
    centroids = centroids.clip(opt_clip_box)

    # Create output directory
    isochrones_dir = os.path.join(opt_base_folder, 'isochrones')
    if not os.path.exists(isochrones_dir):
        os.makedirs(isochrones_dir)

    logger.info('Building run spec')
    run_spec = build_run_spec(name_key=opt_name_key, modes=opt_modes, centroids=centroids, arrive_by=opt_travel_time,
                              travel_time_max=opt_max_travel_time, travel_time_step=opt_isochrone_step,
                              max_walk_distance=opt_max_walk_distance, server=server)

    workers = multiprocessing.Pool(
        processes=opt_num_workers, initializer=setup_worker, initargs=({
            'output_dir': isochrones_dir,
            'centroids': centroids,
            'buffer_size': opt_buffer_size,
            'FILENAME_PATTERN': FILENAME_PATTERN,
            'name_key': opt_name_key,
        },))

    logger.info('Launching batch processor')
    with workers:
        results = workers.imap_unordered(run_batch, run_spec)
        # Combine result set and write ot output file
        matrix = pd.concat(results, ignore_index=True)

    matrix_filename = os.path.join(
        opt_base_folder, f'MSOAtoMSOATravelTimeMatrix_ToArriveBy_{opt_travel_time.isoformat()}.csv')
    matrix.to_csv(matrix_filename, index=False)

    # Stop OTP Server
    server.stop()

    logger.info('Calculated %s rows in %s', len(matrix), _process_timer)


if __name__ == '__main__':
    main()
