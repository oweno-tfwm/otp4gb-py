
import atexit
import itertools
import logging
import multiprocessing
import operator
import os
import pandas as pd
import sys
from otp4gb.batch import build_run_spec, run_batch, setup_worker, run_batch_catch_errors
from otp4gb.centroids import load_centroids
from otp4gb.config import ASSET_DIR, load_config
from otp4gb.logging import file_handler_factory, get_logger
from otp4gb.otp import Server
from otp4gb.util import Timer, chunker


logger = get_logger()
logger.setLevel(logging.INFO)

FILENAME_PATTERN = "Buffered{buffer_size}m_IsochroneBy_{mode}_ToWorkplaceZone_{location_name}_ToArriveBy_{arrival_time:%Y%m%d_%H%M}_within_{journey_time:_>4n}_mins.geojson"


def main():
    _process_timer = Timer()
    matrix = []

    @atexit.register
    def report_time():
        logger.info('Calculated %s rows of matrix in %s',
                    len(matrix), _process_timer)

    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')
    log_file = file_handler_factory(
        'process.log', os.path.join(opt_base_folder, 'logs'))
    logger.addHandler(log_file)

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
    opt_chunk_size = opt_num_workers * 10
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
    logger.info('Considering %d centroids', len(centroids))

    # Create output directory
    isochrones_dir = os.path.join(opt_base_folder, 'isochrones')
    if not os.path.exists(isochrones_dir):
        os.makedirs(isochrones_dir)

    jobs = build_run_spec(name_key=opt_name_key, modes=opt_modes, centroids=centroids, arrive_by=opt_travel_time,
                          travel_time_max=opt_max_travel_time, travel_time_step=opt_isochrone_step,
                          max_walk_distance=opt_max_walk_distance, server=server)

    matrix_filename = os.path.join(
        opt_base_folder,
        f"MSOAtoMSOATravelTimeMatrix_ToArriveBy_{opt_travel_time.strftime('%Y%m%d_%H%M')}.csv"
    )
    logger.info('Launching batch processor to process %d jobs', len(jobs))

    
    config_dict = {
        'output_dir': isochrones_dir,
        'centroids': centroids,
        'buffer_size': opt_buffer_size,
        'FILENAME_PATTERN': FILENAME_PATTERN,
        'name_key': opt_name_key,
    }
    
    run_multiprocessing = config.get("run_multiprocessing", False)
    if run_multiprocessing:
        workers = multiprocessing.Pool(
            processes=opt_num_workers, initializer=setup_worker, initargs=(config_dict,)
        )

        with workers:
            for idx, batch in enumerate(chunker(jobs, opt_chunk_size)):
                logger.info(
                    "==================== Running batch %d ====================", idx+1)
                logger.info("Dispatching %d jobs", len(batch))
                results = workers.imap_unordered(run_batch_catch_errors, batch)

                # This is a list comprehension which flattens the results
                results = [row for result in results for row in result]
                logger.info("Receiving %d results", len(results))

                # Append to matrix
                matrix = matrix + results

                # Write list to csv
                # TODO Check if this is expensive
                logger.info("Writing %d rows to %s", len(matrix), matrix_filename)
                pd.DataFrame.from_dict(matrix).to_csv(matrix_filename, index=False)
    else:
        setup_worker(config_dict)
        matrix = [run_batch_catch_errors(j) for j in jobs]
        # Flatten into list of dictionaries
        matrix = list(itertools.chain.from_iterable(matrix))

        # Write list to csv
        logger.info("Writing %d rows to %s", len(matrix), matrix_filename)
        pd.DataFrame.from_dict(matrix).to_csv(matrix_filename, index=False)


    # Stop OTP Server
    server.stop()


if __name__ == '__main__':
    main()
