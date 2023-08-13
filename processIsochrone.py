"""Run OTP4GB-py and produce isochrones."""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime
import logging
import math
import multiprocessing
import concurrent.futures
import threading
import os
import pandas as pd
import pathlib

from pydantic import dataclasses
import pydantic

from otp4gb.centroids import load_centroids, ZoneCentroidColumns
from otp4gb.config import ASSET_DIR, load_config
from otp4gb.otp import Server
from otp4gb.util import Timer, chunker
from otp4gb import cost, parameters, batch
from otp4gb.batch import build_run_spec, setup_worker, build_run_spec, run_batch, run_batch_catch_errors  
from contextlib import ExitStack
import otp4gb.batch
from otp4gb.logging import configure_app_logging
import time
from process import ProcessArgs, loadCentroids

logger = logging.getLogger(__name__)


_FILENAME_PATTERN = (
    "Buffered{buffer_size}m_IsochroneBy_{mode}_ToWorkplaceZone_"
    "{location_name}_To{arrive_depart}By_{arrival_time}_"
    "within_{journey_time:_>4n}_mins.geojson"
)


def main():   
    arguments = ProcessArgs.parse()

    _process_timer = Timer()
    
    configure_app_logging(logging.INFO, dir=arguments.folder / "logs")

    @atexit.register
    def report_time():
        logger.info("Finished in %s", _process_timer)

    config = load_config(arguments.folder)

    server = Server(arguments.folder, hostname=config.hostname, port=config.port)
    if arguments.save_parameters:
        logger.info("Saving OTP request parameters without starting OTP")
    elif not config.no_server:
        logger.info("Starting server")
        server.start()

    centroids = loadCentroids(config)

    # Create output directory
    isochrones_dir = os.path.join(arguments.folder, 'isochrones')
    if not os.path.exists(isochrones_dir):
        os.makedirs(isochrones_dir)

    #use default isochrone config settings if not supplied
    isochroneConfig = config.isochrone_configuration
    if isochroneConfig is None :
        isochroneConfig = parameters.IsochroneConfiguration()


    for time_period in config.time_periods:

        travel_datetime = datetime.datetime.combine(
            config.date, time_period.travel_time
        )
        # Assume time is in local timezone
        travel_datetime = travel_datetime.astimezone()
        logger.info(
            "Given date / time is assumed to be in local timezone: %s",
            travel_datetime.tzinfo,
        )

        jobs = build_run_spec(name_key=isochroneConfig.zone_column, 
                                    modes=config.modes, 
                                    centroids=centroids, 
                                    arrive_by=travel_datetime,
                                    travel_time_max=time_period.search_window_minutes, 
                                    travel_time_step=isochroneConfig.step_minutes,
                                    max_walk_distance=config.max_walk_distance, 
                                    server=server,
                                    arrive=True)

        matrix_filename = os.path.join(
            arguments.folder,
            f"AreatoAreaTravelTimeMatrix_ToArriveBy_{travel_datetime.strftime('%Y%m%d_%H%M')}.csv"
        )
        logger.info('Launching batch processor to process %d jobs', len(jobs))


        threadPool =  concurrent.futures.ThreadPoolExecutor( max_workers=config.number_of_threads )

        #could probably remove this now moved from executing in seperate processes to threads instead
        setup_worker({
                'output_dir': isochrones_dir,
                'centroids': centroids,
                'buffer_size': isochroneConfig.buffer_metres,
                'FILENAME_PATTERN': _FILENAME_PATTERN,
                'name_key': isochroneConfig.zone_column,
            },)

        with open(matrix_filename, 'w') as file:
            pass  # truncate the file and close it again

        with ExitStack() as stack:
            stack.enter_context(threadPool)
            file = stack.enter_context(open(matrix_filename, 'a'))
            first = True

            numChunks = math.ceil(len(jobs) / (config.number_of_threads * 5))

            for idx, batch in enumerate(chunker(jobs, config.number_of_threads*5)):
                logger.info(
                    "==================== Running batch %d of %d ====================", idx+1, numChunks)
                logger.info("Dispatching %d jobs", len(batch))

                taskFutures = [threadPool.submit(run_batch_catch_errors, job) for job in batch]

                matrix = list()
      
                for task in concurrent.futures.as_completed(taskFutures):
                    taskResult = task.result()

                    if taskResult is None:  
                        logger.error("null taskResult object recieved")
                    else:
                        # This is a list comprehension which flattens the results
                        flattened_list = [result 
                                          for result in taskResult 
                                            if result is not None]

                        logger.info("Receiving %d results", len(flattened_list))

                        # Append to matrix
                        matrix = matrix + flattened_list

                # Write list to csv in batches
                logger.info("Writing %d rows to %s", len(matrix), matrix_filename)
                pd.DataFrame.from_dict(matrix).to_csv(file, index=False, header=first, lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
                first=False
                
            logger.info("all tasks complete, shutting down threadpool")
            threadPool.shutdown()


    # Stop OTP Server
    server.stop()


if __name__ == "__main__":
    main()
