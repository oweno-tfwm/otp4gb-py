"""Run OTP4GB-py and produce cost matrices."""
from __future__ import annotations

import argparse
import atexit
import datetime
from json import load
import logging
import pathlib

from pydantic import dataclasses
import pydantic
from otp4gb import centroids

from otp4gb.centroids import load_centroids, ZoneCentroidColumns, ZoneCentroids
from otp4gb.config import ASSET_DIR, ProcessConfig, load_config
from otp4gb.logging import configure_app_logging
from otp4gb.otp import Server
from otp4gb.util import Timer
from otp4gb import cost, parameters


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProcessArgs:
    """Arguments for process script.

    Attributes
    ----------
    folder : pathlib.Path
        Path to inputs directory.
    save_parameters : bool, default False
        If true saves build parameters and
        exit.
    """

    folder: pydantic.DirectoryPath
    save_parameters: bool = False

    @classmethod
    def parse(cls) -> ProcessArgs:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "folder", type=pathlib.Path, help="folder containing config file and inputs"
        )
        parser.add_argument(
            "-p",
            "--save_parameters",
            action="store_true",
            help="save build parameters to JSON lines files and exit",
        )

        parsed_args = parser.parse_args()
        return ProcessArgs(**vars(parsed_args))


def loadCentroids(config :ProcessConfig) -> ZoneCentroids:

    logger.info("Loading centroids")

    # Check if config.destination_centroids has been supplied
    if config.destination_centroids is None:
        destination_centroids_path = None
        logger.info("No destination centroids detected. Proceeding with %s", config.centroids )
    else:
        destination_centroids_path = pathlib.Path(ASSET_DIR) / config.destination_centroids

    centroids = load_centroids(
        pathlib.Path(ASSET_DIR) / config.centroids,
        destination_centroids_path,
        # TODO(MB) Read parameters for config to define column names
        zone_columns=ZoneCentroidColumns(),
        extents=config.extents,
    )

    logger.info("Considering %d centroids", len(centroids.origins))

    return centroids



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

    for time_period in config.time_periods:
        search_window_seconds = None
        if time_period.search_window_minutes is not None:
            search_window_seconds = time_period.search_window_minutes * 60

        travel_datetime = datetime.datetime.combine(
            config.date, time_period.travel_time
        )
        # Assume time is in local timezone
        travel_datetime = travel_datetime.astimezone()
        logger.info(
            "Given date / time is assumed to be in local timezone: %s",
            travel_datetime.tzinfo,
        )

        for modes in config.modes:
            print()  # Empty line space in cmd window
            cost_settings = parameters.CostSettings(
                server_url="http://" + config.hostname + ":" + str(config.port),
                modes=modes,
                datetime=travel_datetime,
                arrive_by=True,
                search_window_seconds=search_window_seconds,
                max_walk_distance=config.max_walk_distance,
                crowfly_max_distance=config.crowfly_max_distance,
            )

            if arguments.save_parameters:
                logger.info(
                    "Building parameters for %s - %s",
                    time_period.name,
                    ", ".join(modes),
                )

                parameters_path = arguments.folder / (
                    f"parameters/{time_period.name}_{'_'.join(modes)}"
                    f"_parameters_{travel_datetime:%Y%m%dT%H%M}.csv"
                )
                parameters_path.parent.mkdir(exist_ok=True)

                parameters.save_calculation_parameters(
                    zones=centroids,
                    settings=cost_settings,
                    output_file=parameters_path,
                    ruc_lookup=config.ruc_lookup,
                    irrelevant_destinations=config.irrelevant_destinations,
                )
                continue

            logger.info(
                "Calculating costs for %s - %s", time_period.name, ", ".join(modes)
            )
            matrix_path = arguments.folder / (
                f"costs/{time_period.name}/"
                f"{'_'.join(modes)}_costs_{travel_datetime:%Y%m%dT%H%M}.csv"
            )
            matrix_path.parent.mkdir(exist_ok=True, parents=True)

            #TODO: Add requested trips here
            jobs = parameters.build_calculation_parameters(
                zones=centroids,
                settings=cost_settings,
                ruc_lookup=config.ruc_lookup,
                irrelevant_destinations=config.irrelevant_destinations,
            )

            cost.build_cost_matrix(
                jobs=jobs,
                matrix_file=matrix_path,
                generalised_cost_parameters=config.generalised_cost_factors,
                aggregation_method=config.iterinary_aggregation_method,
                workers=config.number_of_threads,
            )

    # Stop OTP Server
    server.stop()


if __name__ == "__main__":
    main()
