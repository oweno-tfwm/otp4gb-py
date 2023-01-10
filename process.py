import atexit
import logging
import os
import pathlib
import sys

from shapely import geometry

from otp4gb.centroids import load_centroids, ZoneCentroidColumns
from otp4gb.config import ASSET_DIR, load_config
from otp4gb.logging import file_handler_factory, get_logger
from otp4gb.otp import Server
from otp4gb.util import Timer
from otp4gb import cost


logger = get_logger()
logger.setLevel(logging.INFO)

FILENAME_PATTERN = (
    "Buffered{buffer_size}m_IsochroneBy_{mode}_ToWorkplaceZone_"
    "{location_name}_ToArriveBy_{arrival_time:%Y%m%d_%H%M}_"
    "within_{journey_time:_>4n}_mins.geojson"
)


def main():
    _process_timer = Timer()

    @atexit.register
    def report_time():
        logger.info("Finished in %s", _process_timer)

    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error("No path provided")
    log_file = file_handler_factory(
        "process.log", os.path.join(opt_base_folder, "logs")
    )
    logger.addHandler(log_file)

    config = load_config(opt_base_folder)

    opt_centroids_path = os.path.join(ASSET_DIR, config.centroids)

    # Start OTP Server
    server = Server(opt_base_folder)
    if not config.no_server:
        logger.info("Starting server")
        server.start()

    # Load Northern MSOAs
    logger.info("Loading centroids")
    # TODO(MB) Read parameters for config to define column names
    centroids_columns = ZoneCentroidColumns()
    centroids = load_centroids(opt_centroids_path, zone_columns=centroids_columns)

    # Filter MSOAs by bounding box
    clip_box = geometry.box(
        config.extents.min_lon,
        config.extents.min_lat,
        config.extents.max_lon,
        config.extents.max_lat,
    )
    centroids = centroids.clip(clip_box)
    logger.info("Considering %d centroids", len(centroids))

    # Build cost matrix
    for mode in config.modes:
        mode = [m.upper().strip() for m in mode.split(",")]
        cost_settings = cost.CostSettings(
            server_url="http://localhost:8080",
            modes=mode,
            datetime=config.travel_time,
            arrive_by=True,
            max_walk_distance=config.max_walk_distance,
        )

        matrix_path = (
            pathlib.Path(opt_base_folder)
            / f"costs/{'_'.join(mode)}_costs_{config.travel_time:%Y%m%dT%H%M}.csv"
        )
        matrix_path.parent.mkdir(exist_ok=True)

        cost.build_cost_matrix(
            centroids,
            centroids_columns,
            cost_settings,
            matrix_path,
            config.number_of_threads,
        )

    # Stop OTP Server
    server.stop()


if __name__ == "__main__":
    main()
