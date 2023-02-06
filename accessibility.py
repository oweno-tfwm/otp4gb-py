# -*- coding: utf-8 -*-
"""Calculates accessibility scores for various land use metrics."""

##### IMPORTS #####
from __future__ import annotations
import argparse

# Standard imports
import logging
import pathlib
from typing import ClassVar
import pandas as pd

import pydantic

# Third party imports

# Local imports
from otp4gb import config_base, cost

##### CONSTANTS #####
LOG = logging.getLogger("otp4gb.accessibility")
COST_INDEX_COLUMNS = [
    f"{i}{j}" for i in ("origin", "destination") for j in ("", "_id", "_zone_system")
]
COST_COLUMNS = ["duration"]

##### CLASSES #####
# TODO(MB) Move class to OTP4GB package and add flexibility
class Log:
    """Initialise root logger for OTP4GB."""

    ROOT_LOGGER: str = "otp4gb"
    LOG_FILE: str = "OTP_accessibility.log"

    def __init__(self, folder: pathlib.Path) -> None:
        """Initialise root logger for OTP4GB.

        Parameters
        ----------
        folder : pathlib.Path
            Folder to save log file to.
        """
        self.logger = logging.getLogger(self.ROOT_LOGGER)
        self.logger.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(
            logging.Formatter(
                "[%(asctime)s - %(levelname)-8.8s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        self.logger.addHandler(sh)

        log_file = folder / self.LOG_FILE
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)-15.15s] [%(levelname)-8.8s] %(message)s"
            )
        )
        self.logger.addHandler(fh)

        self.logger.debug("Initialised %s logger", self.ROOT_LOGGER)
        self.logger.info("Log file saved to: %s", log_file)

    def __enter__(self) -> Log:
        """Method to allow use as context manager."""
        return self

    def __exit__(self, excepType, excepVal, traceback) -> None:
        """Method to allow use as context manager."""
        if excepType is not None or excepVal is not None or traceback is not None:
            self.logger.critical("Oh no a critical error occurred", exc_info=True)
        else:
            self.logger.info("Program completed without any fatal errors")

        self.logger.info("Closing log file")
        logging.shutdown()


class AccessibilityParameters(config_base.BaseConfig):
    """Config class for accessibility script."""

    cost_metrics: list[pydantic.FilePath]
    output_folder: pathlib.Path
    accessibility_time_minutes: int
    landuse_data: list[pydantic.FilePath]
    aggregation_method: cost.AggregationMethod = cost.AggregationMethod.MEAN


class AccessibilityArguments(pydantic.BaseModel):
    """Handles parsing commandline arguments."""

    config_file: pydantic.FilePath

    CONFIG_FILE: ClassVar[str] = "config/accessibility_config.yml"

    @classmethod
    def parse(cls) -> AccessibilityArguments:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "-c",
            "--config",
            help="Path to config file",
            default=pathlib.Path(cls.CONFIG_FILE),
        )

        args = parser.parse_args()

        return AccessibilityArguments(config_file=args.config)


##### FUNCTIONS #####
def load_cost_metrics(
    path: pathlib.Path,
    aggregation_method: cost.AggregationMethod,
    filter_time_mins: int,
) -> pd.DataFrame:
    """Load and filter cost metrics CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Path to cost metrics file.
    aggregation_method : cost.AggregationMethod
        Aggregated duration column to use e.g. 'mean_duration'.
    filter_time_mins : int
        Maximum time (minutes) for OD pairs to include in the cost metrics.

    Returns
    -------
    pd.DataFrame
        Filtered cost durations.
    """
    LOG.info("Loading cost metrics from %s", path)
    duration_column = f"{aggregation_method.lower()}_duration"

    costs = pd.read_csv(
        path,
        index_col=COST_INDEX_COLUMNS,
        usecols=COST_INDEX_COLUMNS + [duration_column],
    )
    mask = costs[duration_column] <= filter_time_mins * 60
    LOG.info(
        "Dropped %s rows with %s duration > %s mins, %s rows remaining",
        len(mask) - mask.sum(),
        aggregation_method,
        filter_time_mins,
        mask.sum(),
    )

    return costs.loc[mask]


def calculate_accessibility(
    landuse_file: pathlib.Path, duration: pd.DataFrame
) -> pd.DataFrame:
    """Calculate aggregated land use for all zones in `duration`.

    Parameters
    ----------
    landuse_file : pathlib.Path
        Path to file containing land use data.
    duration : pd.DataFrame
        Duration to access various zones.

    Returns
    -------
    pd.DataFrame
        Aggregated land use data for zones given in `duration`.

    Raises
    ------
    ValueError
        If zones found in `duration` are missing from
        land use data.
    """
    LOG.info("Calculating accessibility to %s", landuse_file.stem)
    landuse = pd.read_csv(landuse_file, index_col="id")

    index_columns = duration.index.names
    merged = duration.reset_index().merge(
        landuse,
        how="left",
        validate="m:1",
        left_on="destination_id",
        right_index=True,
        indicator=True,
    )
    merged = merged.drop(
        columns=[i for i in index_columns if i.startswith("destination")]
        + duration.columns.tolist()
    )
    merged.insert(0, "num_zones", 1)

    missing = merged["_merge"] != "both"
    if missing.sum() > 0:
        raise ValueError(f"land use data missing for {missing.sum()} rows")

    grouped = merged.groupby([i for i in index_columns if i.startswith("origin")]).sum(
        numeric_only=True
    )
    return grouped


def main(params: AccessibilityParameters) -> None:
    """Calculate aggregated accessibility metrics."""
    for cost_file in params.cost_metrics:
        costs = load_cost_metrics(
            cost_file, params.aggregation_method, params.accessibility_time_minutes
        )

        output_folder = params.output_folder / cost_file.parent.name
        output_folder.mkdir(exist_ok=True)

        for landuse_file in params.landuse_data:
            accessibility = calculate_accessibility(landuse_file, costs)

            output_file = output_folder / f"{cost_file.stem}-{landuse_file.stem}.csv"
            accessibility.to_csv(output_file)
            LOG.info("Written: %s", output_file)


def _run() -> None:
    arguments = AccessibilityArguments.parse()
    parameters = AccessibilityParameters.load_yaml(arguments.config_file)

    parameters.output_folder.mkdir(exist_ok=True, parents=True)

    with Log(parameters.output_folder):
        main(parameters)


##### MAIN #####
if __name__ == "__main__":
    _run()
