"""Module for loading the zone centroids file."""

import pathlib
from typing import NamedTuple, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class ZoneCentroidColumns(NamedTuple):
    """Names of columns in zone centroids GeoDataFrame."""

    name: str = "zone_name"
    id: str = "zone_id"
    system: str = "zone_system"


def load_centroids(
    path: pathlib.Path,
    zone_columns: Optional[ZoneCentroidColumns] = None,
    longitude_column: str = "longitude",
    latitude_column: str = "latitude",
) -> gpd.GeoDataFrame:
    """Load zone centroids CSV file.

    Parameters
    ----------
    path : pathlib.Path
        CSV file containing zone centroids with columns:
        zone_id, zone_name, zone_system, longitude and latitude
    zone_columns : ZoneCentroidColumns, optional
        Custom names for the columns containing zone ID data.
    longitude_column : str, default 'longitude'
        Name of the column containing the longitudes.
    latitude_column : str, default 'latitude'
        Name of the column containing the latitudes.

    Returns
    -------
    gpd.GeoDataFrame
        Centroids data with columns: zone_id, zone_name and zone_system.
    """
    if zone_columns is None:
        zone_columns = list(ZoneCentroidColumns())
    else:
        zone_columns = list(zone_columns)

    data = pd.read_csv(path, usecols=zone_columns + [longitude_column, latitude_column])
    geometry = data.apply(
        lambda x: Point([x[longitude_column], x[latitude_column]]), axis=1
    )
    data = gpd.GeoDataFrame(
        data.loc[:, zone_columns],
        geometry=geometry.values,
        crs="EPSG:4326",
    )
    return data
