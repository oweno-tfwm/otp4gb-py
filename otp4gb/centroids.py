"""Module for loading the zone centroids file."""
from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
from re import I
from typing import NamedTuple, Optional

import geopandas as gpd
import pandas as pd
from shapely import geometry


LOG = logging.getLogger(__name__)
_CENTROIDS_CRS: str = "EPSG:4326"


class Bounds(NamedTuple):
    """Bounding box."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Bounds:
        values = []
        missing = []
        invalid = []
        for nm in cls._fields:
            if nm not in data:
                missing.append(nm)
            else:
                try:
                    values.append(float(data[nm]))
                except ValueError:
                    invalid.append(str(data[nm]))

        if missing:
            raise ValueError(f"missing values: {', '.join(missing)}")
        if invalid:
            raise TypeError(f"invalid values: {', '.join(invalid)}")

        return cls(*values)


class ZoneCentroidColumns(NamedTuple):
    """Names of columns in zone centroids GeoDataFrame."""

    name: str = "zone_name"
    id: str = "zone_id"
    system: str = "zone_system"


@dataclasses.dataclass
class ZoneCentroids:
    """Zone centroids data."""

    columns: ZoneCentroidColumns
    origins: gpd.GeoDataFrame
    destinations: Optional[gpd.GeoDataFrame] = None


def _read_centroids(
    path,
    zone_columns: list[str],
    longitude_column: str,
    latitude_column: str,
    polygon_column: str
) -> gpd.GeoDataFrame:
    """Read centroids CSV and convert to GeoDataFrame."""
    
    # Read the header of the CSV file
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader) 
        
    if polygon_column in header:
        
        data = pd.read_csv(path, usecols=zone_columns + [longitude_column, latitude_column, polygon_column])

        points = data.apply(
            lambda x: geometry.Point([x[longitude_column], x[latitude_column]]), axis=1)

        data['polygon'] = gpd.GeoSeries.from_wkt( data[polygon_column], crs=_CENTROIDS_CRS ).make_valid()
    
        data = gpd.GeoDataFrame(
            data.loc[:, zone_columns + ['polygon'] ], geometry=points.values, crs=_CENTROIDS_CRS )
    else:
        data = pd.read_csv(path, usecols=zone_columns + [longitude_column, latitude_column])
        
        points = data.apply(
            lambda x: geometry.Point([x[longitude_column], x[latitude_column]]), axis=1)
    
        data = gpd.GeoDataFrame(
            data.loc[:, zone_columns ], geometry=points.values, crs=_CENTROIDS_CRS )    

    return data


def _clip(
    centroids: gpd.GeoDataFrame, extents: Optional[Bounds] = None
) -> gpd.GeoDataFrame:
    if extents is None:
        return centroids

    clip_box = geometry.box(
        extents.min_lon,
        extents.min_lat,
        extents.max_lon,
        extents.max_lat,
    )

    return centroids.clip(clip_box)

'''
def _add_centroids_back(
    clipped: gpd.GeoDataFrame,
    original: gpd.GeoDataFrame,
    other_ids: pd.Series,
    id_column: str,
) -> gpd.GeoDataFrame:
    """Add any missing IDs back into `clipped` from `original`.

    Parameters
    ----------
    clipped : gpd.GeoDataFrame
        Centroids data after clipped to bounding box.
    original : gpd.GeoDataFrame
        Centroids before clipping to bounding box,
        should have the same columns as `clipped`.
    other_ids : pd.Series
        IDs which should be added into `clipped` if they're
        missing.
    id_column : str
        Name of the column containing IDs in both GeoDataFrames.

    Returns
    -------
    gpd.GeoDataFrame
        `clipped` with additional centroids from `original`.
    """
    missing_ids = other_ids.loc[~other_ids.isin(clipped[id_column])]
    if len(missing_ids) == 0:
        return clipped

    additional_data = original.loc[original[id_column].isin(missing_ids)]

    return pd.concat([clipped, additional_data])
'''

def load_centroids(
    origins_path: pathlib.Path,
    destinations_path: Optional[pathlib.Path] = None,
    zone_columns: Optional[ZoneCentroidColumns] = None,
    longitude_column: str = "longitude",
    latitude_column: str = "latitude",
    polygon_column: str = "poly_wkt",
    extents: Optional[Bounds] = None,
) -> ZoneCentroids:
    """Load zone centroids CSV file.

    Parameters
    ----------
    origins_path : pathlib.Path
        CSV file containing zone centroids with columns:
        zone_id, zone_name, zone_system, longitude and latitude
    destinations_path : pathlib.Path, optional
        Optional CSV file containing different zone centroids for
        destinations in the same format as the origins CSV.
    zone_columns : ZoneCentroidColumns, optional
        Custom names for the columns containing zone ID data.
    longitude_column : str, default 'longitude'
        Name of the column containing the longtudes.
    latitude_column : str, default 'latitude'
        Name of the column containing the latitudes.
    polygon_column : str, default 'poly_wkt'
        Name of the column containing WKT (well-known-text) describing a polygon. (in lat,long EPSG:4326)
    extents: Bounds, optional
        Boundary box to filter centroids.

    Returns
    -------
    ZoneCentroids
        Centroids data for origins (and possibly destinations)
        with columns: zone_id, zone_name and zone_system.
    """
    if zone_columns is None:
        zone_columns = ZoneCentroidColumns()
    columns = list(zone_columns)

    origins = _read_centroids(origins_path, columns, longitude_column, latitude_column, polygon_column)
    origins_clipped = _clip(origins, extents)

    if destinations_path is None:
        LOG.info("Loaded origin centroids only from: %s", origins_path.name)
        return ZoneCentroids(zone_columns, origins_clipped, None)

    destinations = _read_centroids(destinations_path, columns, longitude_column, latitude_column, polygon_column)
    destinations_clipped = _clip(destinations, extents)

    
    '''
    a whole load of logic to ensure the same number and ID codes for origin and destination 
    - don't really understand why (at least for isochrone calculations), 
    removed it all to allow for rectangular O/D matrix and different ID codes for origin and destination.
    
    if len(origins) != len(destinations):
        raise ValueError(
            f"{len(origins)} origin centroids given but {len(destinations)} "
            "zones given, these should be the same."
        )
        
    origin_ids = origins[zone_columns.id].sort_values()
    destination_ids = destinations[zone_columns.id].sort_values()
    
    if not destination_ids.equals(origin_ids):
        difference = destination_ids != origin_ids
        raise ValueError(
            "Destination centroids does not contain the same zone IDs "
            f"as origin centroids, {difference.sum()} IDs are different."
        )
    '''


    '''
    origin_ids = origins_clipped[zone_columns.id].sort_values()
    destination_ids = destinations_clipped[zone_columns.id].sort_values()
    if not origin_ids.equals(destination_ids):
        origins_clipped = _add_centroids_back(
            origins_clipped, origins, destination_ids, zone_columns.id
        )
        destinations_clipped = _add_centroids_back(
            destinations_clipped, destinations, origin_ids, zone_columns.id
        )
    '''
        
    LOG.info(
        "Loaded origin and destination centroids from: %s and %s",
        origins_path.name,
        destinations_path.name,
    )
    return ZoneCentroids(zone_columns, origins_clipped, destinations_clipped)
