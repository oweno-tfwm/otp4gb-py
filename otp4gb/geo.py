import geopandas as gpd
import json
from geojson_rewind import rewind
from otp4gb.centroids import _CENTROIDS_CRS


def parse_to_geo(text):
    data = json.loads(text)
    data = gpd.GeoDataFrame.from_features(data, crs=_CENTROIDS_CRS)
    return data


def buffer_geometry(data, buffer_size):
    #TODO: add comment why this CRS is used here (suspect it is to use a CRS that is in metres and valid across a wide area)
    new_geom = data.geometry.to_crs("EPSG:23030")
    buffered_geom = new_geom.buffer(buffer_size)
    data.geometry = buffered_geom.to_crs(data.crs).simplify(
        tolerance=0.0001, preserve_topology=True
    )
    return data


def sort_by_descending_time(data):
    data['time'] = data['time'].astype(int)
    return data.sort_values(by="time", ascending=False)


def get_valid_json(data):
    #rewind throws if geometry is empty
    if data.geometry.is_empty.all():
        return data.to_json()
    else:
        return rewind(data.to_json())
