import geopandas as gpd
import json


def parse_to_geo(text):
    data = json.loads(text)
    crs = data['crs']['properties']['name']
    data = gpd.GeoDataFrame.from_features(data, crs=crs)
    return data


def buffer_geometry(data, buffer_size=100):
    new_geom = data.geometry.to_crs('EPSG:23030')
    buffered_geom = new_geom.buffer(buffer_size)
    data.geometry = buffered_geom.to_crs(data.crs).simplify(
        tolerance=0.0001, preserve_topology=True)
    return data
