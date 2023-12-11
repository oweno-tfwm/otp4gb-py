import geopandas as gpd
import json
from geojson_rewind import rewind
from pandas import DataFrame
from otp4gb.centroids import _CENTROIDS_CRS


_PROCESSING_CRS :str = "EPSG:27700"


def parse_to_geo(text :str) -> gpd.GeoDataFrame:
    data = json.loads(text)
    data = gpd.GeoDataFrame.from_features(data, crs=_CENTROIDS_CRS)
    return data


# performance - buffer is massively expensive on complex geometries (e.g. car accessibility isochrones) 
# taking a comparable amount of time / cpu as the entire call to the OTP server. 
# 
# To put it in context if .to_crs takes 1 unit of time, .simplify takes 10 units of time, 
# .buffer takes 100 units of time, and the entire call to the OTP server takes 200 units of time.
 
def buffer_geometry(data :gpd.GeoDataFrame, buffer_size :int) -> gpd.GeoDataFrame:
    #TODO: add comment why this CRS is used here (suspect it is to use a CRS that is in metres and valid across a wide area)
    new_geom = data.geometry.to_crs( _PROCESSING_CRS ) #"EPSG:23030"
    buffered_geom = new_geom.buffer(buffer_size)
    data.geometry = buffered_geom.to_crs(data.crs).simplify(
        tolerance=0.0001, preserve_topology=True
    )
    return data


def sort_by_descending_time(data :dict) -> dict:
    data['time'] = data['time'].astype(int)
    return data.sort_values(by="time", ascending=False)


def get_valid_json(data :gpd.GeoDataFrame) -> str:
    #rewind throws if geometry is empty
    if data.geometry.is_empty.all():
        return data.to_json( to_wgs84=True )
    else:
        return rewind(data.to_json( to_wgs84=True ))
