import geopandas as gpd
import json
import io
from geojson_rewind import rewind
from pandas import DataFrame
from otp4gb.centroids import _CENTROIDS_CRS
from shapely.geometry import shape, GeometryCollection, MultiPolygon


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




def _geometrycollection_to_multipolygon(geometry_collection):
    polygons = [shape(geometry) for geometry in geometry_collection['geometries'] if geometry['type'] == 'Polygon']
    multipolygon = MultiPolygon(polygons)
    
    return multipolygon.__geo_interface__
      


def _round_coordinates(coordinates, minNumCoords):
    rounded_coords = []
    prev_coord = None
    for coord in coordinates:
        rounded_coord = (round(coord[0], 7), round(coord[1], 7))
        if prev_coord != rounded_coord:
            rounded_coords.append(rounded_coord)
            prev_coord = rounded_coord
            
    if len(rounded_coords) < minNumCoords:
        return None
    
    return rounded_coords



def _cleanCoordinates(data):

    for feature in data['features']:
        if 'geometry' in feature:
            ftype = feature['geometry']['type']
            if ftype == 'Polygon':
                out = []
                for idx, ring in enumerate(feature['geometry']['coordinates']):
                    clean = _round_coordinates(ring,4)
                    if not clean is None:
                        out.append(clean)
                        
                if len(out)>0:
                    feature['geometry']['coordinates'] = out
                else:
                    feature['geometry']['coordinates'] = None
                
            elif ftype == 'MultiPolygon':
                for poly_idx, polygon in enumerate(feature['geometry']['coordinates']):
                    out = []
                    for ring_idx, ring in enumerate(polygon):
                        clean = _round_coordinates(ring,4)
                        if not clean is None:
                            out.append(clean)
                            
                    if len(out)>0:
                        feature['geometry']['coordinates'][poly_idx] = out
                    else:
                        feature['geometry']['coordinates'][poly_idx] = None

            elif ftype == 'GeometryCollection':
                for geo_idx, geo in enumerate(feature['geometry']['geometries']):
                    if geo['type'] == 'Polygon':
                        out = []                        
                        for idx, ring in enumerate(geo['coordinates']):                               
                            clean = _round_coordinates(ring,4)
                            if not clean is None:
                                out.append(clean)
                        if len(out)>0:
                            feature['geometry']['geometries'][geo_idx]['coordinates'] = out
                        else:
                            feature['geometry']['geometries'][geo_idx]['coordinates'] = None


                    elif geo['type'] == 'LineString':
                        feature['geometry']['geometries'][geo_idx]['coordinates'] = _round_coordinates( geo['coordinates'],2)                        
                            
                feature['geometry'] = _geometrycollection_to_multipolygon( feature['geometry'] )
            
            elif ftype == 'LineString':
                feature['geometry']['coordinates'] = _round_coordinates( feature['geometry']['coordinates'],2)
                
    return data




def get_valid_json(data :gpd.GeoDataFrame) -> str:
    #rewind throws if geometry is empty
    if data.geometry.is_empty.all():
        return data.to_json( to_wgs84=True )
    else:
        jsonStr = rewind(data.to_json( to_wgs84=True ))
        data = json.loads( jsonStr )
        data = _cleanCoordinates( data )
        return json.dumps( data )


def write_valid_json(data :gpd.GeoDataFrame, f:io.IOBase ):
    #rewind throws if geometry is empty
    if data.geometry.is_empty.all():
        return f.write( data.to_json( to_wgs84=True ) )
    else:
        jsonStr = rewind(data.to_json( to_wgs84=True ))
        data = json.loads( jsonStr )
        data = _cleanCoordinates( data )
        return json.dump( data, f )
