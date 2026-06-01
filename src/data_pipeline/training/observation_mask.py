# data_pipeline/training_ds/create_mask.py
from pyproj import Geod
from shapely.geometry import box, MultiPolygon
import pandas as pd
import geopandas as gpd
import numpy as np

_GEOD = Geod(ellps="WGS84")

def _get_polygon_offset_km(value):

    mapping = {
        "-999": 5,
        "Less than 10km": 10,
        "Less than 25km": 25,
        "Less than 100km": 100,
        "2": 110,
    }

    return mapping.get(str(value), 110)

def _make_geodetic_box(x, y, offset_km):
    
    half_distance_m = offset_km / 2 * 1e3

    north = _GEOD.fwd(x, y, 0, half_distance_m)
    south = _GEOD.fwd(x, y, 180, half_distance_m)
    east  = _GEOD.fwd(x, y, 90, half_distance_m)
    west  = _GEOD.fwd(x, y, 270, half_distance_m)

    minx = west[0]
    maxx = east[0]
    miny = south[1]
    maxy = north[1]

    geometry = box(minx, miny, maxx, maxy)

    if minx > maxx:
        geometry = MultiPolygon([
            box(minx, miny, 180, maxy),
            box(-180, miny, maxx, maxy)
        ])

    return geometry

def _make_geodetic_mask(row):
    x = row.geometry.x
    y = row.geometry.y

    offset_km = _get_polygon_offset_km(str(row["POLYGON_ADMIN"]))

    return _make_geodetic_box(x, y, offset_km)

def load_compendium_raw(compendium_fpath_csv):
    compendium_data = pd.read_csv(compendium_fpath_csv) 
    compendium_data = compendium_data.loc[~compendium_data["YEAR"].isna()] #filter out those without time information
    compendium_data = compendium_data.loc[compendium_data["YEAR"]!='2006-2008'] #filter out strange timing
    compendium_data["YEAR"] = compendium_data["YEAR"].astype(np.int64)
    compendium_data = compendium_data.loc[compendium_data["YEAR"] > 1974] #we start from 1975
    compendium_gdf = gpd.GeoDataFrame(compendium_data, geometry=gpd.points_from_xy(compendium_data.X, compendium_data.Y),
                                crs='EPSG:4326') # EPSG:4326 is WGS84 Latitude/Longitude
    return compendium_gdf

def create_geodetic_mask(compendium_gdf):
    gdf = compendium_gdf.copy()
    gdf["mask_geometry"] = gdf.apply(_make_geodetic_mask, axis=1)
    gdf = gdf.set_geometry("mask_geometry")
    return gdf

def build_observation_mask(observation_path):
    observation_gdf_raw = load_compendium_raw(observation_path)
    observation_gdf_mask = create_geodetic_mask(observation_gdf_raw)
    return observation_gdf_mask