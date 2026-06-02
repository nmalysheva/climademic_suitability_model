# src/data_pipeline/worldwide/extract_land_use.py
import numpy as np
import pandas as pd
import rasterio as rs
from shapely import box

from src.data_pipeline.common import get_file, get_grid, GEOD, calculate_area, implace_nan

def calculate_window_bounds(lon, lat, grid_step):
    left = lon - grid_step / 2
    right = lon + grid_step / 2
                
    bottom = lat - grid_step / 2
    top = lat + grid_step/ 2
    return left, bottom, right, top

def process_window(index, lon, lat, year, hilada_raster, categories, grid_resolution):
    print(index/1036800)
    result = {
            "longitude": lon,
            "latitude": lat,
            "year": year,
            "geometry": None,
            "area_total_km": np.nan,
            "area_land_km": np.nan
        }
    
    valid_cat = categories.keys() - {"None"} #all categories, except "None/NaN"
    for category in valid_cat:
        result[category] = np.nan

    minx, miny, maxx, maxy = calculate_window_bounds(lon, lat, grid_resolution)
    window = rs.windows.from_bounds(minx, miny, maxx, maxy, hilada_raster.transform)
    lulc_data = hilada_raster.read(1, window=window)
    lulc_data = implace_nan(lulc_data, categories["None"]) #replace 99 with nan values

    total_pixels = np.sum(~np.isnan(lulc_data))
    if total_pixels > 0 :
        lulc_polygon = box(minx, miny, maxx, maxy)
        result["geometry"] = lulc_polygon

        area_total, _ = GEOD.geometry_area_perimeter(lulc_polygon)
        result["area_total_km"] = area_total
        result["area_land_km"]  = calculate_area(lulc_data, [categories["Water"], categories["Ocean"]], hilada_raster.transform)
        
        for category in valid_cat:
                result[category] = np.sum(lulc_data == categories[category]) / total_pixels

    return result


def extract_land_use_features(lulc_dir_path, year, resolution=0.25):
    hilda_file = get_file (lulc_dir_path, f"*_{year}_*")
    grid = get_grid(-180, 180, -90, 90, resolution) # get coordinates of the centers of the cells

    categories_hilda = {
        "Ocean": 0, 
        "Urban": 11,
        "Cropland": 22, 
        "Pasture":33, 
        "Forest": 44, 
        "Shrub":  55,  
        "Barren": 66, 
        "Water": 77, 
        "None": 99}
    
    with rs.open(hilda_file) as hilada_raster:
        results = [ process_window(index, lon, lat, year, hilada_raster, categories_hilda, resolution)
                   for index, lon, lat in grid ]
        df_lulc = pd.DataFrame(results)
        
    return df_lulc
