import rasterio as rs
import numpy as np
from shapely import box, Polygon
from pyproj import Geod
import pandas as pd
from pathlib import Path

from src.data_pipeline.common import get_window, get_file

_GEOD = Geod(ellps="WGS84")

def _count_categories(row, raster, categories):

    hilda_window = get_window(row.mask_geometry.bounds, raster)
    transform = raster.window_transform(hilda_window)  # Get transform for this window
    
    #read only part of the data inside the defined window
    data = raster.read(1, window=hilda_window)

    data = np.where(data == 99, np.nan, data) #replace 99 with nan values    
    total_pixels = np.sum(~np.isnan(data))

    result = {str(category): np.nan for category in categories}
    
    if total_pixels > 0: #if it is 0 - do nothing. default values are already nan
        for category in categories:
            result[str(category)] =  (np.sum(data == category) / total_pixels).round(3)
    
    hilda_window_bounds = rs.windows.bounds(hilda_window, raster.transform)
    hilda_polygon = box(*hilda_window_bounds)
    result['hilda_polygon'] = hilda_polygon
    
    valid_mask = (data != 77) & ~np.isnan(data) & (data != 0)
    rows, cols = np.where(valid_mask)
    total_area_km2 = 0.0
    for r, col in zip(rows, cols):
        # Get the coordinates of the cell corners
        lon1, lat1 = transform * (col, r)  # Top-left corner
        lon2, lat2 = transform * (col + 1, r)  # Top-right corner
        lon3, lat3 = transform * (col + 1, r + 1)  # Bottom-right corner
        lon4, lat4 = transform * (col, r + 1)  # Bottom-left corner
        
        # Calculate the area of the cell
        
        polygon = Polygon([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)])
        area, _ = _GEOD.geometry_area_perimeter(polygon)
        total_area_km2 += abs(area) / 1e6 
    result["square_km"] = total_area_km2
    
    return pd.Series(result)

def extract_land_use_features(lulc_dir_path, mask_gdf):

    hilda_gdf = mask_gdf.copy()
    hilda_dir = Path(lulc_dir_path)
    years = np.arange(1975, 2015)

    categories = [0, 11, 22, 33, 44, 55, 66, 77]

    feature_cols = [str(category) for category in categories] + [
        "hilda_polygon",
        "square_km",]

    for category in categories: 
        hilda_gdf[str(category)] = np.nan 
    hilda_gdf["square_km"] = np.nan 
    hilda_gdf["hilda_polygon"] = None
    
    for year in years:
        # for a particular year, find corresponding HILDA+ file. 
        # It is assumed that the filename pattern contains the "_{year}_" part
        # and can be uniquely assigned to a particular year.    
        hilda_fname = get_file(hilda_dir, f"*_{year}_*")
    
        print("Processing year: ", year, ", HILDA+ file  ", hilda_fname.name)
        year_mask = (hilda_gdf["YEAR"] == year)

        if (year_mask.any()):
            with rs.open(hilda_fname) as hilda_raster:
                #hilda_gdf.loc[year_mask] = hilda_gdf.loc[year_mask].apply(lambda row: count_categories(row, hilda_raster, categories), axis=1)
                result = hilda_gdf.loc[year_mask].apply(lambda row: _count_categories(row, hilda_raster, categories), axis=1,)

            hilda_gdf.loc[year_mask, feature_cols] = result[feature_cols]
    return hilda_gdf