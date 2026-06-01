import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr

def _extract_12_months(x, y, year, var_name, climate_data):
    year = int(year)

    times = pd.date_range(
        start=f"{year}-01-01",
        periods=12,
        freq="MS"
    )

    lon = x
    lat = y

    # df.X is -180..180 but NetCDF is 0..360
    lon = lon if lon >= 0 else lon + 360

    values = climate_data[var_name].sel(
        time=times,
        longitude=lon,
        latitude=lat,
        method="nearest"
    ).values

    return values #np.squeeze(values)

def extract_climate_features(climate_fpath, mask_gdf):

    clima_gdf = mask_gdf.copy()
    year_start = clima_gdf["YEAR"].min()
    year_end = clima_gdf["YEAR"].max()
    clima_variables = ["t2m", "tp", "d2m", "si10"]
    with xr.open_dataset(climate_fpath) as clima_ds:
        clima_ds = clima_ds.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"), expver=1)

        for var_name in clima_variables:
            print(var_name)

            #print(da5.count().item())
            extracted = clima_gdf.apply(lambda row: _extract_12_months(row.X, row.Y, row.YEAR, var_name, clima_ds),axis=1)
            monthly_df = pd.DataFrame(extracted.tolist(),columns=[f"{var_name}_{m:02d}" for m in range(1, 13)],
                                    index=clima_gdf.index)

            clima_gdf = pd.concat([clima_gdf, monthly_df], axis=1)
    return clima_gdf
