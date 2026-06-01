from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0] # set .../Climademic_Suitability_mdel/ a project root

CONFIG = {
    "paths": {
        "artifacts": PROJECT_ROOT / "artifacts",
        "raw_data": {
            "compendium": PROJECT_ROOT / "artifacts/data_raw/Compendium/aegypti_albopictus.csv", #Kraemer's global compendium dataset
            "climate": PROJECT_ROOT / "artifacts/data_raw/ERA5/era5_monthly_mean_1958-2023.nc", #Path to thr ERA5 files
            "land_use": PROJECT_ROOT / "artifacts/data_raw/HILDA+",
            "population": PROJECT_ROOT / "artifacts/data_raw/GHSL_POP",
            "gmod_inaturalist": PROJECT_ROOT / "artifacts/data_raw/GMOD/inaturalistpoints.csv",
            "gmod_mosquito_alert": PROJECT_ROOT / "artifacts/data_raw/GMOD/mosquitoAlert.csv"
            },
        "processed_data":{
            "training_dataset": PROJECT_ROOT / "artifacts/processed_data/training_dataset/training_dataset.csv",
            "climate_worldwide": PROJECT_ROOT / "artifacts/processed_data/global_dataset/climate/",
            "population_worldwide":PROJECT_ROOT / "artifacts/processed_data/global_dataset/population/",
            "land_use_worldwide": PROJECT_ROOT / "artifacts/processed_data/global_dataset/land_use/",}
    }
}