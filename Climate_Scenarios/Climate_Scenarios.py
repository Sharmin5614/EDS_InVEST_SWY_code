#!/usr/bin/env python3
"""InVEST Seasonal Water Yield climate scenarios C1–C5.

Each run preserves the supplied scenario's input files, model parameters,
output folders, quickflow analysis, and optional LULC-baseline comparison.

Run: python Climate_All_Scenarios.py --scenario all
     python Climate_All_Scenarios.py --scenario C3
"""

import argparse


def run_c1():
    # ============================================================
    # C1: CLIMATE SCENARIO (1985–1995)
    # FIXED 1985 LULC AND FIXED 2015 ET0
    # InVEST Seasonal Water Yield 3.20.2
    # ============================================================

    import csv
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio

    from rasterio.features import geometry_mask
    from natcap.invest import seasonal_water_yield


    # ------------------------------------------------------------
    # 1. PROJECT DIRECTORY
    # ------------------------------------------------------------

    ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")


    # ------------------------------------------------------------
    # 2. FIXED 1985 LULC
    # ------------------------------------------------------------

    LULC_1985 = (
        ROOT / "1985" / "LULC" / "Fiinal_LULC_1985.tif"
    )


    # ------------------------------------------------------------
    # 3. CLIMATE INPUTS
    # ------------------------------------------------------------

    # 1985 precipitation
    PRECIP_1985 = (
        ROOT / "1985" / "FINAL_PRECIPITATION"
    )

    # 1995 precipitation
    PRECIP_1995 = (
        ROOT / "1995" / "Precipitation_final"
    )

    # Same 2015 ET0 dataset for both runs
    ET0_FIXED = (
        ROOT / "2015" / "FINAL_EVAPOTRANSPIRATION"
    )

    # 1985 rainfall events
    RAIN_EVENTS_1985 = (
        ROOT / "1985" / "Rain_Events_Table.csv"
    )

    # 1995 rainfall events
    RAIN_EVENTS_1995 = (
        ROOT / "1995" / "Rain_Events_Table.csv"
    )


    # ------------------------------------------------------------
    # 4. BIOPHYSICAL AND SPATIAL INPUTS
    # ------------------------------------------------------------

    BIOPHYSICAL = (
        ROOT / "1985" / "Biophysical_Table.csv"
    )

    DEM = (
        ROOT / "DEM" / "111111Final_DEM.tif"
    )

    SOIL = (
        ROOT / "Soil_Hydrology"
        / "Final_soil_hydfology.tif"
    )

    AOI = (
        ROOT / "AOI"
        / "final_watershed_area11122.shp"
    )


    # ------------------------------------------------------------
    # 5. OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    # New folder preserves previous C1 results
    C1_ROOT = (
        ROOT / "Climate_Scenarios" / "C1_Rerun"
    )

    TABLE_DIR = C1_ROOT / "Input_Tables"

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ------------------------------------------------------------
    # 6. VERIFY INPUT PATHS
    # ------------------------------------------------------------

    inputs = {
        "1985 LULC": LULC_1985,
        "1985 precipitation": PRECIP_1985,
        "1995 precipitation": PRECIP_1995,
        "Fixed ET0": ET0_FIXED,
        "1985 rainfall events": RAIN_EVENTS_1985,
        "1995 rainfall events": RAIN_EVENTS_1995,
        "Biophysical table": BIOPHYSICAL,
        "DEM": DEM,
        "Soil": SOIL,
        "AOI": AOI,
    }

    for name, path in inputs.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    for folder in [
        PRECIP_1985,
        PRECIP_1995,
        ET0_FIXED
    ]:

        if not folder.is_dir():
            raise NotADirectoryError(folder)

    print("All C1 input paths exist.")


    # ------------------------------------------------------------
    # 7. IDENTIFY MONTH FROM FILENAME
    # ------------------------------------------------------------

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }


    def identify_month(filename):

        stem = Path(filename).stem.lower()

        # P1, P01, P12, ET0_01, M01, etc.
        match = re.fullmatch(
            r"(?:p|precip|et|et0|eto|m|month)"
            r"[_\-\s]*(0?[1-9]|1[0-2])",
            stem
        )

        if match:
            return int(match.group(1))

        # January, February, etc.
        for name, month in MONTH_NAMES.items():

            if re.search(
                rf"(?<![a-z]){name}[a-z]*(?![a-z])",
                stem
            ):
                return month

        # Embedded month marker
        match = re.search(
            r"(?:^|[_\-\s])(?:month|mon|m)"
            r"[_\-\s]*(0?[1-9]|1[0-2])"
            r"(?=$|[_\-\s])",
            stem
        )

        if match:
            return int(match.group(1))

        # Example: precip_1995_01
        tokens = re.split(r"[_\-\s]+", stem)

        candidates = [
            int(token)
            for token in tokens
            if re.fullmatch(
                r"0?[1-9]|1[0-2]",
                token
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None


    # ------------------------------------------------------------
    # 8. CREATE MONTHLY CLIMATE TABLES
    # ------------------------------------------------------------

    def create_monthly_table(folder, output_csv):

        rasters = sorted([
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
        ])

        if len(rasters) != 12:
            raise ValueError(
                f"Expected 12 monthly rasters in {folder}; "
                f"found {len(rasters)}:\n"
                + "\n".join(p.name for p in rasters)
            )

        mapping = {}

        for raster in rasters:

            month = identify_month(raster.name)

            if month is None:
                raise ValueError(
                    f"Cannot identify month: {raster.name}"
                )

            if month in mapping:
                raise ValueError(
                    f"Duplicate month {month}: "
                    f"{mapping[month].name} and {raster.name}"
                )

            mapping[month] = raster

        if set(mapping) != set(range(1, 13)):
            raise ValueError(
                "Monthly rasters must cover months 1–12."
            )

        print("\n" + "=" * 60)
        print(f"MONTHLY RASTER MAPPING: {folder}")
        print("=" * 60)

        for month in range(1, 13):
            print(f"{month:02d}: {mapping[month].name}")

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(["month", "path"])

            for month in range(1, 13):
                writer.writerow([
                    month,
                    str(mapping[month])
                ])

        print(f"\nSaved monthly table: {output_csv}")

        return output_csv


    PRECIP_TABLE_1985 = create_monthly_table(
        PRECIP_1985,
        TABLE_DIR / "precip_1985.csv"
    )

    PRECIP_TABLE_1995 = create_monthly_table(
        PRECIP_1995,
        TABLE_DIR / "precip_1995.csv"
    )

    ET0_TABLE_FIXED = create_monthly_table(
        ET0_FIXED,
        TABLE_DIR / "et0_fixed_2015.csv"
    )


    # ------------------------------------------------------------
    # 9. SHARED MODEL PARAMETERS
    # ------------------------------------------------------------

    common_args = {

        "aoi_path": str(AOI),

        "dem_raster_path": str(DEM),

        "soil_group_path": str(SOIL),

        "lulc_raster_path": str(LULC_1985),

        "et0_raster_table": str(ET0_TABLE_FIXED),

        "biophysical_table_path": str(BIOPHYSICAL),

        "threshold_flow_accumulation": 1000,

        "alpha_m": 1 / 12,

        "beta_i": 1,

        "gamma": 1,

        "user_defined_local_recharge": False,

        "user_defined_climate_zones": False,

        "monthly_alpha": False,

        "flow_dir_algorithm": "D8",
    }


    # ------------------------------------------------------------
    # 10. DEFINE C1 RUNS
    # ------------------------------------------------------------

    runs = {

        "Baseline_1985_Climate_1985_LULC": {

            "precip": PRECIP_TABLE_1985,

            "rain": RAIN_EVENTS_1985,

            "suffix": "C1_baseline",
        },

        "Scenario_1995_Climate_1985_LULC": {

            "precip": PRECIP_TABLE_1995,

            "rain": RAIN_EVENTS_1995,

            "suffix": "C1_scenario",
        },
    }


    # ------------------------------------------------------------
    # 11. VALIDATE BOTH RUNS
    # ------------------------------------------------------------

    prepared_runs = []

    for run_name, config in runs.items():

        args = common_args.copy()

        args["precip_raster_table"] = str(
            config["precip"]
        )

        args["rain_events_table_path"] = str(
            config["rain"]
        )

        args["workspace_dir"] = str(
            C1_ROOT / run_name
        )

        args["results_suffix"] = config["suffix"]

        print(f"\nVALIDATING: {run_name}")

        errors = seasonal_water_yield.validate(args)

        if errors:
            raise ValueError(
                f"Validation errors for {run_name}:\n{errors}"
            )

        print("Validation passed.")

        prepared_runs.append((run_name, args))


    # ------------------------------------------------------------
    # 12. EXECUTE BOTH C1 RUNS
    # ------------------------------------------------------------

    for run_name, args in prepared_runs:

        print("\n" + "=" * 60)
        print(f"RUNNING: {run_name}")
        print("=" * 60)

        Path(args["workspace_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        seasonal_water_yield.execute(args)

        print(f"COMPLETED: {run_name}")

    print("\nBOTH C1 SIMULATIONS COMPLETED.")


    # ------------------------------------------------------------
    # 13. EXTRACT ANNUAL QUICKFLOW
    # ------------------------------------------------------------

    watersheds = gpd.read_file(AOI)

    if len(watersheds) != 6:
        raise ValueError(
            f"Expected 6 subwatersheds, "
            f"found {len(watersheds)}"
        )

    if watersheds.crs is None:
        raise ValueError("AOI has no CRS.")

    if "HUC8" not in watersheds.columns:
        raise ValueError("HUC8 field missing.")

    if watersheds["HUC8"].isna().any():
        raise ValueError("Missing HUC8 IDs.")

    if watersheds["HUC8"].duplicated().any():
        raise ValueError("Duplicate HUC8 IDs.")


    baseline_qf = (
        C1_ROOT
        / "Baseline_1985_Climate_1985_LULC"
        / "QF_C1_baseline.tif"
    )

    scenario_qf = (
        C1_ROOT
        / "Scenario_1995_Climate_1985_LULC"
        / "QF_C1_scenario.tif"
    )

    for path in [baseline_qf, scenario_qf]:

        if not path.exists():
            raise FileNotFoundError(path)


    with rasterio.open(baseline_qf) as base:
        with rasterio.open(scenario_qf) as scen:

            if (
                base.crs != scen.crs
                or base.transform != scen.transform
                or base.width != scen.width
                or base.height != scen.height
            ):
                raise ValueError(
                    "C1 quickflow rasters are not aligned."
                )


    def mean_qf_by_watershed(raster_path, polygons):

        means = []

        with rasterio.open(raster_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {raster_path}"
                )

            projected = polygons.to_crs(src.crs)
            raster = src.read(1, masked=True)

            for geom in projected.geometry:

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty watershed geometry."
                    )

                mask = geometry_mask(
                    [geom],
                    transform=src.transform,
                    out_shape=(src.height, src.width),
                    invert=True,
                    all_touched=False
                )

                selected = raster[mask]

                if selected.count() == 0:
                    raise ValueError(
                        f"No valid quickflow cells: {raster_path}"
                    )

                value = float(selected.mean())

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid quickflow mean: {raster_path}"
                    )

                means.append(value)

        return means


    baseline_values = mean_qf_by_watershed(
        baseline_qf,
        watersheds
    )

    scenario_values = mean_qf_by_watershed(
        scenario_qf,
        watersheds
    )


    # ------------------------------------------------------------
    # 14. CALCULATE C1 QUICKFLOW CHANGE
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "Subwatershed": watersheds["HUC8"].astype(str),
        "QF_Baseline_mm": baseline_values,
        "QF_Scenario_mm": scenario_values,
    })

    if "NAME" in watersheds.columns:
        results.insert(
            1,
            "Name",
            watersheds["NAME"].values
        )

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"]
        - results["QF_Baseline_mm"]
    )

    results["QF_Change_Percent"] = np.nan

    valid = results["QF_Baseline_mm"] != 0

    results.loc[valid, "QF_Change_Percent"] = (
        results.loc[valid, "QF_Change_mm"]
        / results.loc[valid, "QF_Baseline_mm"]
    ) * 100


    # ------------------------------------------------------------
    # 15. DISPLAY AND SAVE C1 RESULTS
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("C1: PRECIPITATION AND RAINFALL-EVENT EFFECT")
    print("1985–1995 | 1985 LULC AND 2015 ET0 FIXED")
    print("=" * 75)

    print(results.round(3).to_string(index=False))

    OUTPUT_CSV = (
        C1_ROOT / "C1_Quickflow_Comparison.csv"
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nC1 QUICKFLOW ANALYSIS COMPLETED.")


    # ------------------------------------------------------------
    # 16. OPTIONAL CONSISTENCY CHECK WITH L1
    # ------------------------------------------------------------

    L1_BASELINE_QF = (
        ROOT
        / "LULC_Scenarios"
        / "L1"
        / "Baseline_1985_LULC_1995_Climate"
        / "QF_L1_baseline.tif"
    )

    if L1_BASELINE_QF.exists():

        l1_values = mean_qf_by_watershed(
            L1_BASELINE_QF,
            watersheds
        )

        comparison = pd.DataFrame({
            "Subwatershed": watersheds["HUC8"].astype(str),
            "C1_Scenario_mm": scenario_values,
            "L1_Baseline_mm": l1_values,
        })

        comparison["Difference_mm"] = (
            comparison["C1_Scenario_mm"]
            - comparison["L1_Baseline_mm"]
        )

        print("\nC1 SCENARIO VS L1 BASELINE")
        print(comparison.round(6).to_string(index=False))

        print(
            "\nIf the inputs and parameters match, "
            "these differences should be near zero."
        )

    else:
        print(
            "\nL1 baseline raster not found at the expected path. "
            "Skipping consistency comparison."
        )

def run_c2():
    # ============================================================
    # C2: CLIMATE SCENARIO (1995–2005)
    # FIXED 1995 LULC AND FIXED 2015 ET0
    # InVEST Seasonal Water Yield 3.20.2
    # ============================================================

    import csv
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio

    from rasterio.features import geometry_mask
    from natcap.invest import seasonal_water_yield


    # ------------------------------------------------------------
    # 1. PROJECT DIRECTORY
    # ------------------------------------------------------------

    ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")


    # ------------------------------------------------------------
    # 2. FIXED 1995 LULC
    # ------------------------------------------------------------

    LULC_1995 = (
        ROOT / "1995" / "LULC_1995" / "Final_LULC_1995.tif"
    )


    # ------------------------------------------------------------
    # 3. CLIMATE INPUTS
    # ------------------------------------------------------------

    PRECIP_1995 = ROOT / "1995" / "Precipitation_final"

    PRECIP_2005 = ROOT / "2005" / "FINAL_PRECIPITATION"

    # Same ET0 dataset for both runs
    ET0_FIXED = ROOT / "2015" / "FINAL_EVAPOTRANSPIRATION"

    RAIN_EVENTS_1995 = ROOT / "1995" / "Rain_Events_Table.csv"

    RAIN_EVENTS_2005 = ROOT / "2005" / "Rain_Events_Table.csv"


    # ------------------------------------------------------------
    # 4. BIOPHYSICAL AND SPATIAL INPUTS
    # ------------------------------------------------------------

    # Same biophysical table used in your earlier L2
    BIOPHYSICAL = ROOT / "2015" / "Biophysical_Table.csv"

    DEM = ROOT / "DEM" / "111111Final_DEM.tif"

    SOIL = (
        ROOT / "Soil_Hydrology" / "Final_soil_hydfology.tif"
    )

    AOI = (
        ROOT / "AOI" / "final_watershed_area11122.shp"
    )


    # ------------------------------------------------------------
    # 5. OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    # New folder preserves previous C2 results
    C2_ROOT = ROOT / "Climate_Scenarios" / "C2_Rerun"

    TABLE_DIR = C2_ROOT / "Input_Tables"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------------------------
    # 6. VERIFY ALL INPUT PATHS
    # ------------------------------------------------------------

    inputs = {
        "1995 LULC": LULC_1995,
        "1995 precipitation": PRECIP_1995,
        "2005 precipitation": PRECIP_2005,
        "Fixed ET0": ET0_FIXED,
        "1995 rainfall events": RAIN_EVENTS_1995,
        "2005 rainfall events": RAIN_EVENTS_2005,
        "Biophysical table": BIOPHYSICAL,
        "DEM": DEM,
        "Soil": SOIL,
        "AOI": AOI,
    }

    for name, path in inputs.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    for folder in [
        PRECIP_1995,
        PRECIP_2005,
        ET0_FIXED
    ]:
        if not folder.is_dir():
            raise NotADirectoryError(folder)

    print("All C2 input paths exist.")


    # ------------------------------------------------------------
    # 7. IDENTIFY MONTH FROM RASTER FILENAME
    # ------------------------------------------------------------

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }


    def identify_month(filename):

        stem = Path(filename).stem.lower()

        # Examples: 1.tif, 01.tif, P1.tif, ET0_01.tif
        if re.fullmatch(r"0?[1-9]|1[0-2]", stem):
            return int(stem)

        match = re.fullmatch(
            r"(?:p|precip|et|et0|eto|m|month)"
            r"[_\-\s]*(0?[1-9]|1[0-2])",
            stem
        )

        if match:
            return int(match.group(1))

        # Examples: January.tif, Feb.tif
        for name, month in MONTH_NAMES.items():
            if re.search(
                rf"(?<![a-z]){name}[a-z]*(?![a-z])",
                stem
            ):
                return month

        # Examples: precip_2005_01.tif
        tokens = re.split(r"[_\-\s]+", stem)

        candidates = [
            int(token)
            for token in tokens
            if re.fullmatch(r"0?[1-9]|1[0-2]", token)
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None


    # ------------------------------------------------------------
    # 8. CREATE MONTHLY RASTER TABLES
    # ------------------------------------------------------------

    def create_monthly_table(folder, output_csv):

        rasters = sorted([
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
        ])

        if len(rasters) != 12:
            raise ValueError(
                f"Expected 12 monthly rasters in {folder}, "
                f"found {len(rasters)}:\n"
                + "\n".join(p.name for p in rasters)
            )

        mapping = {}

        for raster in rasters:

            month = identify_month(raster.name)

            if month is None:
                raise ValueError(
                    f"Cannot identify month: {raster.name}"
                )

            if month in mapping:
                raise ValueError(
                    f"Duplicate month {month}: "
                    f"{mapping[month].name} and {raster.name}"
                )

            mapping[month] = raster

        if set(mapping) != set(range(1, 13)):
            raise ValueError(
                "Monthly rasters must cover months 1–12."
            )

        print("\n" + "=" * 60)
        print(f"MONTHLY RASTER MAPPING: {folder}")
        print("=" * 60)

        for month in range(1, 13):
            print(f"{month:02d}: {mapping[month].name}")

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(["month", "path"])

            for month in range(1, 13):
                writer.writerow([
                    month,
                    str(mapping[month])
                ])

        print(f"\nSaved: {output_csv}")

        return output_csv


    PRECIP_TABLE_1995 = create_monthly_table(
        PRECIP_1995,
        TABLE_DIR / "precip_1995.csv"
    )

    PRECIP_TABLE_2005 = create_monthly_table(
        PRECIP_2005,
        TABLE_DIR / "precip_2005.csv"
    )

    ET0_TABLE_FIXED = create_monthly_table(
        ET0_FIXED,
        TABLE_DIR / "et0_fixed_2015.csv"
    )


    # ------------------------------------------------------------
    # 9. COMMON MODEL PARAMETERS
    # ------------------------------------------------------------

    common_args = {

        "aoi_path": str(AOI),

        "dem_raster_path": str(DEM),

        "soil_group_path": str(SOIL),

        "lulc_raster_path": str(LULC_1995),

        "et0_raster_table": str(ET0_TABLE_FIXED),

        "biophysical_table_path": str(BIOPHYSICAL),

        "threshold_flow_accumulation": 1000,

        "alpha_m": 1 / 12,

        "beta_i": 1,

        "gamma": 1,

        "user_defined_local_recharge": False,

        "user_defined_climate_zones": False,

        "monthly_alpha": False,

        "flow_dir_algorithm": "D8",
    }


    # ------------------------------------------------------------
    # 10. DEFINE BASELINE AND SCENARIO
    # ------------------------------------------------------------

    runs = {

        "Baseline_1995_Climate_1995_LULC": {
            "precip": PRECIP_TABLE_1995,
            "rain": RAIN_EVENTS_1995,
            "suffix": "C2_baseline",
        },

        "Scenario_2005_Climate_1995_LULC": {
            "precip": PRECIP_TABLE_2005,
            "rain": RAIN_EVENTS_2005,
            "suffix": "C2_scenario",
        },
    }


    # ------------------------------------------------------------
    # 11. VALIDATE BOTH RUNS
    # ------------------------------------------------------------

    prepared_runs = []

    for run_name, config in runs.items():

        args = common_args.copy()

        args["precip_raster_table"] = str(config["precip"])

        args["rain_events_table_path"] = str(config["rain"])

        args["workspace_dir"] = str(C2_ROOT / run_name)

        args["results_suffix"] = config["suffix"]

        print(f"\nVALIDATING: {run_name}")

        warnings = seasonal_water_yield.validate(args)

        if warnings:
            raise ValueError(
                f"Validation warnings for {run_name}:\n"
                f"{warnings}"
            )

        print("Validation passed.")

        prepared_runs.append((run_name, args))


    # ------------------------------------------------------------
    # 12. RUN InVEST SWY
    # ------------------------------------------------------------

    for run_name, args in prepared_runs:

        print("\n" + "=" * 60)
        print(f"RUNNING: {run_name}")
        print("=" * 60)

        Path(args["workspace_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        seasonal_water_yield.execute(args)

        print(f"COMPLETED: {run_name}")

    print("\nBOTH C2 SIMULATIONS COMPLETED.")


    # ------------------------------------------------------------
    # 13. LOCATE QUICKFLOW OUTPUTS
    # ------------------------------------------------------------

    baseline_qf = (
        C2_ROOT
        / "Baseline_1995_Climate_1995_LULC"
        / "QF_C2_baseline.tif"
    )

    scenario_qf = (
        C2_ROOT
        / "Scenario_2005_Climate_1995_LULC"
        / "QF_C2_scenario.tif"
    )

    for path in [baseline_qf, scenario_qf]:
        if not path.exists():
            raise FileNotFoundError(path)


    # ------------------------------------------------------------
    # 14. EXTRACT MEAN QUICKFLOW BY SUBWATERSHED
    # ------------------------------------------------------------

    watersheds = gpd.read_file(AOI)

    if len(watersheds) != 6:
        raise ValueError(
            f"Expected 6 subwatersheds, found {len(watersheds)}"
        )

    if watersheds.crs is None:
        raise ValueError("AOI has no CRS.")

    if "HUC8" not in watersheds.columns:
        raise ValueError("HUC8 field missing.")

    if watersheds["HUC8"].isna().any():
        raise ValueError("Missing HUC8 IDs.")

    if watersheds["HUC8"].duplicated().any():
        raise ValueError("Duplicate HUC8 IDs.")


    with rasterio.open(baseline_qf) as base:
        with rasterio.open(scenario_qf) as scen:

            if (
                base.crs != scen.crs
                or base.transform != scen.transform
                or base.width != scen.width
                or base.height != scen.height
            ):
                raise ValueError(
                    "Baseline and scenario QF rasters are not aligned."
                )


    def mean_qf_by_watershed(raster_path, polygons):

        means = []

        with rasterio.open(raster_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {raster_path}"
                )

            projected = polygons.to_crs(src.crs)
            raster = src.read(1, masked=True)

            for geom in projected.geometry:

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty watershed geometry."
                    )

                mask = geometry_mask(
                    [geom],
                    transform=src.transform,
                    out_shape=(src.height, src.width),
                    invert=True,
                    all_touched=False
                )

                selected = raster[mask]

                if selected.count() == 0:
                    raise ValueError(
                        f"No valid quickflow cells: {raster_path}"
                    )

                value = float(selected.mean())

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid quickflow mean: {raster_path}"
                    )

                means.append(value)

        return means


    baseline_values = mean_qf_by_watershed(
        baseline_qf,
        watersheds
    )

    scenario_values = mean_qf_by_watershed(
        scenario_qf,
        watersheds
    )


    # ------------------------------------------------------------
    # 15. CALCULATE AND SAVE C2 QUICKFLOW CHANGE
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "Subwatershed": watersheds["HUC8"].astype(str),
        "QF_Baseline_mm": baseline_values,
        "QF_Scenario_mm": scenario_values,
    })

    if "NAME" in watersheds.columns:
        results.insert(
            1,
            "Name",
            watersheds["NAME"].values
        )

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"]
        - results["QF_Baseline_mm"]
    )

    results["QF_Change_Percent"] = np.nan

    valid = results["QF_Baseline_mm"] != 0

    results.loc[valid, "QF_Change_Percent"] = (
        results.loc[valid, "QF_Change_mm"]
        / results.loc[valid, "QF_Baseline_mm"]
    ) * 100


    print("\n" + "=" * 75)
    print("C2: PRECIPITATION AND RAINFALL-EVENT EFFECT")
    print("1995–2005 | 1995 LULC AND 2015 ET0 FIXED")
    print("=" * 75)

    print(results.round(3).to_string(index=False))

    OUTPUT_CSV = C2_ROOT / "C2_Quickflow_Comparison.csv"

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nC2 QUICKFLOW ANALYSIS COMPLETED.")


    # ------------------------------------------------------------
    # 16. OPTIONAL C2 SCENARIO VS L2 BASELINE CHECK
    # ------------------------------------------------------------

    L2_BASELINE_QF = (
        ROOT
        / "LULC_Scenarios"
        / "L2"
        / "Baseline_1995_LULC_2005_Climate"
        / "QF_L2_baseline.tif"
    )

    if L2_BASELINE_QF.exists():

        l2_values = mean_qf_by_watershed(
            L2_BASELINE_QF,
            watersheds
        )

        comparison = pd.DataFrame({
            "Subwatershed": watersheds["HUC8"].astype(str),
            "C2_Scenario_mm": scenario_values,
            "L2_Baseline_mm": l2_values,
        })

        comparison["Difference_mm"] = (
            comparison["C2_Scenario_mm"]
            - comparison["L2_Baseline_mm"]
        )

        print("\nC2 SCENARIO VS L2 BASELINE")
        print(comparison.round(6).to_string(index=False))

    else:
        print(
            "\nL2 baseline raster not found at expected path. "
            "Skipping comparison."
        )

def run_c3():

    # ============================================================
    # C3: CLIMATE SCENARIO (2005–2015)
    # FIXED 2005 LULC AND FIXED 2015 ET0
    # InVEST Seasonal Water Yield 3.20.2
    # ============================================================

    import csv
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio

    from rasterio.features import geometry_mask
    from natcap.invest import seasonal_water_yield


    # ------------------------------------------------------------
    # 1. PROJECT DIRECTORY
    # ------------------------------------------------------------

    ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")


    # ------------------------------------------------------------
    # 2. FIXED 2005 LULC
    # ------------------------------------------------------------

    LULC_2005 = (
        ROOT / "2005" / "LULC" / "LULC_Final_2005.tif"
    )


    # ------------------------------------------------------------
    # 3. CLIMATE INPUTS
    # ------------------------------------------------------------

    PRECIP_2005 = (
        ROOT / "2005" / "FINAL_PRECIPITATION"
    )

    PRECIP_2015 = (
        ROOT / "2015" / "FINAL_PRECIPITATION"
    )

    # Fixed 2015 ET0 in both runs
    ET0_FIXED = (
        ROOT / "2015" / "FINAL_EVAPOTRANSPIRATION"
    )

    RAIN_EVENTS_2005 = (
        ROOT / "2005" / "Rain_Events_Table.csv"
    )

    RAIN_EVENTS_2015 = (
        ROOT / "2015" / "Rain_Events_Table.csv"
    )


    # ------------------------------------------------------------
    # 4. BIOPHYSICAL AND SPATIAL INPUTS
    # ------------------------------------------------------------

    BIOPHYSICAL = (
        ROOT / "2015" / "Biophysical_Table.csv"
    )

    DEM = (
        ROOT / "DEM" / "111111Final_DEM.tif"
    )

    SOIL = (
        ROOT / "Soil_Hydrology"
        / "Final_soil_hydfology.tif"
    )

    AOI = (
        ROOT / "AOI"
        / "final_watershed_area11122.shp"
    )


    # ------------------------------------------------------------
    # 5. OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    # Preserves previous C3 results
    C3_ROOT = (
        ROOT / "Climate_Scenarios" / "C3_Rerun"
    )

    TABLE_DIR = C3_ROOT / "Input_Tables"

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ------------------------------------------------------------
    # 6. VERIFY INPUT PATHS
    # ------------------------------------------------------------

    inputs = {
        "2005 LULC": LULC_2005,
        "2005 precipitation": PRECIP_2005,
        "2015 precipitation": PRECIP_2015,
        "Fixed ET0": ET0_FIXED,
        "2005 rainfall events": RAIN_EVENTS_2005,
        "2015 rainfall events": RAIN_EVENTS_2015,
        "Biophysical table": BIOPHYSICAL,
        "DEM": DEM,
        "Soil": SOIL,
        "AOI": AOI,
    }

    for name, path in inputs.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    for folder in [
        PRECIP_2005,
        PRECIP_2015,
        ET0_FIXED
    ]:
        if not folder.is_dir():
            raise NotADirectoryError(folder)

    print("All C3 input paths exist.")


    # ------------------------------------------------------------
    # 7. IDENTIFY MONTH FROM RASTER FILENAME
    # ------------------------------------------------------------

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }


    def identify_month(filename):

        stem = Path(filename).stem.lower()

        # 1.tif, 01.tif, P1.tif, ET0_01.tif
        if re.fullmatch(r"0?[1-9]|1[0-2]", stem):
            return int(stem)

        match = re.fullmatch(
            r"(?:p|precip|et|et0|eto|m|month)"
            r"[_\-\s]*(0?[1-9]|1[0-2])",
            stem
        )

        if match:
            return int(match.group(1))

        # January.tif, Feb.tif, etc.
        for name, month in MONTH_NAMES.items():

            if re.search(
                rf"(?<![a-z]){name}[a-z]*(?![a-z])",
                stem
            ):
                return month

        # precip_2005_01.tif
        tokens = re.split(r"[_\-\s]+", stem)

        candidates = [
            int(token)
            for token in tokens
            if re.fullmatch(
                r"0?[1-9]|1[0-2]",
                token
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None


    # ------------------------------------------------------------
    # 8. CREATE MONTHLY CLIMATE TABLES
    # ------------------------------------------------------------

    def create_monthly_table(folder, output_csv):

        rasters = sorted([
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
        ])

        if len(rasters) != 12:
            raise ValueError(
                f"Expected 12 monthly rasters in {folder}; "
                f"found {len(rasters)}:\n"
                + "\n".join(p.name for p in rasters)
            )

        mapping = {}

        for raster in rasters:

            month = identify_month(raster.name)

            if month is None:
                raise ValueError(
                    f"Cannot identify month: {raster.name}"
                )

            if month in mapping:
                raise ValueError(
                    f"Duplicate month {month}: "
                    f"{mapping[month].name} and {raster.name}"
                )

            mapping[month] = raster

        if set(mapping) != set(range(1, 13)):
            raise ValueError(
                "Monthly rasters must cover months 1–12."
            )

        print("\n" + "=" * 60)
        print(f"MONTHLY RASTER MAPPING: {folder}")
        print("=" * 60)

        for month in range(1, 13):
            print(f"{month:02d}: {mapping[month].name}")

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(["month", "path"])

            for month in range(1, 13):
                writer.writerow([
                    month,
                    str(mapping[month])
                ])

        print(f"\nSaved monthly table: {output_csv}")

        return output_csv


    PRECIP_TABLE_2005 = create_monthly_table(
        PRECIP_2005,
        TABLE_DIR / "precip_2005.csv"
    )

    PRECIP_TABLE_2015 = create_monthly_table(
        PRECIP_2015,
        TABLE_DIR / "precip_2015.csv"
    )

    ET0_TABLE_FIXED = create_monthly_table(
        ET0_FIXED,
        TABLE_DIR / "et0_fixed_2015.csv"
    )


    # ------------------------------------------------------------
    # 9. SHARED MODEL PARAMETERS
    # ------------------------------------------------------------

    common_args = {

        "aoi_path": str(AOI),

        "dem_raster_path": str(DEM),

        "soil_group_path": str(SOIL),

        "lulc_raster_path": str(LULC_2005),

        "et0_raster_table": str(ET0_TABLE_FIXED),

        "biophysical_table_path": str(BIOPHYSICAL),

        "threshold_flow_accumulation": 1000,

        "alpha_m": 1 / 12,

        "beta_i": 1,

        "gamma": 1,

        "user_defined_local_recharge": False,

        "user_defined_climate_zones": False,

        "monthly_alpha": False,

        "flow_dir_algorithm": "D8",
    }


    # ------------------------------------------------------------
    # 10. DEFINE C3 BASELINE AND SCENARIO
    # ------------------------------------------------------------

    runs = {

        "Baseline_2005_Climate_2005_LULC": {

            "precip": PRECIP_TABLE_2005,

            "rain": RAIN_EVENTS_2005,

            "suffix": "C3_baseline",
        },

        "Scenario_2015_Climate_2005_LULC": {

            "precip": PRECIP_TABLE_2015,

            "rain": RAIN_EVENTS_2015,

            "suffix": "C3_scenario",
        },
    }


    # ------------------------------------------------------------
    # 11. VALIDATE BOTH RUNS
    # ------------------------------------------------------------

    prepared_runs = []

    for run_name, config in runs.items():

        args = common_args.copy()

        args["precip_raster_table"] = str(
            config["precip"]
        )

        args["rain_events_table_path"] = str(
            config["rain"]
        )

        args["workspace_dir"] = str(
            C3_ROOT / run_name
        )

        args["results_suffix"] = config["suffix"]

        print(f"\nVALIDATING: {run_name}")

        warnings = seasonal_water_yield.validate(args)

        if warnings:
            raise ValueError(
                f"Validation warnings for {run_name}:\n"
                f"{warnings}"
            )

        print("Validation passed.")

        prepared_runs.append((run_name, args))


    # ------------------------------------------------------------
    # 12. EXECUTE BOTH C3 RUNS
    # ------------------------------------------------------------

    for run_name, args in prepared_runs:

        print("\n" + "=" * 60)
        print(f"RUNNING: {run_name}")
        print("=" * 60)

        Path(args["workspace_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        seasonal_water_yield.execute(args)

        print(f"COMPLETED: {run_name}")

    print("\nBOTH C3 SIMULATIONS COMPLETED.")


    # ------------------------------------------------------------
    # 13. LOCATE QUICKFLOW OUTPUTS
    # ------------------------------------------------------------

    baseline_qf = (
        C3_ROOT
        / "Baseline_2005_Climate_2005_LULC"
        / "QF_C3_baseline.tif"
    )

    scenario_qf = (
        C3_ROOT
        / "Scenario_2015_Climate_2005_LULC"
        / "QF_C3_scenario.tif"
    )

    for path in [baseline_qf, scenario_qf]:

        if not path.exists():
            raise FileNotFoundError(path)


    # ------------------------------------------------------------
    # 14. EXTRACT MEAN QUICKFLOW BY SUBWATERSHED
    # ------------------------------------------------------------

    watersheds = gpd.read_file(AOI)

    if len(watersheds) != 6:
        raise ValueError(
            f"Expected 6 subwatersheds, "
            f"found {len(watersheds)}"
        )

    if watersheds.crs is None:
        raise ValueError("AOI has no CRS.")

    if "HUC8" not in watersheds.columns:
        raise ValueError("HUC8 field missing.")

    if watersheds["HUC8"].isna().any():
        raise ValueError("Missing HUC8 IDs.")

    if watersheds["HUC8"].duplicated().any():
        raise ValueError("Duplicate HUC8 IDs.")


    with rasterio.open(baseline_qf) as base:
        with rasterio.open(scenario_qf) as scen:

            if (
                base.crs != scen.crs
                or base.transform != scen.transform
                or base.width != scen.width
                or base.height != scen.height
            ):
                raise ValueError(
                    "C3 quickflow rasters are not aligned."
                )


    def mean_qf_by_watershed(raster_path, polygons):

        means = []

        with rasterio.open(raster_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {raster_path}"
                )

            projected = polygons.to_crs(src.crs)
            raster = src.read(1, masked=True)

            for geom in projected.geometry:

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty watershed geometry."
                    )

                mask = geometry_mask(
                    [geom],
                    transform=src.transform,
                    out_shape=(src.height, src.width),
                    invert=True,
                    all_touched=False
                )

                selected = raster[mask]

                if selected.count() == 0:
                    raise ValueError(
                        f"No valid quickflow cells: {raster_path}"
                    )

                value = float(selected.mean())

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid quickflow mean: {raster_path}"
                    )

                means.append(value)

        return means


    baseline_values = mean_qf_by_watershed(
        baseline_qf,
        watersheds
    )

    scenario_values = mean_qf_by_watershed(
        scenario_qf,
        watersheds
    )


    # ------------------------------------------------------------
    # 15. CALCULATE C3 QUICKFLOW CHANGE
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "Subwatershed": watersheds["HUC8"].astype(str),
        "QF_Baseline_mm": baseline_values,
        "QF_Scenario_mm": scenario_values,
    })

    if "NAME" in watersheds.columns:
        results.insert(
            1,
            "Name",
            watersheds["NAME"].values
        )

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"]
        - results["QF_Baseline_mm"]
    )

    results["QF_Change_Percent"] = np.nan

    valid = results["QF_Baseline_mm"] != 0

    results.loc[valid, "QF_Change_Percent"] = (
        results.loc[valid, "QF_Change_mm"]
        / results.loc[valid, "QF_Baseline_mm"]
    ) * 100


    # ------------------------------------------------------------
    # 16. DISPLAY AND SAVE C3 RESULTS
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("C3: PRECIPITATION AND RAINFALL-EVENT EFFECT")
    print("2005–2015 | 2005 LULC AND 2015 ET0 FIXED")
    print("=" * 75)

    print(results.round(3).to_string(index=False))

    OUTPUT_CSV = (
        C3_ROOT / "C3_Quickflow_Comparison.csv"
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nC3 QUICKFLOW ANALYSIS COMPLETED.")


    # ------------------------------------------------------------
    # 17. OPTIONAL C3 SCENARIO VS L3 BASELINE CHECK
    # ------------------------------------------------------------

    L3_BASELINE_QF = (
        ROOT
        / "LULC_Scenarios"
        / "L3"
        / "Baseline_2005_LULC_2015_Climate"
        / "QF_L3_baseline.tif"
    )

    if L3_BASELINE_QF.exists():

        l3_values = mean_qf_by_watershed(
            L3_BASELINE_QF,
            watersheds
        )

        comparison = pd.DataFrame({
            "Subwatershed": watersheds["HUC8"].astype(str),
            "C3_Scenario_mm": scenario_values,
            "L3_Baseline_mm": l3_values,
        })

        comparison["Difference_mm"] = (
            comparison["C3_Scenario_mm"]
            - comparison["L3_Baseline_mm"]
        )

        print("\nC3 SCENARIO VS L3 BASELINE")
        print(comparison.round(6).to_string(index=False))

    else:
        print(
            "\nL3 baseline raster not found at expected path. "
            "Skipping comparison."
        )

def run_c4():
    # ============================================================
    # C4: CLIMATE SCENARIO (2015–2023)
    # FIXED 2015 LULC AND FIXED 2015 ET0
    # InVEST Seasonal Water Yield 3.20.2
    # ============================================================

    import csv
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio

    from rasterio.features import geometry_mask
    from natcap.invest import seasonal_water_yield


    # ------------------------------------------------------------
    # 1. PROJECT DIRECTORY
    # ------------------------------------------------------------

    ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")


    # ------------------------------------------------------------
    # 2. FIXED 2015 LULC
    # ------------------------------------------------------------

    LULC_2015 = ROOT / "2015" / "LULC" / "Final_LULC.tif"


    # ------------------------------------------------------------
    # 3. CLIMATE INPUTS
    # ------------------------------------------------------------

    PRECIP_2015 = ROOT / "2015" / "FINAL_PRECIPITATION"

    PRECIP_2023 = ROOT / "FINAL_PRECIPITATION"

    # Fixed 2015 ET0 for both simulations
    ET0_FIXED = (
        ROOT / "2015" / "FINAL_EVAPOTRANSPIRATION"
    )

    RAIN_EVENTS_2015 = (
        ROOT / "2015" / "Rain_Events_Table.csv"
    )

    RAIN_EVENTS_2023 = (
        ROOT / "Rainfall_Event_Latest.csv"
    )


    # ------------------------------------------------------------
    # 4. BIOPHYSICAL AND SPATIAL INPUTS
    # ------------------------------------------------------------

    # Matches previous L4 biophysical table
    BIOPHYSICAL = ROOT / "Biophysical_table_new.csv"

    DEM = ROOT / "DEM" / "111111Final_DEM.tif"

    SOIL = (
        ROOT / "Soil_Hydrology" / "Final_soil_hydfology.tif"
    )

    AOI = (
        ROOT / "AOI" / "final_watershed_area11122.shp"
    )


    # ------------------------------------------------------------
    # 5. OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    # Separate folder preserves earlier C4 outputs
    C4_ROOT = ROOT / "Climate_Scenarios" / "C4_Rerun"

    TABLE_DIR = C4_ROOT / "Input_Tables"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------------------------
    # 6. VERIFY INPUT PATHS
    # ------------------------------------------------------------

    inputs = {
        "2015 LULC": LULC_2015,
        "2015 precipitation": PRECIP_2015,
        "2023 precipitation": PRECIP_2023,
        "Fixed 2015 ET0": ET0_FIXED,
        "2015 rainfall events": RAIN_EVENTS_2015,
        "2023 rainfall events": RAIN_EVENTS_2023,
        "Biophysical table": BIOPHYSICAL,
        "DEM": DEM,
        "Soil": SOIL,
        "AOI": AOI,
    }

    for name, path in inputs.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    for folder in [
        PRECIP_2015,
        PRECIP_2023,
        ET0_FIXED
    ]:
        if not folder.is_dir():
            raise NotADirectoryError(folder)

    print("All C4 input paths exist.")


    # ------------------------------------------------------------
    # 7. IDENTIFY MONTH FROM RASTER FILENAME
    # ------------------------------------------------------------

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }


    def identify_month(filename):

        stem = Path(filename).stem.lower()

        # 1.tif, 01.tif, P1.tif, ET0_01.tif
        if re.fullmatch(r"0?[1-9]|1[0-2]", stem):
            return int(stem)

        match = re.fullmatch(
            r"(?:p|precip|et|et0|eto|m|month)"
            r"[_\-\s]*(0?[1-9]|1[0-2])",
            stem
        )

        if match:
            return int(match.group(1))

        # January.tif, February.tif, etc.
        for name, month in MONTH_NAMES.items():

            if re.search(
                rf"(?<![a-z]){name}[a-z]*(?![a-z])",
                stem
            ):
                return month

        # precip_2023_01.tif
        tokens = re.split(r"[_\-\s]+", stem)

        candidates = [
            int(token)
            for token in tokens
            if re.fullmatch(
                r"0?[1-9]|1[0-2]",
                token
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None


    # ------------------------------------------------------------
    # 8. CREATE MONTHLY CLIMATE TABLES
    # ------------------------------------------------------------

    def create_monthly_table(folder, output_csv):

        rasters = sorted([
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
        ])

        if len(rasters) != 12:
            raise ValueError(
                f"Expected 12 monthly rasters in {folder}; "
                f"found {len(rasters)}:\n"
                + "\n".join(p.name for p in rasters)
            )

        mapping = {}

        for raster in rasters:

            month = identify_month(raster.name)

            if month is None:
                raise ValueError(
                    f"Cannot identify month: {raster.name}"
                )

            if month in mapping:
                raise ValueError(
                    f"Duplicate month {month}: "
                    f"{mapping[month].name} and {raster.name}"
                )

            mapping[month] = raster

        if set(mapping) != set(range(1, 13)):
            raise ValueError(
                "Monthly rasters must cover months 1–12."
            )

        print("\n" + "=" * 60)
        print(f"MONTHLY RASTER MAPPING: {folder}")
        print("=" * 60)

        for month in range(1, 13):
            print(f"{month:02d}: {mapping[month].name}")

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(["month", "path"])

            for month in range(1, 13):
                writer.writerow([
                    month,
                    str(mapping[month])
                ])

        print(f"\nSaved monthly table: {output_csv}")

        return output_csv


    PRECIP_TABLE_2015 = create_monthly_table(
        PRECIP_2015,
        TABLE_DIR / "precip_2015.csv"
    )

    PRECIP_TABLE_2023 = create_monthly_table(
        PRECIP_2023,
        TABLE_DIR / "precip_2023.csv"
    )

    ET0_TABLE_FIXED = create_monthly_table(
        ET0_FIXED,
        TABLE_DIR / "et0_fixed_2015.csv"
    )


    # ------------------------------------------------------------
    # 9. SHARED MODEL PARAMETERS
    # ------------------------------------------------------------

    common_args = {

        "aoi_path": str(AOI),

        "dem_raster_path": str(DEM),

        "soil_group_path": str(SOIL),

        "lulc_raster_path": str(LULC_2015),

        "et0_raster_table": str(ET0_TABLE_FIXED),

        "biophysical_table_path": str(BIOPHYSICAL),

        "threshold_flow_accumulation": 1000,

        "alpha_m": 1 / 12,

        "beta_i": 1,

        "gamma": 1,

        "user_defined_local_recharge": False,

        "user_defined_climate_zones": False,

        "monthly_alpha": False,

        "flow_dir_algorithm": "D8",
    }


    # ------------------------------------------------------------
    # 10. DEFINE C4 BASELINE AND SCENARIO
    # ------------------------------------------------------------

    runs = {

        "Baseline_2015_Climate_2015_LULC": {

            "precip": PRECIP_TABLE_2015,

            "rain": RAIN_EVENTS_2015,

            "suffix": "C4_baseline",
        },

        "Scenario_2023_Climate_2015_LULC": {

            "precip": PRECIP_TABLE_2023,

            "rain": RAIN_EVENTS_2023,

            "suffix": "C4_scenario",
        },
    }


    # ------------------------------------------------------------
    # 11. VALIDATE BOTH RUNS
    # ------------------------------------------------------------

    prepared_runs = []

    for run_name, config in runs.items():

        args = common_args.copy()

        args["precip_raster_table"] = str(
            config["precip"]
        )

        args["rain_events_table_path"] = str(
            config["rain"]
        )

        args["workspace_dir"] = str(
            C4_ROOT / run_name
        )

        args["results_suffix"] = config["suffix"]

        print(f"\nVALIDATING: {run_name}")

        warnings = seasonal_water_yield.validate(args)

        if warnings:
            raise ValueError(
                f"Validation warnings for {run_name}:\n"
                f"{warnings}"
            )

        print("Validation passed.")

        prepared_runs.append((run_name, args))


    # ------------------------------------------------------------
    # 12. EXECUTE BOTH C4 RUNS
    # ------------------------------------------------------------

    for run_name, args in prepared_runs:

        print("\n" + "=" * 60)
        print(f"RUNNING: {run_name}")
        print("=" * 60)

        Path(args["workspace_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        seasonal_water_yield.execute(args)

        print(f"COMPLETED: {run_name}")

    print("\nBOTH C4 SIMULATIONS COMPLETED.")


    # ------------------------------------------------------------
    # 13. LOCATE QUICKFLOW OUTPUTS
    # ------------------------------------------------------------

    baseline_qf = (
        C4_ROOT
        / "Baseline_2015_Climate_2015_LULC"
        / "QF_C4_baseline.tif"
    )

    scenario_qf = (
        C4_ROOT
        / "Scenario_2023_Climate_2015_LULC"
        / "QF_C4_scenario.tif"
    )

    for path in [baseline_qf, scenario_qf]:

        if not path.exists():
            raise FileNotFoundError(path)


    # ------------------------------------------------------------
    # 14. EXTRACT MEAN QUICKFLOW BY SUBWATERSHED
    # ------------------------------------------------------------

    watersheds = gpd.read_file(AOI)

    if len(watersheds) != 6:
        raise ValueError(
            f"Expected 6 subwatersheds, "
            f"found {len(watersheds)}"
        )

    if watersheds.crs is None:
        raise ValueError("AOI has no CRS.")

    if "HUC8" not in watersheds.columns:
        raise ValueError("HUC8 field missing.")

    if watersheds["HUC8"].isna().any():
        raise ValueError("Missing HUC8 IDs.")

    if watersheds["HUC8"].duplicated().any():
        raise ValueError("Duplicate HUC8 IDs.")


    with rasterio.open(baseline_qf) as base:
        with rasterio.open(scenario_qf) as scen:

            if (
                base.crs != scen.crs
                or base.transform != scen.transform
                or base.width != scen.width
                or base.height != scen.height
            ):
                raise ValueError(
                    "C4 quickflow rasters are not aligned."
                )


    def mean_qf_by_watershed(raster_path, polygons):

        means = []

        with rasterio.open(raster_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {raster_path}"
                )

            projected = polygons.to_crs(src.crs)
            raster = src.read(1, masked=True)

            for geom in projected.geometry:

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty watershed geometry."
                    )

                mask = geometry_mask(
                    [geom],
                    transform=src.transform,
                    out_shape=(src.height, src.width),
                    invert=True,
                    all_touched=False
                )

                selected = raster[mask]

                if selected.count() == 0:
                    raise ValueError(
                        f"No valid quickflow cells: {raster_path}"
                    )

                value = float(selected.mean())

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid quickflow mean: {raster_path}"
                    )

                means.append(value)

        return means


    baseline_values = mean_qf_by_watershed(
        baseline_qf,
        watersheds
    )

    scenario_values = mean_qf_by_watershed(
        scenario_qf,
        watersheds
    )


    # ------------------------------------------------------------
    # 15. CALCULATE C4 QUICKFLOW CHANGE
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "Subwatershed": watersheds["HUC8"].astype(str),
        "QF_Baseline_mm": baseline_values,
        "QF_Scenario_mm": scenario_values,
    })

    if "NAME" in watersheds.columns:
        results.insert(
            1,
            "Name",
            watersheds["NAME"].values
        )

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"]
        - results["QF_Baseline_mm"]
    )

    results["QF_Change_Percent"] = np.nan

    valid = results["QF_Baseline_mm"] != 0

    results.loc[valid, "QF_Change_Percent"] = (
        results.loc[valid, "QF_Change_mm"]
        / results.loc[valid, "QF_Baseline_mm"]
    ) * 100


    # ------------------------------------------------------------
    # 16. DISPLAY AND SAVE C4 RESULTS
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("C4: PRECIPITATION AND RAINFALL-EVENT EFFECT")
    print("2015–2023 | 2015 LULC AND 2015 ET0 FIXED")
    print("=" * 75)

    print(results.round(3).to_string(index=False))

    OUTPUT_CSV = (
        C4_ROOT / "C4_Quickflow_Comparison.csv"
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nC4 QUICKFLOW ANALYSIS COMPLETED.")


    # ------------------------------------------------------------
    # 17. OPTIONAL C4 SCENARIO VS L4 BASELINE CHECK
    # ------------------------------------------------------------

    L4_BASELINE_QF = (
        ROOT
        / "LULC_Scenarios"
        / "L4"
        / "Baseline_2015_LULC_2023_Climate"
        / "QF_L4_baseline.tif"
    )

    if L4_BASELINE_QF.exists():

        l4_values = mean_qf_by_watershed(
            L4_BASELINE_QF,
            watersheds
        )

        comparison = pd.DataFrame({
            "Subwatershed": watersheds["HUC8"].astype(str),
            "C4_Scenario_mm": scenario_values,
            "L4_Baseline_mm": l4_values,
        })

        comparison["Difference_mm"] = (
            comparison["C4_Scenario_mm"]
            - comparison["L4_Baseline_mm"]
        )

        print("\nC4 SCENARIO VS L4 BASELINE")
        print(comparison.round(6).to_string(index=False))

    else:
        print(
            "\nL4 baseline raster not found at expected path. "
            "Skipping comparison."
        )

def run_c5():
    # ============================================================
    # C5: CLIMATE SCENARIO (1985–2023)
    # FIXED 1985 LULC AND FIXED 2015 ET0
    # InVEST Seasonal Water Yield 3.20.2
    # ============================================================

    import csv
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import rasterio

    from rasterio.features import geometry_mask
    from natcap.invest import seasonal_water_yield


    # ------------------------------------------------------------
    # 1. PROJECT DIRECTORY
    # ------------------------------------------------------------

    ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")


    # ------------------------------------------------------------
    # 2. FIXED 1985 LULC
    # ------------------------------------------------------------

    LULC_1985 = (
        ROOT / "1985" / "LULC" / "Fiinal_LULC_1985.tif"
    )


    # ------------------------------------------------------------
    # 3. CLIMATE INPUTS
    # ------------------------------------------------------------

    # Baseline precipitation: 1985
    PRECIP_1985 = (
        ROOT / "1985" / "FINAL_PRECIPITATION"
    )

    # Scenario precipitation: 2023
    PRECIP_2023 = (
        ROOT / "FINAL_PRECIPITATION"
    )

    # Fixed ET0 for both runs
    ET0_FIXED = (
        ROOT / "2015" / "FINAL_EVAPOTRANSPIRATION"
    )

    # Baseline rainfall events: 1985
    RAIN_EVENTS_1985 = (
        ROOT / "1985" / "Rain_Events_Table.csv"
    )

    # Scenario rainfall events: 2023
    RAIN_EVENTS_2023 = (
        ROOT / "Rainfall_Event_Latest.csv"
    )


    # ------------------------------------------------------------
    # 4. BIOPHYSICAL AND SPATIAL INPUTS
    # ------------------------------------------------------------

    # Same biophysical table as earlier L5
    BIOPHYSICAL = ROOT / "Biophysical_table_new.csv"

    DEM = ROOT / "DEM" / "111111Final_DEM.tif"

    SOIL = (
        ROOT / "Soil_Hydrology" / "Final_soil_hydfology.tif"
    )

    AOI = (
        ROOT / "AOI" / "final_watershed_area11122.shp"
    )


    # ------------------------------------------------------------
    # 5. OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    # Preserve previous C5 outputs
    C5_ROOT = ROOT / "Climate_Scenarios" / "C5_Rerun"

    TABLE_DIR = C5_ROOT / "Input_Tables"

    TABLE_DIR.mkdir(parents=True, exist_ok=True)


    # ------------------------------------------------------------
    # 6. VERIFY INPUT PATHS
    # ------------------------------------------------------------

    inputs = {
        "1985 LULC": LULC_1985,
        "1985 precipitation": PRECIP_1985,
        "2023 precipitation": PRECIP_2023,
        "Fixed 2015 ET0": ET0_FIXED,
        "1985 rainfall events": RAIN_EVENTS_1985,
        "2023 rainfall events": RAIN_EVENTS_2023,
        "Biophysical table": BIOPHYSICAL,
        "DEM": DEM,
        "Soil": SOIL,
        "AOI": AOI,
    }

    for name, path in inputs.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name}: {path}"
            )

    for folder in [
        PRECIP_1985,
        PRECIP_2023,
        ET0_FIXED
    ]:
        if not folder.is_dir():
            raise NotADirectoryError(folder)

    print("All C5 input paths exist.")


    # ------------------------------------------------------------
    # 7. IDENTIFY MONTH FROM RASTER FILENAME
    # ------------------------------------------------------------

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }


    def identify_month(filename):

        stem = Path(filename).stem.lower()

        # Examples: 1.tif, 01.tif, P1.tif, ET0_01.tif
        if re.fullmatch(r"0?[1-9]|1[0-2]", stem):
            return int(stem)

        match = re.fullmatch(
            r"(?:p|precip|et|et0|eto|m|month)"
            r"[_\-\s]*(0?[1-9]|1[0-2])",
            stem
        )

        if match:
            return int(match.group(1))

        # January.tif, February.tif, etc.
        for name, month in MONTH_NAMES.items():

            if re.search(
                rf"(?<![a-z]){name}[a-z]*(?![a-z])",
                stem
            ):
                return month

        # Example: precip_2023_01.tif
        tokens = re.split(r"[_\-\s]+", stem)

        candidates = [
            int(token)
            for token in tokens
            if re.fullmatch(
                r"0?[1-9]|1[0-2]",
                token
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None


    # ------------------------------------------------------------
    # 8. CREATE MONTHLY CLIMATE TABLES
    # ------------------------------------------------------------

    def create_monthly_table(folder, output_csv):

        rasters = sorted([
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".tif", ".tiff")
        ])

        if len(rasters) != 12:
            raise ValueError(
                f"Expected 12 monthly rasters in {folder}; "
                f"found {len(rasters)}:\n"
                + "\n".join(p.name for p in rasters)
            )

        mapping = {}

        for raster in rasters:

            month = identify_month(raster.name)

            if month is None:
                raise ValueError(
                    f"Cannot identify month: {raster.name}"
                )

            if month in mapping:
                raise ValueError(
                    f"Duplicate month {month}: "
                    f"{mapping[month].name} and {raster.name}"
                )

            mapping[month] = raster

        if set(mapping) != set(range(1, 13)):
            raise ValueError(
                "Monthly rasters must cover months 1–12."
            )

        print("\n" + "=" * 60)
        print(f"MONTHLY RASTER MAPPING: {folder}")
        print("=" * 60)

        for month in range(1, 13):
            print(f"{month:02d}: {mapping[month].name}")

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)
            writer.writerow(["month", "path"])

            for month in range(1, 13):
                writer.writerow([
                    month,
                    str(mapping[month])
                ])

        print(f"\nSaved monthly table: {output_csv}")

        return output_csv


    PRECIP_TABLE_1985 = create_monthly_table(
        PRECIP_1985,
        TABLE_DIR / "precip_1985.csv"
    )

    PRECIP_TABLE_2023 = create_monthly_table(
        PRECIP_2023,
        TABLE_DIR / "precip_2023.csv"
    )

    ET0_TABLE_FIXED = create_monthly_table(
        ET0_FIXED,
        TABLE_DIR / "et0_fixed_2015.csv"
    )


    # ------------------------------------------------------------
    # 9. SHARED MODEL PARAMETERS
    # ------------------------------------------------------------

    common_args = {

        "aoi_path": str(AOI),

        "dem_raster_path": str(DEM),

        "soil_group_path": str(SOIL),

        "lulc_raster_path": str(LULC_1985),

        "et0_raster_table": str(ET0_TABLE_FIXED),

        "biophysical_table_path": str(BIOPHYSICAL),

        "threshold_flow_accumulation": 1000,

        "alpha_m": 1 / 12,

        "beta_i": 1,

        "gamma": 1,

        "user_defined_local_recharge": False,

        "user_defined_climate_zones": False,

        "monthly_alpha": False,

        "flow_dir_algorithm": "D8",
    }


    # ------------------------------------------------------------
    # 10. DEFINE C5 BASELINE AND SCENARIO
    # ------------------------------------------------------------

    runs = {

        "Baseline_1985_Climate_1985_LULC": {

            "precip": PRECIP_TABLE_1985,

            "rain": RAIN_EVENTS_1985,

            "suffix": "C5_baseline",
        },

        "Scenario_2023_Climate_1985_LULC": {

            "precip": PRECIP_TABLE_2023,

            "rain": RAIN_EVENTS_2023,

            "suffix": "C5_scenario",
        },
    }


    # ------------------------------------------------------------
    # 11. VALIDATE BOTH RUNS
    # ------------------------------------------------------------

    prepared_runs = []

    for run_name, config in runs.items():

        args = common_args.copy()

        args["precip_raster_table"] = str(
            config["precip"]
        )

        args["rain_events_table_path"] = str(
            config["rain"]
        )

        args["workspace_dir"] = str(
            C5_ROOT / run_name
        )

        args["results_suffix"] = config["suffix"]

        print(f"\nVALIDATING: {run_name}")

        warnings = seasonal_water_yield.validate(args)

        if warnings:
            raise ValueError(
                f"Validation warnings for {run_name}:\n"
                f"{warnings}"
            )

        print("Validation passed.")

        prepared_runs.append((run_name, args))


    # ------------------------------------------------------------
    # 12. EXECUTE BOTH C5 RUNS
    # ------------------------------------------------------------

    for run_name, args in prepared_runs:

        print("\n" + "=" * 60)
        print(f"RUNNING: {run_name}")
        print("=" * 60)

        Path(args["workspace_dir"]).mkdir(
            parents=True,
            exist_ok=True
        )

        seasonal_water_yield.execute(args)

        print(f"COMPLETED: {run_name}")

    print("\nBOTH C5 SIMULATIONS COMPLETED.")


    # ------------------------------------------------------------
    # 13. LOCATE QUICKFLOW OUTPUTS
    # ------------------------------------------------------------

    baseline_qf = (
        C5_ROOT
        / "Baseline_1985_Climate_1985_LULC"
        / "QF_C5_baseline.tif"
    )

    scenario_qf = (
        C5_ROOT
        / "Scenario_2023_Climate_1985_LULC"
        / "QF_C5_scenario.tif"
    )

    for path in [baseline_qf, scenario_qf]:

        if not path.exists():
            raise FileNotFoundError(path)


    # ------------------------------------------------------------
    # 14. EXTRACT MEAN QUICKFLOW BY SUBWATERSHED
    # ------------------------------------------------------------

    watersheds = gpd.read_file(AOI)

    if len(watersheds) != 6:
        raise ValueError(
            f"Expected 6 subwatersheds, "
            f"found {len(watersheds)}"
        )

    if watersheds.crs is None:
        raise ValueError("AOI has no CRS.")

    if "HUC8" not in watersheds.columns:
        raise ValueError("HUC8 field missing.")

    if watersheds["HUC8"].isna().any():
        raise ValueError("Missing HUC8 IDs.")

    if watersheds["HUC8"].duplicated().any():
        raise ValueError("Duplicate HUC8 IDs.")


    with rasterio.open(baseline_qf) as base:
        with rasterio.open(scenario_qf) as scen:

            if (
                base.crs != scen.crs
                or base.transform != scen.transform
                or base.width != scen.width
                or base.height != scen.height
            ):
                raise ValueError(
                    "C5 quickflow rasters are not aligned."
                )


    def mean_qf_by_watershed(raster_path, polygons):

        means = []

        with rasterio.open(raster_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Raster has no CRS: {raster_path}"
                )

            projected = polygons.to_crs(src.crs)
            raster = src.read(1, masked=True)

            for geom in projected.geometry:

                if geom is None or geom.is_empty:
                    raise ValueError(
                        "Empty watershed geometry."
                    )

                mask = geometry_mask(
                    [geom],
                    transform=src.transform,
                    out_shape=(src.height, src.width),
                    invert=True,
                    all_touched=False
                )

                selected = raster[mask]

                if selected.count() == 0:
                    raise ValueError(
                        f"No valid quickflow cells: {raster_path}"
                    )

                value = float(selected.mean())

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid quickflow mean: {raster_path}"
                    )

                means.append(value)

        return means


    baseline_values = mean_qf_by_watershed(
        baseline_qf,
        watersheds
    )

    scenario_values = mean_qf_by_watershed(
        scenario_qf,
        watersheds
    )


    # ------------------------------------------------------------
    # 15. CALCULATE C5 QUICKFLOW CHANGE
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "Subwatershed": watersheds["HUC8"].astype(str),
        "QF_Baseline_mm": baseline_values,
        "QF_Scenario_mm": scenario_values,
    })

    if "NAME" in watersheds.columns:
        results.insert(
            1,
            "Name",
            watersheds["NAME"].values
        )

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"]
        - results["QF_Baseline_mm"]
    )

    results["QF_Change_Percent"] = np.nan

    valid = results["QF_Baseline_mm"] != 0

    results.loc[valid, "QF_Change_Percent"] = (
        results.loc[valid, "QF_Change_mm"]
        / results.loc[valid, "QF_Baseline_mm"]
    ) * 100


    # ------------------------------------------------------------
    # 16. DISPLAY AND SAVE C5 RESULTS
    # ------------------------------------------------------------

    print("\n" + "=" * 75)
    print("C5: PRECIPITATION AND RAINFALL-EVENT EFFECT")
    print("1985–2023 | 1985 LULC AND 2015 ET0 FIXED")
    print("=" * 75)

    print(results.round(3).to_string(index=False))

    OUTPUT_CSV = (
        C5_ROOT / "C5_Quickflow_Comparison.csv"
    )

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nC5 QUICKFLOW ANALYSIS COMPLETED.")


    # ------------------------------------------------------------
    # 17. OPTIONAL C5 SCENARIO VS L5 BASELINE CHECK
    # ------------------------------------------------------------

    L5_BASELINE_QF = (
        ROOT
        / "LULC_Scenarios"
        / "L5"
        / "Baseline_1985_LULC_2023_Climate"
        / "QF_L5_baseline.tif"
    )

    if L5_BASELINE_QF.exists():

        l5_values = mean_qf_by_watershed(
            L5_BASELINE_QF,
            watersheds
        )

        comparison = pd.DataFrame({
            "Subwatershed": watersheds["HUC8"].astype(str),
            "C5_Scenario_mm": scenario_values,
            "L5_Baseline_mm": l5_values,
        })

        comparison["Difference_mm"] = (
            comparison["C5_Scenario_mm"]
            - comparison["L5_Baseline_mm"]
        )

        print("\nC5 SCENARIO VS L5 BASELINE")
        print(comparison.round(6).to_string(index=False))

    else:
        print(
            "\nL5 baseline raster not found at expected path. "
            "Skipping comparison."
        )


SCENARIO_RUNNERS = {
    "C1": run_c1,
    "C2": run_c2,
    "C3": run_c3,
    "C4": run_c4,
    "C5": run_c5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the original C1–C5 InVEST SWY climate scenarios."
    )
    parser.add_argument(
        "--scenario", choices=["all", *SCENARIO_RUNNERS], default="all",
        help="Scenario to run (default: all).",
    )
    options = parser.parse_args()
    names = list(SCENARIO_RUNNERS) if options.scenario == "all" else [options.scenario]
    for name in names:
        print("\n" + "=" * 75)
        print(f"STARTING CLIMATE SCENARIO {name}")
        print("=" * 75)
        SCENARIO_RUNNERS[name]()


if __name__ == "__main__":
    main()
