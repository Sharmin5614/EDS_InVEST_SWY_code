#!/usr/bin/env python3
"""Unified InVEST SWY LULC scenario runner and annual quickflow comparison.

Examples:
  python LULC_All_Scenarios.py --scenario all
  python LULC_All_Scenarios.py --scenario L2 --run-model
  python LULC_All_Scenarios.py --scenario L1 --root "E:\\MYLSU\\1.2_Seasonal_Water_Yield"

By default, compares EXISTING quickflow rasters (does not rerun InVEST).
With --run-model, executes baseline and scenario InVEST runs first.
L2-L5 have original simulation inputs and paths; L1 is comparison-only because its InVEST input configuration was not supplied.
"""

import argparse
import csv
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

DEFAULT_ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")
MONTH_NAMES = {name: idx for idx, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
)}

# All paths below reproduce those explicitly provided in the original scripts.
# L1 source contains quickflow extraction only (existing model outputs).
# L2 ET0 is labeled 2005 but its original source path is in the 2015 folder.
# The original spelling "Fiinal_LULC_1985.tif" is deliberately preserved.
SCENARIOS = {
    "L1": {
        "base_year": 1985, "scenario_year": 1995, "climate_year": 1995,
        "baseline_lulc": None, "scenario_lulc": None,
        "precip_folder": None, "et0_folder": None, "rain_events": None,
        "biophysical": None,
    },
    "L2": {
        "base_year": 1995, "scenario_year": 2005, "climate_year": 2005,
        "baseline_lulc": "1995/LULC_1995/Final_LULC_1995.tif",
        "scenario_lulc": "2005/LULC/LULC_Final_2005.tif",
        "precip_folder": "2005/FINAL_PRECIPITATION",
        "et0_folder": "2015/FINAL_EVAPOTRANSPIRATION",  # Original L2 path; verify year
        "rain_events": "2005/Rain_Events_Table.csv",
        "biophysical": "2015/Biophysical_Table.csv",
    },
    "L3": {
        "base_year": 2005, "scenario_year": 2015, "climate_year": 2015,
        "baseline_lulc": "2005/LULC/LULC_Final_2005.tif",
        "scenario_lulc": "2015/LULC/Final_LULC.tif",
        "precip_folder": "2015/FINAL_PRECIPITATION",
        "et0_folder": "2015/FINAL_EVAPOTRANSPIRATION",
        "rain_events": "2015/Rain_Events_Table.csv",
        "biophysical": "2015/Biophysical_Table.csv",
    },
    "L4": {
        "base_year": 2015, "scenario_year": 2023, "climate_year": 2023,
        "baseline_lulc": "2015/LULC/Final_LULC.tif",
        "scenario_lulc": "LULC/NEW1.1_FINAL_LULC.tif",
        "precip_folder": "FINAL_PRECIPITATION",
        "et0_folder": "Final_evapotranspiration_2023",
        "rain_events": "Rainfall_Event_Latest.csv",
        "biophysical": "Biophysical_table_new.csv",
    },
    "L5": {
        "base_year": 1985, "scenario_year": 2023, "climate_year": 2023,
        "baseline_lulc": "1985/LULC/Fiinal_LULC_1985.tif",
        "scenario_lulc": "LULC/NEW1.1_FINAL_LULC.tif",
        "precip_folder": "FINAL_PRECIPITATION",
        "et0_folder": "Final_evapotranspiration_2023",
        "rain_events": "Rainfall_Event_Latest.csv",
        "biophysical": "Biophysical_table_new.csv",
    },
}

# Shared model inputs/parameters: from supplied L2 source.
AOI_REL = "AOI/final_watershed_area11122.shp"
DEM_REL = "DEM/111111Final_DEM.tif"
SOIL_REL = "Soil_Hydrology/Final_soil_hydfology.tif"
MODEL_PARAMETERS = {
    "threshold_flow_accumulation": 1000,
    "alpha_m": 1 / 12,
    "beta_i": 1,
    "gamma": 1,
    "user_defined_local_recharge": False,
    "user_defined_climate_zones": False,
    "monthly_alpha": False,
    "flow_dir_algorithm": "D8",
}


def required_path(root, relative, label):
    if relative is None:
        raise ValueError(f"Unconfigured input: {label}. Edit SCENARIOS above.")
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def identify_month(filename):
    stem = Path(filename).stem.lower()
    # Handle compact monthly filenames (P01, ET0_01, M12) used in L4/L5.
    match = re.fullmatch(
        r"(?:p|precip|et|et0|eto|m|month)[_\-\s]*(0?[1-9]|1[0-2])", stem
    )
    if match:
        return int(match.group(1))
    for name, month in MONTH_NAMES.items():
        if re.search(rf"(?<![a-z]){name}[a-z]*(?![a-z])", stem):
            return month
    match = re.search(
        r"(?:^|[_\-\s])(?:month|mon|m)[_\-\s]*(0?[1-9]|1[0-2])(?=$|[_\-\s])", stem
    )
    if match:
        return int(match.group(1))
    tokens = re.split(r"[_\-\s]+", stem)
    candidates = [int(t) for t in tokens if re.fullmatch(r"0?[1-9]|1[0-2]", t)]
    return candidates[0] if len(candidates) == 1 else None


def monthly_table(folder, target):
    rasters = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if len(rasters) != 12:
        raise ValueError(f"Expected 12 monthly rasters in {folder}; found {len(rasters)}")
    mapping = {}
    for raster in rasters:
        month = identify_month(raster.name)
        if month is None or month in mapping:
            raise ValueError(f"Ambiguous, missing, or duplicate month: {raster.name}")
        mapping[month] = raster
    if set(mapping) != set(range(1, 13)):
        raise ValueError(f"Monthly files do not cover months 1–12: {folder}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("month", "path"))
        writer.writerows((month, str(mapping[month])) for month in range(1, 13))
    return target


def run_invest(root, code, cfg):
    # Lazy import: comparisons from existing QF maps do not need InVEST installed.
    from natcap.invest import seasonal_water_yield

    if cfg["baseline_lulc"] is None:
        raise ValueError(
            f"{code}: original source only supplied quickflow comparison; "
            "no complete model input paths were provided. "
            "Use existing QF rasters without --run-model for L1."
        )
    output = root / "LULC_Scenarios" / code
    base = cfg["base_year"]
    scen = cfg["scenario_year"]
    climate = cfg["climate_year"]
    if any(value is None for value in (base, scen, climate)):
        raise ValueError(f"{code} years have not been configured.")

    paths = {name: required_path(root, cfg[key], f"{code} {name}") for name, key in {
        "baseline LULC": "baseline_lulc", "scenario LULC": "scenario_lulc",
        "precipitation": "precip_folder", "ET0": "et0_folder",
        "rain events": "rain_events", "biophysical table": "biophysical",
    }.items()}
    aoi = required_path(root, AOI_REL, "watersheds")
    dem = required_path(root, DEM_REL, "DEM")
    soil = required_path(root, SOIL_REL, "soil hydrology")
    if not paths["precipitation"].is_dir() or not paths["ET0"].is_dir():
        raise NotADirectoryError("Precipitation and ET0 paths must be folders")

    tables = output / "Input_Tables"
    precip_table = monthly_table(paths["precipitation"], tables / f"precip_{climate}.csv")
    et0_table = monthly_table(paths["ET0"], tables / f"et0_{climate}.csv")
    shared = dict(MODEL_PARAMETERS, aoi_path=str(aoi), dem_raster_path=str(dem),
                  soil_group_path=str(soil), precip_raster_table=str(precip_table),
                  et0_raster_table=str(et0_table), rain_events_table_path=str(paths["rain events"]),
                  biophysical_table_path=str(paths["biophysical table"]))
    runs = [
        (f"Baseline_{base}_LULC_{climate}_Climate", paths["baseline LULC"], f"{code}_baseline"),
        (f"Scenario_{scen}_LULC_{climate}_Climate", paths["scenario LULC"], f"{code}_scenario"),
    ]
    prepared = []
    for name, lulc, suffix in runs:
        args = dict(shared, lulc_raster_path=str(lulc), workspace_dir=str(output / name),
                    results_suffix=suffix)
        issues = seasonal_water_yield.validate(args)
        if issues:
            raise ValueError(f"{code} {name} validation findings: {issues}")
        prepared.append((name, args))
    for name, args in prepared:
        print(f"  Running InVEST: {name}")
        Path(args["workspace_dir"]).mkdir(parents=True, exist_ok=True)
        seasonal_water_yield.execute(args)


def quickflow_paths(root, code, cfg):
    output = root / "LULC_Scenarios" / code
    base, scen, climate = cfg["base_year"], cfg["scenario_year"], cfg["climate_year"]
    if any(v is None for v in (base, scen, climate)):
        raise ValueError(f"{code}: define base_year, scenario_year, climate_year in SCENARIOS.")
    baseline = output / f"Baseline_{base}_LULC_{climate}_Climate" / f"QF_{code}_baseline.tif"
    scenario = output / f"Scenario_{scen}_LULC_{climate}_Climate" / f"QF_{code}_scenario.tif"
    for item in (baseline, scenario):
        if not item.exists():
            raise FileNotFoundError(f"{code}: quickflow raster not found: {item}")
    return baseline, scenario, output


def compare_quickflow(root, code, cfg):
    baseline, scenario, output = quickflow_paths(root, code, cfg)
    aoi = required_path(root, AOI_REL, "six subwatershed polygons")
    watersheds = gpd.read_file(aoi)
    if watersheds.crs is None or len(watersheds) != 6:
        raise ValueError("AOI must have a CRS and exactly six watershed polygons")
    if "HUC8" not in watersheds.columns or watersheds["HUC8"].isna().any() or watersheds["HUC8"].duplicated().any():
        raise ValueError("AOI needs six unique, nonmissing HUC8 identifiers")
    if watersheds.geometry.isna().any() or watersheds.geometry.is_empty.any():
        raise ValueError("Watershed geometries cannot be null or empty")

    def means(src, projected):
        raster = src.read(1, masked=True)
        values = []
        for geom in projected.geometry:
            mask = geometry_mask([geom], out_shape=(src.height, src.width),
                                 transform=src.transform, invert=True, all_touched=False)
            selected = raster[mask]
            if selected.count() == 0:
                raise ValueError(f"No valid raster pixels in watershed for {src.name}")
            values.append(float(selected.mean()))
        return values

    with rasterio.open(baseline) as b, rasterio.open(scenario) as s:
        if (b.crs != s.crs or b.transform != s.transform or
                b.width != s.width or b.height != s.height):
            raise ValueError(f"{code}: baseline and scenario QF rasters have different grids")
        if b.crs is None:
            raise ValueError(f"{code}: quickflow raster has no CRS")
        polygons = watersheds.to_crs(b.crs)
        base_values, scen_values = means(b, polygons), means(s, polygons)

    results = pd.DataFrame({
        "Scenario": code,
        "Subwatershed": watersheds["HUC8"].astype(str).to_numpy(),
        "QF_Baseline_mm": base_values,
        "QF_Scenario_mm": scen_values,
    })
    if "NAME" in watersheds.columns:
        results.insert(2, "Name", watersheds["NAME"].to_numpy())
    results["QF_Change_mm"] = results["QF_Scenario_mm"] - results["QF_Baseline_mm"]
    results["QF_Change_Percent"] = np.where(
        results["QF_Baseline_mm"] != 0,
        results["QF_Change_mm"] / results["QF_Baseline_mm"].replace(0, np.nan) * 100,
        np.nan,
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{code}_Quickflow_Comparison.csv"
    results.to_csv(destination, index=False)
    print(f"  Saved: {destination}")
    print(results.round(3).to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--run-model", action="store_true", help="Run InVEST before analyzing QF rasters (L2-L5 only)")
    args = parser.parse_args()
    targets = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    outputs, failures = [], []
    if args.run_model and "L1" in targets:
        print("NOTE: L1 simulation inputs were not provided; L1 will use existing QF rasters.")
    for code in targets:
        print(f"\n{'=' * 65}\n{code}: LULC QUICKFLOW ANALYSIS\n{'=' * 65}")
        try:
            if args.run_model and code != "L1":
                run_invest(args.root, code, SCENARIOS[code])
            outputs.append(compare_quickflow(args.root, code, SCENARIOS[code]))
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            print(f"  NOT COMPLETED: {exc}")
            failures.append(code)
    if outputs:
        combined = pd.concat(outputs, ignore_index=True)
        path = args.root / "LULC_Scenarios" / "All_LULC_Quickflow_Comparison.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
        print(f"\nCombined completed scenarios saved: {path}")
    if failures:
        print(f"\nIncomplete scenarios: {', '.join(failures)}")
        raise SystemExit(1)
    print("\nRequested scenario analyses completed.")


if __name__ == "__main__":
    main()
