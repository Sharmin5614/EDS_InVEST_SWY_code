#!/usr/bin/env python3
"""L1 scenario: land-use/land-cover effects on annual quickflow (1985–1995).

Compare annual quickflow for six subwatersheds between:

* Baseline: 1985 LULC with 1995 climate.
* Scenario: 1995 LULC with 1995 climate.

Holding climate fixed at 1995 isolates the modeled response to LULC change.
For each subwatershed, calculate the mean of valid quickflow raster cells
whose centers fall within the watershed polygon, then report the absolute
and percentage differences (scenario minus baseline).

Output
------
LULC_Scenarios/L1/L1_Quickflow_Comparison.csv

Usage
-----
    python L1_Quickflow_Comparison.py
    python L1_Quickflow_Comparison.py --root "E:\\MYLSU\\1.2_Seasonal_Water_Yield"

Dependencies
------------
    numpy, pandas, geopandas, rasterio

Notes
-----
* Input rasters must have matching CRS, transform, width, and height.
* The shapefile must have six polygons with unique, non-null HUC8 values.
* Raster values are assumed to represent annual quickflow in millimeters.
* This is an unweighted mean of included raster cells, following the
  original analysis; it is not a watershed water-volume calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_ROOT = Path(r"E:\MYLSU\1.2_Seasonal_Water_Yield")
WATERSHED_ID_FIELD = "HUC8"
EXPECTED_WATERSHEDS = 6


def get_input_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """Build input and output paths from the project directory."""
    l1_root = root / "LULC_Scenarios" / "L1"
    baseline_qf = (
        l1_root / "Baseline_1985_LULC_1995_Climate" / "QF_L1_baseline.tif"
    )
    scenario_qf = (
        l1_root / "Scenario_1995_LULC_1995_Climate" / "QF_L1_scenario.tif"
    )
    watersheds_path = root / "AOI" / "final_watershed_area11122.shp"
    output_csv = l1_root / "L1_Quickflow_Comparison.csv"
    return baseline_qf, scenario_qf, watersheds_path, output_csv


def load_watersheds(watersheds_path: Path) -> gpd.GeoDataFrame:
    """Read and validate six uniquely identified subwatersheds."""
    if not watersheds_path.is_file():
        raise FileNotFoundError(f"Missing watershed file: {watersheds_path}")

    watersheds = gpd.read_file(watersheds_path)
    if len(watersheds) != EXPECTED_WATERSHEDS:
        raise ValueError(
            f"Expected {EXPECTED_WATERSHEDS} subwatersheds, "
            f"found {len(watersheds)}."
        )
    if watersheds.crs is None:
        raise ValueError("Watershed shapefile has no CRS.")
    if WATERSHED_ID_FIELD not in watersheds.columns:
        raise ValueError(
            f"Missing ID field '{WATERSHED_ID_FIELD}'. "
            f"Available fields: {watersheds.columns.tolist()}"
        )
    if watersheds[WATERSHED_ID_FIELD].isna().any():
        raise ValueError("Missing HUC8 values.")
    if watersheds[WATERSHED_ID_FIELD].duplicated().any():
        raise ValueError("Duplicate HUC8 values.")
    if watersheds.geometry.isna().any() or watersheds.geometry.is_empty.any():
        raise ValueError("A watershed geometry is missing or empty.")

    return watersheds


def validate_raster_alignment(baseline_path: Path, scenario_path: Path) -> None:
    """Require the two quickflow rasters to share the same spatial grid."""
    for raster_path in (baseline_path, scenario_path):
        if not raster_path.is_file():
            raise FileNotFoundError(f"Missing raster file: {raster_path}")

    with rasterio.open(baseline_path) as baseline, rasterio.open(
        scenario_path
    ) as scenario:
        if baseline.crs is None or scenario.crs is None:
            raise ValueError("Both quickflow rasters must have a CRS.")
        if (
            baseline.crs != scenario.crs
            or baseline.transform != scenario.transform
            or baseline.width != scenario.width
            or baseline.height != scenario.height
        ):
            raise ValueError("Baseline and scenario rasters are not on the same grid.")


def extract_mean_quickflow(
    raster_path: Path, watersheds: gpd.GeoDataFrame
) -> list[float]:
    """Calculate per-watershed mean quickflow using pixel-center inclusion.

    Nodata/masked pixels are excluded. The ``all_touched=False`` rule
    matches the original script: a cell is included when its center falls
    within a polygon.
    """
    means = []
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        projected = watersheds.to_crs(src.crs)
        raster = src.read(1, masked=True)

        for watershed_id, geometry in zip(
            projected[WATERSHED_ID_FIELD], projected.geometry
        ):
            if geometry is None or geometry.is_empty:
                raise ValueError(f"Empty watershed geometry: {watershed_id}")

            inside = geometry_mask(
                [geometry],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
                all_touched=False,
            )
            selected = raster[inside]
            if selected.count() == 0:
                raise ValueError(
                    f"No valid quickflow cells for watershed {watershed_id} "
                    f"in {raster_path.name}."
                )
            means.append(float(selected.mean()))

    return means


def compare_quickflow(
    watersheds: gpd.GeoDataFrame,
    baseline_values: list[float],
    scenario_values: list[float],
) -> pd.DataFrame:
    """Calculate baseline/scenario differences and percent change."""
    results = pd.DataFrame(
        {
            "Subwatershed": watersheds[WATERSHED_ID_FIELD].astype(str).to_numpy(),
            "QF_Baseline_mm": baseline_values,
            "QF_Scenario_mm": scenario_values,
        }
    )
    if "NAME" in watersheds.columns:
        results.insert(1, "Name", watersheds["NAME"].to_numpy())

    results["QF_Change_mm"] = (
        results["QF_Scenario_mm"] - results["QF_Baseline_mm"]
    )
    results["QF_Change_Percent"] = np.where(
        results["QF_Baseline_mm"] != 0,
        100 * results["QF_Change_mm"] / results["QF_Baseline_mm"],
        np.nan,
    )
    return results


def main(root: Path = DEFAULT_ROOT) -> None:
    """Run the complete L1 quickflow comparison."""
    baseline_path, scenario_path, watersheds_path, output_csv = get_input_paths(root)
    watersheds = load_watersheds(watersheds_path)
    validate_raster_alignment(baseline_path, scenario_path)

    baseline_values = extract_mean_quickflow(baseline_path, watersheds)
    scenario_values = extract_mean_quickflow(scenario_path, watersheds)
    results = compare_quickflow(watersheds, baseline_values, scenario_values)

    print("\n" + "=" * 75)
    print("L1: LULC EFFECT ON ANNUAL QUICKFLOW")
    print("1985–1995 | CLIMATE FIXED AT 1995")
    print("=" * 75)
    print(results.round(3).to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"\nResults saved to:\n{output_csv}")
    print("\nL1 QUICKFLOW ANALYSIS COMPLETED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Project root directory (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args()
    main(args.root)
