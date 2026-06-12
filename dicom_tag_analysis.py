#!/usr/bin/env python3
"""Analyze one-row-per-series MRI DICOM tag CSV files.

The script detects common DICOM tag columns, classifies MRI series into broad
image-condition buckets, derives geometry parameters, and writes summary CSVs,
figures, and a short Markdown report.

Example:
    python3 dicom_tag_analysis.py dicom_series_tags.csv -o dicom_tag_report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_CACHE_DIR = Path(tempfile.gettempdir()) / "dicom_tag_analysis_cache"
try:
    (_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))
except OSError:
    pass

plt = None
sns = None


MISSING_TEXT = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "<na>",
    "unknown",
    "not found",
    "not available",
}

CONDITION_ORDER = [
    "T1",
    "T2",
    "FLAIR",
    "MRA",
    "T2star/SWI",
    "DWI",
    "ADC",
    "Localizer",
    "Other",
]

DIMENSION_ORDER = ["2D", "3D", "Unknown"]

COLUMN_ALIASES = {
    "series_description": [
        "SeriesDescription",
        "Series Description",
        "(0008,103E)",
        "0008,103E",
        "0008103E",
    ],
    "protocol_name": [
        "ProtocolName",
        "Protocol Name",
        "(0018,1030)",
        "0018,1030",
        "00181030",
    ],
    "sequence_name": [
        "SequenceName",
        "Sequence Name",
        "(0018,0024)",
        "0018,0024",
        "00180024",
    ],
    "scanning_sequence": [
        "ScanningSequence",
        "Scanning Sequence",
        "(0018,0020)",
        "0018,0020",
        "00180020",
    ],
    "sequence_variant": [
        "SequenceVariant",
        "Sequence Variant",
        "(0018,0021)",
        "0018,0021",
        "00180021",
    ],
    "scan_options": [
        "ScanOptions",
        "Scan Options",
        "(0018,0022)",
        "0018,0022",
        "00180022",
    ],
    "mr_acquisition_type": [
        "MRAcquisitionType",
        "MR Acquisition Type",
        "(0018,0023)",
        "0018,0023",
        "00180023",
    ],
    "image_type": [
        "ImageType",
        "Image Type",
        "(0008,0008)",
        "0008,0008",
        "00080008",
    ],
    "slice_thickness": [
        "SliceThickness",
        "Slice Thickness",
        "(0018,0050)",
        "0018,0050",
        "00180050",
    ],
    "spacing_between_slices": [
        "SpacingBetweenSlices",
        "Spacing Between Slices",
        "(0018,0088)",
        "0018,0088",
        "00180088",
    ],
    "pixel_spacing": [
        "PixelSpacing",
        "Pixel Spacing",
        "(0028,0030)",
        "0028,0030",
        "00280030",
    ],
    "rows": ["Rows", "(0028,0010)", "0028,0010", "00280010"],
    "columns": ["Columns", "(0028,0011)", "0028,0011", "00280011"],
    "acquisition_matrix": [
        "AcquisitionMatrix",
        "Acquisition Matrix",
        "(0018,1310)",
        "0018,1310",
        "00181310",
    ],
    "repetition_time": [
        "RepetitionTime",
        "Repetition Time",
        "TR",
        "(0018,0080)",
        "0018,0080",
        "00180080",
    ],
    "echo_time": [
        "EchoTime",
        "Echo Time",
        "TE",
        "(0018,0081)",
        "0018,0081",
        "00180081",
    ],
    "inversion_time": [
        "InversionTime",
        "Inversion Time",
        "TI",
        "(0018,0082)",
        "0018,0082",
        "00180082",
    ],
    "flip_angle": [
        "FlipAngle",
        "Flip Angle",
        "(0018,1314)",
        "0018,1314",
        "00181314",
    ],
    "number_of_slices": [
        "NumberOfSlices",
        "Number of Slices",
        "ImagesInAcquisition",
        "Images In Acquisition",
        "(0020,1002)",
        "0020,1002",
        "00201002",
        "(0054,0081)",
        "0054,0081",
        "00540081",
    ],
    "magnetic_field_strength": [
        "MagneticFieldStrength",
        "Magnetic Field Strength",
        "(0018,0087)",
        "0018,0087",
        "00180087",
    ],
    "manufacturer": ["Manufacturer", "(0008,0070)", "0008,0070", "00080070"],
    "body_part": [
        "BodyPartExamined",
        "Body Part Examined",
        "(0018,0015)",
        "0018,0015",
        "00180015",
    ],
    "series_uid": [
        "SeriesInstanceUID",
        "Series Instance UID",
        "(0020,000E)",
        "0020,000E",
        "0020000E",
    ],
}

TEXT_KEYS = [
    "series_description",
    "protocol_name",
    "sequence_name",
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",
    "image_type",
    "body_part",
]

NUMERIC_METRICS = [
    "slice_thickness_mm",
    "slice_spacing_mm",
    "slice_gap_mm",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
    "matrix_rows",
    "matrix_columns",
    "number_of_slices",
    "repetition_time_ms",
    "echo_time_ms",
    "inversion_time_ms",
    "flip_angle_deg",
    "field_strength_t",
    "fov_row_mm",
    "fov_col_mm",
    "voxel_volume_mm3",
]

PLOT_METRICS = [
    ("slice_thickness_mm", "Slice thickness (mm)"),
    ("slice_spacing_mm", "Slice spacing (mm)"),
    ("slice_gap_mm", "Slice gap (mm)"),
    ("pixel_spacing_row_mm", "Pixel spacing row (mm)"),
    ("pixel_spacing_col_mm", "Pixel spacing col (mm)"),
    ("matrix_rows", "Matrix rows"),
    ("matrix_columns", "Matrix columns"),
    ("number_of_slices", "Number of slices"),
    ("repetition_time_ms", "TR (ms)"),
    ("echo_time_ms", "TE (ms)"),
    ("inversion_time_ms", "TI (ms)"),
    ("flip_angle_deg", "Flip angle (deg)"),
]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_TEXT:
        return ""
    return text


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_column_name(col): col for col in df.columns}
    columns: dict[str, str | None] = {}

    for logical_name, aliases in COLUMN_ALIASES.items():
        alias_norms = [normalize_column_name(alias) for alias in aliases]
        found = None

        for alias in alias_norms:
            if alias in normalized:
                found = normalized[alias]
                break

        if found is None:
            searchable_aliases = [alias for alias in alias_norms if len(alias) >= 4]
            for col in df.columns:
                col_norm = normalize_column_name(col)
                if any(alias and alias in col_norm for alias in searchable_aliases):
                    found = col
                    break

        columns[logical_name] = found

    return columns


def extract_numbers(value: object) -> list[float]:
    text = clean_text(value)
    if not text:
        return []
    text = text.replace("\\", " ")
    text = text.replace("x", " ")
    text = text.replace("X", " ")
    numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    result = []
    for number in numbers:
        try:
            result.append(float(number))
        except ValueError:
            continue
    return result


def first_number(value: object) -> float:
    numbers = extract_numbers(value)
    return numbers[0] if numbers else np.nan


def two_numbers(value: object) -> tuple[float, float]:
    numbers = extract_numbers(value)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        return numbers[0], np.nan
    return np.nan, np.nan


def acquisition_matrix_numbers(value: object) -> tuple[float, float]:
    numbers = extract_numbers(value)
    if len(numbers) >= 4:
        # DICOM AcquisitionMatrix is:
        # frequency rows, frequency columns, phase rows, phase columns.
        rows = max(numbers[0], numbers[2])
        cols = max(numbers[1], numbers[3])
        return rows or np.nan, cols or np.nan
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return np.nan, np.nan


def get_cell(row: pd.Series, columns: dict[str, str | None], key: str) -> str:
    column = columns.get(key)
    if column is None:
        return ""
    return clean_text(row.get(column, ""))


def build_search_text(row: pd.Series, columns: dict[str, str | None]) -> str:
    parts = []
    for key in TEXT_KEYS:
        value = get_cell(row, columns, key)
        if value:
            parts.append(value)
    return " ".join(parts)


def text_forms(text: str) -> tuple[str, str]:
    lower = text.lower()
    spaced = re.sub(r"[^a-z0-9]+", " ", lower)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    return spaced, compact


def has_word(spaced_text: str, words: Iterable[str]) -> bool:
    padded = f" {spaced_text} "
    for word in words:
        if f" {word.lower()} " in padded:
            return True
    return False


def has_compact(compact_text: str, fragments: Iterable[str]) -> bool:
    return any(fragment.lower() in compact_text for fragment in fragments)


def classify_condition(text: str) -> tuple[str, str]:
    spaced, compact = text_forms(text)

    if has_word(spaced, ["localizer", "locator", "scout", "survey", "pilot"]):
        return "Localizer", "matched localizer/scout/survey"

    if (
        has_word(spaced, ["mra", "angio", "angiography", "venography", "venogram"])
        or has_compact(
            compact,
            [
                "mra",
                "tof",
                "timeofflight",
                "phasecontrastangio",
                "pcangio",
                "cetra",
                "tricks",
                "twist",
            ],
        )
    ):
        return "MRA", "matched MRA/angio/TOF"

    if has_word(spaced, ["adc"]) or has_compact(compact, ["apparentdiffusioncoefficient"]):
        return "ADC", "matched ADC"

    if (
        has_word(spaced, ["dwi", "diffusion", "trace"])
        or has_compact(compact, ["dwi", "diffusion", "b1000", "b800", "b0"])
    ):
        return "DWI", "matched DWI/diffusion"

    if has_word(spaced, ["flair"]) or has_compact(
        compact, ["flair", "fluidattenuatedinversionrecovery"]
    ):
        return "FLAIR", "matched FLAIR"

    if (
        "t2*" in text.lower()
        or has_word(spaced, ["swi", "swan", "t2star", "t2s", "hemo", "hemosiderin"])
        or has_compact(
            compact,
            ["t2star", "t2s", "t2gre", "swi", "swan", "susceptibilityweighted"],
        )
    ):
        return "T2star/SWI", "matched T2star/SWI"

    if has_word(spaced, ["t2", "t2w", "t2wi"]) or has_compact(
        compact,
        ["t2weighted", "t2wi", "t2w", "tse2", "t2tse", "t2space", "space"],
    ):
        return "T2", "matched T2"

    if has_word(spaced, ["t1", "t1w", "t1wi"]) or has_compact(
        compact,
        [
            "t1weighted",
            "t1wi",
            "t1w",
            "mprage",
            "mp2rage",
            "bravo",
            "spgr",
            "fspgr",
            "tfe",
            "vibe",
            "lava",
            "mpr",
        ],
    ):
        return "T1", "matched T1/MPRAGE/SPGR"

    return "Other", "no rule matched"


def classify_dimensionality(
    row: pd.Series,
    columns: dict[str, str | None],
    search_text: str,
    infer_from_geometry: bool,
) -> tuple[str, str]:
    mr_acquisition_type = get_cell(row, columns, "mr_acquisition_type").lower()
    if "3d" in mr_acquisition_type or mr_acquisition_type.strip() == "3":
        return "3D", "MR Acquisition Type"
    if "2d" in mr_acquisition_type or mr_acquisition_type.strip() == "2":
        return "2D", "MR Acquisition Type"

    spaced, compact = text_forms(search_text)

    if has_word(spaced, ["3d", "volume", "volumetric", "isotropic"]):
        return "3D", "matched 3D text"
    if has_compact(
        compact,
        [
            "3d",
            "mprage",
            "mp2rage",
            "bravo",
            "spgr",
            "fspgr",
            "tfe3d",
            "3dtfe",
            "vibe",
            "lava",
            "cube",
            "space",
            "vista",
            "3dtof",
        ],
    ):
        return "3D", "matched 3D sequence keyword"

    if has_word(spaced, ["2d", "multislice", "multi", "slice"]):
        return "2D", "matched 2D text"
    if has_compact(compact, ["2d", "2dtof", "tse2d", "se2d", "gre2d"]):
        return "2D", "matched 2D sequence keyword"

    if infer_from_geometry:
        thickness = first_number(get_cell(row, columns, "slice_thickness"))
        spacing = first_number(get_cell(row, columns, "spacing_between_slices"))
        if np.isfinite(thickness) and thickness <= 1.6:
            if not np.isfinite(spacing) or spacing <= 1.8:
                return "3D", "inferred from thin contiguous slices"
        if np.isfinite(thickness) and thickness >= 2.0:
            return "2D", "inferred from slice thickness"

    return "Unknown", "no rule matched"


def series_from_column(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return df[column].map(first_number)


def add_derived_columns(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    infer_dim_from_geometry: bool,
) -> pd.DataFrame:
    out = df.copy()

    search_text = out.apply(lambda row: build_search_text(row, columns), axis=1)
    classified = search_text.map(classify_condition)
    out["image_condition"] = [item[0] for item in classified]
    out["condition_reason"] = [item[1] for item in classified]

    dimensionality = out.apply(
        lambda row: classify_dimensionality(
            row,
            columns,
            build_search_text(row, columns),
            infer_dim_from_geometry,
        ),
        axis=1,
    )
    out["dimensionality"] = [item[0] for item in dimensionality]
    out["dimensionality_reason"] = [item[1] for item in dimensionality]

    out["slice_thickness_mm"] = series_from_column(out, columns["slice_thickness"])
    out["slice_spacing_mm"] = series_from_column(out, columns["spacing_between_slices"])
    out["repetition_time_ms"] = series_from_column(out, columns["repetition_time"])
    out["echo_time_ms"] = series_from_column(out, columns["echo_time"])
    out["inversion_time_ms"] = series_from_column(out, columns["inversion_time"])
    out["flip_angle_deg"] = series_from_column(out, columns["flip_angle"])
    out["number_of_slices"] = series_from_column(out, columns["number_of_slices"])
    out["field_strength_t"] = series_from_column(out, columns["magnetic_field_strength"])

    if columns["pixel_spacing"] is not None:
        spacing_pairs = out[columns["pixel_spacing"]].map(two_numbers)
        out["pixel_spacing_row_mm"] = [item[0] for item in spacing_pairs]
        out["pixel_spacing_col_mm"] = [item[1] for item in spacing_pairs]
    else:
        out["pixel_spacing_row_mm"] = np.nan
        out["pixel_spacing_col_mm"] = np.nan

    out["matrix_rows"] = series_from_column(out, columns["rows"])
    out["matrix_columns"] = series_from_column(out, columns["columns"])

    if columns["acquisition_matrix"] is not None:
        acq_pairs = out[columns["acquisition_matrix"]].map(acquisition_matrix_numbers)
        acq_rows = pd.Series([item[0] for item in acq_pairs], index=out.index)
        acq_cols = pd.Series([item[1] for item in acq_pairs], index=out.index)
        out["matrix_rows"] = out["matrix_rows"].fillna(acq_rows)
        out["matrix_columns"] = out["matrix_columns"].fillna(acq_cols)

    out["slice_gap_mm"] = out["slice_spacing_mm"] - out["slice_thickness_mm"]
    out.loc[out["slice_gap_mm"].abs() < 1e-6, "slice_gap_mm"] = 0.0

    out["fov_row_mm"] = out["matrix_rows"] * out["pixel_spacing_row_mm"]
    out["fov_col_mm"] = out["matrix_columns"] * out["pixel_spacing_col_mm"]
    out["voxel_volume_mm3"] = (
        out["pixel_spacing_row_mm"]
        * out["pixel_spacing_col_mm"]
        * out["slice_thickness_mm"]
    )

    out["matrix_size"] = [
        format_matrix_size(rows, cols)
        for rows, cols in zip(out["matrix_rows"], out["matrix_columns"])
    ]
    out["pixel_spacing_mm"] = [
        format_pair(row, col)
        for row, col in zip(out["pixel_spacing_row_mm"], out["pixel_spacing_col_mm"])
    ]

    return out


def format_matrix_size(rows: object, cols: object) -> str:
    if pd.isna(rows) or pd.isna(cols):
        return ""
    return f"{int(round(float(rows)))}x{int(round(float(cols)))}"


def format_pair(first: object, second: object, decimals: int = 3) -> str:
    if pd.isna(first) or pd.isna(second):
        return ""
    return f"{float(first):.{decimals}f}x{float(second):.{decimals}f}"


def read_csv_auto(path: Path, encoding: str | None, delimiter: str | None) -> tuple[pd.DataFrame, str]:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]
    last_error: Exception | None = None
    separator = delimiter if delimiter else None

    for enc in encodings:
        if enc is None:
            continue
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=enc,
                sep=separator,
                engine="python" if separator is None else None,
            )
            return df, enc
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            if encoding:
                raise

    raise RuntimeError(f"Failed to read {path}: {last_error}")


def condition_category(series: pd.Series) -> pd.Categorical:
    observed = [item for item in CONDITION_ORDER if item in set(series.dropna())]
    extra = sorted(set(series.dropna()) - set(CONDITION_ORDER))
    return pd.Categorical(series, categories=observed + extra, ordered=True)


def dimension_category(series: pd.Series) -> pd.Categorical:
    observed = [item for item in DIMENSION_ORDER if item in set(series.dropna())]
    extra = sorted(set(series.dropna()) - set(DIMENSION_ORDER))
    return pd.Categorical(series, categories=observed + extra, ordered=True)


def make_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["image_condition", "dimensionality"], dropna=False)
        .size()
        .reset_index(name="series_count")
        .sort_values(["image_condition", "dimensionality"])
    )
    total = max(len(df), 1)
    counts["percent"] = counts["series_count"] / total * 100.0
    return counts


def top_values(series: pd.Series, max_items: int = 5, decimals: int = 3) -> str:
    clean = series.dropna()
    if clean.empty:
        return ""
    if pd.api.types.is_numeric_dtype(clean):
        clean = clean.round(decimals)
    counts = clean.value_counts(dropna=True).head(max_items)
    return "; ".join(f"{value} (n={count})" for value, count in counts.items())


def make_metric_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["image_condition", "dimensionality"]
    for group_values, group in df.groupby(group_cols, dropna=False):
        condition, dimensionality = group_values
        total = len(group)
        for metric in NUMERIC_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row = {
                "image_condition": condition,
                "dimensionality": dimensionality,
                "metric": metric,
                "series_count": total,
                "valid_count": int(values.shape[0]),
                "missing_count": int(total - values.shape[0]),
                "missing_percent": (total - values.shape[0]) / max(total, 1) * 100.0,
                "top_values": top_values(values),
            }
            if values.empty:
                row.update(
                    {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "q1": np.nan,
                        "median": np.nan,
                        "q3": np.nan,
                        "max": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "mean": values.mean(),
                        "std": values.std(ddof=1) if values.shape[0] > 1 else 0.0,
                        "min": values.min(),
                        "q1": values.quantile(0.25),
                        "median": values.median(),
                        "q3": values.quantile(0.75),
                        "max": values.max(),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def make_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    value_specs = {
        "slice_thickness_mm": 3,
        "slice_spacing_mm": 3,
        "slice_gap_mm": 3,
        "pixel_spacing_row_mm": 3,
        "pixel_spacing_col_mm": 3,
        "matrix_rows": 0,
        "matrix_columns": 0,
        "matrix_size": None,
        "pixel_spacing_mm": None,
        "number_of_slices": 0,
        "repetition_time_ms": 1,
        "echo_time_ms": 1,
        "inversion_time_ms": 1,
        "flip_angle_deg": 1,
    }
    rows = []
    group_cols = ["image_condition", "dimensionality"]

    for (condition, dimensionality), group in df.groupby(group_cols, dropna=False):
        group_total = len(group)
        for metric, decimals in value_specs.items():
            if metric not in group:
                continue
            series = group[metric]
            if decimals is not None:
                series = pd.to_numeric(series, errors="coerce").round(decimals)
            series = series.replace("", np.nan).dropna()
            counts = series.value_counts(dropna=True)
            for value, count in counts.items():
                rows.append(
                    {
                        "image_condition": condition,
                        "dimensionality": dimensionality,
                        "metric": metric,
                        "value": value,
                        "series_count": int(count),
                        "group_percent": count / max(group_total, 1) * 100.0,
                    }
                )

    value_counts = pd.DataFrame(rows)
    if value_counts.empty:
        return value_counts
    return value_counts.sort_values(
        ["image_condition", "dimensionality", "metric", "series_count"],
        ascending=[True, True, True, False],
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def import_plot_libraries() -> None:
    global plt, sns
    if plt is not None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as matplotlib_pyplot
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "matplotlib is required for figures. Install it or run with --no-plots."
        ) from exc

    try:
        import seaborn as seaborn_module
    except ImportError:  # pragma: no cover - optional dependency
        seaborn_module = None

    plt = matplotlib_pyplot
    sns = seaborn_module


def configure_plot_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def seaborn_at_least(major: int, minor: int) -> bool:
    if sns is None:
        return False
    match = re.match(r"(\d+)\.(\d+)", getattr(sns, "__version__", "0.0"))
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def save_counts_plot(counts: pd.DataFrame, figures_dir: Path) -> Path | None:
    if counts.empty:
        return None
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "counts_by_condition_dimension.png"

    plot_df = counts.copy()
    plot_df["image_condition"] = condition_category(plot_df["image_condition"])
    plot_df["dimensionality"] = dimension_category(plot_df["dimensionality"])

    plt.figure(figsize=(10, 5.8))
    if sns is not None:
        kwargs = {
            "data": plot_df,
            "x": "image_condition",
            "y": "series_count",
            "hue": "dimensionality",
        }
        if seaborn_at_least(0, 12):
            kwargs["errorbar"] = None
        else:
            kwargs["ci"] = None
        ax = sns.barplot(**kwargs)
    else:
        pivot = plot_df.pivot_table(
            index="image_condition",
            columns="dimensionality",
            values="series_count",
            fill_value=0,
            aggfunc="sum",
        )
        ax = pivot.plot(kind="bar", figsize=(10, 5.8)).axes
    ax.set_title("Series count by image condition and dimensionality")
    ax.set_xlabel("Image condition")
    ax.set_ylabel("Series count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Dimensionality")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    label: str,
    figures_dir: Path,
    include_other: bool,
) -> Path | None:
    values = pd.to_numeric(df.get(metric), errors="coerce")
    plot_df = df.loc[values.notna(), ["image_condition", "dimensionality"]].copy()
    plot_df[metric] = values[values.notna()]
    if not include_other:
        plot_df = plot_df[plot_df["image_condition"] != "Other"]
    if plot_df.empty or plot_df["image_condition"].nunique() == 0:
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"box_{safe_filename(metric)}.png"
    plot_df["image_condition"] = condition_category(plot_df["image_condition"])
    plot_df["dimensionality"] = dimension_category(plot_df["dimensionality"])

    plt.figure(figsize=(11, 5.8))
    if sns is not None:
        strip_palette = {
            str(level): "black"
            for level in plot_df["dimensionality"].dropna().astype(str).unique()
        }
        ax = sns.boxplot(
            data=plot_df,
            x="image_condition",
            y=metric,
            hue="dimensionality",
            showfliers=False,
        )
        sns.stripplot(
            data=plot_df,
            x="image_condition",
            y=metric,
            hue="dimensionality",
            dodge=True,
            palette=strip_palette,
            alpha=0.22,
            size=2,
        )
    else:
        ax = plot_df.boxplot(column=metric, by="image_condition", figsize=(11, 5.8))
        plt.suptitle("")
    ax.set_title(label)
    ax.set_xlabel("Image condition")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=30)
    if sns is not None:
        handles, labels = ax.get_legend_handles_labels()
        unique = []
        seen = set()
        for handle, label_text in zip(handles, labels):
            if label_text in seen:
                continue
            seen.add(label_text)
            unique.append((handle, label_text))
        if unique:
            ax.legend(
                [item[0] for item in unique],
                [item[1] for item in unique],
                title="Dimensionality",
            )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_scatter_plot(df: pd.DataFrame, figures_dir: Path, include_other: bool) -> Path | None:
    required = ["slice_thickness_mm", "slice_spacing_mm"]
    if any(col not in df for col in required):
        return None
    plot_df = df.copy()
    plot_df["slice_thickness_mm"] = pd.to_numeric(
        plot_df["slice_thickness_mm"], errors="coerce"
    )
    plot_df["slice_spacing_mm"] = pd.to_numeric(
        plot_df["slice_spacing_mm"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=required)
    if not include_other:
        plot_df = plot_df[plot_df["image_condition"] != "Other"]
    if plot_df.empty:
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "scatter_slice_thickness_vs_spacing.png"
    plt.figure(figsize=(8.5, 6.2))
    if sns is not None:
        ax = sns.scatterplot(
            data=plot_df,
            x="slice_thickness_mm",
            y="slice_spacing_mm",
            hue="image_condition",
            style="dimensionality",
            alpha=0.78,
            s=52,
        )
    else:
        ax = plt.gca()
        for condition, group in plot_df.groupby("image_condition"):
            ax.scatter(
                group["slice_thickness_mm"],
                group["slice_spacing_mm"],
                label=condition,
                alpha=0.78,
            )
    ax.set_title("Slice thickness vs slice spacing")
    ax.set_xlabel("Slice thickness (mm)")
    ax.set_ylabel("Slice spacing (mm)")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def make_figures(
    df: pd.DataFrame,
    counts: pd.DataFrame,
    output_dir: Path,
    include_other: bool,
) -> list[Path]:
    import_plot_libraries()
    configure_plot_style()
    figures_dir = output_dir / "figures"
    paths: list[Path] = []

    count_path = save_counts_plot(counts, figures_dir)
    if count_path:
        paths.append(count_path)

    for metric, label in PLOT_METRICS:
        path = save_metric_boxplot(df, metric, label, figures_dir, include_other)
        if path:
            paths.append(path)

    scatter_path = save_scatter_plot(df, figures_dir, include_other)
    if scatter_path:
        paths.append(scatter_path)

    return paths


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except ImportError:
        return "```\n" + df.head(max_rows).to_string(index=False) + "\n```"


def make_report(
    input_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    columns: dict[str, str | None],
    counts: pd.DataFrame,
    stats: pd.DataFrame,
    figures: list[Path],
    encoding: str,
) -> str:
    detected_rows = [
        {"logical_field": key, "csv_column": value or ""}
        for key, value in columns.items()
    ]
    detected = pd.DataFrame(detected_rows)

    condition_counts = (
        df["image_condition"]
        .value_counts(dropna=False)
        .rename_axis("image_condition")
        .reset_index(name="series_count")
    )
    dimension_counts = (
        df["dimensionality"]
        .value_counts(dropna=False)
        .rename_axis("dimensionality")
        .reset_index(name="series_count")
    )

    missing = []
    for metric in NUMERIC_METRICS:
        if metric in df:
            valid = pd.to_numeric(df[metric], errors="coerce").notna().sum()
            missing.append(
                {
                    "metric": metric,
                    "valid_count": int(valid),
                    "missing_percent": (len(df) - valid) / max(len(df), 1) * 100.0,
                }
            )
    missing_df = pd.DataFrame(missing).sort_values("missing_percent")

    key_stats = stats[
        stats["metric"].isin(
            [
                "slice_thickness_mm",
                "slice_spacing_mm",
                "matrix_rows",
                "matrix_columns",
                "pixel_spacing_row_mm",
                "echo_time_ms",
                "repetition_time_ms",
            ]
        )
    ].copy()
    if not key_stats.empty:
        key_stats = key_stats[
            [
                "image_condition",
                "dimensionality",
                "metric",
                "valid_count",
                "median",
                "q1",
                "q3",
                "top_values",
            ]
        ]

    figure_lines = "\n".join(
        f"- `{path.relative_to(output_dir)}`" for path in figures
    ) or "- No figures generated."

    report = f"""# DICOM Tag Analysis Report

Input: `{input_path}`

Output directory: `{output_dir}`

Rows: {len(df)}

CSV encoding: `{encoding}`

## Generated files

- `series_with_derived_columns.csv`
- `condition_dimension_counts.csv`
- `metric_stats_by_condition_dimension.csv`
- `metric_value_counts_by_condition_dimension.csv`
- `detected_columns.json`
- `analysis_report.md`
{figure_lines}

## Image condition counts

{markdown_table(condition_counts)}

## Dimensionality counts

{markdown_table(dimension_counts)}

## Condition x dimensionality counts

{markdown_table(counts)}

## Key metric summary

{markdown_table(key_stats, max_rows=40)}

## Metric missingness

{markdown_table(missing_df, max_rows=40)}

## Detected input columns

{markdown_table(detected, max_rows=80)}

## Notes

- Image condition and 2D/3D labels are heuristic classifications based mainly on
  series description, protocol name, sequence name, image type, and MR acquisition
  type when present.
- Review `condition_reason` and `dimensionality_reason` in
  `series_with_derived_columns.csv` for questionable series.
- If many rows are `Other` or `Unknown`, add more descriptive DICOM tag columns to
  the input CSV, especially SeriesDescription, ProtocolName, SequenceName, ImageType,
  and MRAcquisitionType.
"""
    return report


def save_outputs(
    input_path: Path,
    df: pd.DataFrame,
    columns: dict[str, str | None],
    output_dir: Path,
    no_plots: bool,
    include_other_plots: bool,
    encoding: str,
) -> dict[str, Path | list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = make_counts(df)
    stats = make_metric_stats(df)
    value_counts = make_value_counts(df)

    derived_path = output_dir / "series_with_derived_columns.csv"
    counts_path = output_dir / "condition_dimension_counts.csv"
    stats_path = output_dir / "metric_stats_by_condition_dimension.csv"
    value_counts_path = output_dir / "metric_value_counts_by_condition_dimension.csv"
    columns_path = output_dir / "detected_columns.json"
    report_path = output_dir / "analysis_report.md"

    write_csv(df, derived_path)
    write_csv(counts, counts_path)
    write_csv(stats, stats_path)
    write_csv(value_counts, value_counts_path)

    columns_path.write_text(
        json.dumps(columns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figures: list[Path] = []
    if not no_plots:
        figures = make_figures(df, counts, output_dir, include_other_plots)

    report = make_report(
        input_path=input_path,
        output_dir=output_dir,
        df=df,
        columns=columns,
        counts=counts,
        stats=stats,
        figures=figures,
        encoding=encoding,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "derived": derived_path,
        "counts": counts_path,
        "stats": stats_path,
        "value_counts": value_counts_path,
        "columns": columns_path,
        "report": report_path,
        "figures": figures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a one-row-per-series MRI DICOM tag CSV and summarize "
            "image-condition specific geometry/protocol statistics."
        )
    )
    parser.add_argument("csv", type=Path, help="Input CSV file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("dicom_tag_analysis_output"),
        help="Directory for summary CSVs, figures, and report.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="CSV encoding. If omitted, utf-8-sig, utf-8, cp932, shift_jis, latin1 are tried.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="CSV delimiter. If omitted, pandas tries to detect it.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG figure generation.",
    )
    parser.add_argument(
        "--include-other-plots",
        action="store_true",
        help="Include Other condition in metric boxplots and scatter plots.",
    )
    parser.add_argument(
        "--infer-dim-from-geometry",
        action="store_true",
        help=(
            "If textual 2D/3D hints are missing, infer 3D for thin contiguous "
            "slices and 2D for thicker slices."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.csv
    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        raw_df, encoding = read_csv_auto(input_path, args.encoding, args.delimiter)

    if raw_df.empty:
        print(f"Input CSV has no rows: {input_path}", file=sys.stderr)
        return 2

    columns = detect_columns(raw_df)
    derived = add_derived_columns(
        raw_df,
        columns,
        infer_dim_from_geometry=args.infer_dim_from_geometry,
    )
    outputs = save_outputs(
        input_path=input_path,
        df=derived,
        columns=columns,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
        include_other_plots=args.include_other_plots,
        encoding=encoding,
    )

    print(f"Analyzed {len(derived)} series")
    print(f"Report: {outputs['report']}")
    print(f"Derived CSV: {outputs['derived']}")
    print(f"Stats CSV: {outputs['stats']}")
    if not args.no_plots:
        print(f"Figures: {len(outputs['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
