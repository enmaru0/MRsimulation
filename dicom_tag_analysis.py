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
PLANE_ORDER = ["AX", "SAG", "COR", "Oblique", "Unknown"]

DEFAULT_EXCLUDE_SLICE_MM = 10.0
DEFAULT_EXCLUDE_SLICES_LE = 7

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
    "frame_type": [
        "FrameType",
        "Frame Type",
        "(0008,9007)",
        "0008,9007",
        "00089007",
    ],
    "complex_image_component": [
        "ComplexImageComponent",
        "Complex Image Component",
        "(0008,9208)",
        "0008,9208",
        "00089208",
    ],
    "acquisition_contrast": [
        "AcquisitionContrast",
        "Acquisition Contrast",
        "(0008,9209)",
        "0008,9209",
        "00089209",
    ],
    "derivation_description": [
        "DerivationDescription",
        "Derivation Description",
        "(0008,2111)",
        "0008,2111",
        "00082111",
    ],
    "image_comments": [
        "ImageComments",
        "Image Comments",
        "(0020,4000)",
        "0020,4000",
        "00204000",
    ],
    "image_orientation_patient": [
        "ImageOrientationPatient",
        "Image Orientation Patient",
        "Image Orientation (Patient)",
        "(0020,0037)",
        "0020,0037",
        "00200037",
    ],
    "patient_orientation": [
        "PatientOrientation",
        "Patient Orientation",
        "(0020,0020)",
        "0020,0020",
        "00200020",
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
    "echo_train_length": [
        "EchoTrainLength",
        "Echo Train Length",
        "ETL",
        "TurboFactor",
        "Turbo Factor",
        "(0018,0091)",
        "0018,0091",
        "00180091",
    ],
    "echo_numbers": [
        "EchoNumbers",
        "Echo Numbers",
        "EchoNumber",
        "Echo Number",
        "(0018,0086)",
        "0018,0086",
        "00180086",
    ],
    "diffusion_b_value": [
        "DiffusionBValue",
        "Diffusion B Value",
        "Diffusion b-value",
        "BValue",
        "B Value",
        "B-value",
        "(0018,9087)",
        "0018,9087",
        "00189087",
    ],
    "diffusion_directionality": [
        "DiffusionDirectionality",
        "Diffusion Directionality",
        "(0018,9075)",
        "0018,9075",
        "00189075",
    ],
    "diffusion_anisotropy_type": [
        "DiffusionAnisotropyType",
        "Diffusion Anisotropy Type",
        "(0018,9147)",
        "0018,9147",
        "00189147",
    ],
    "diffusion_gradient_orientation": [
        "DiffusionGradientOrientation",
        "Diffusion Gradient Orientation",
        "(0018,9089)",
        "0018,9089",
        "00189089",
    ],
    "contrast_bolus_agent": [
        "ContrastBolusAgent",
        "Contrast Bolus Agent",
        "(0018,0010)",
        "0018,0010",
        "00180010",
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

SERIES_PROTOCOL_TEXT_KEYS = [
    "series_description",
    "protocol_name",
]

CLASSIFICATION_TEXT_KEYS = [
    "sequence_name",
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",
    "image_type",
    "frame_type",
    "complex_image_component",
    "acquisition_contrast",
    "derivation_description",
    "image_comments",
    "diffusion_directionality",
    "diffusion_anisotropy_type",
    "contrast_bolus_agent",
    "body_part",
]

TEXT_KEYS = SERIES_PROTOCOL_TEXT_KEYS + CLASSIFICATION_TEXT_KEYS

NUMERIC_METRICS = [
    "slice_thickness_mm",
    "slice_spacing_mm",
    "slice_gap_mm",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
    "matrix_rows",
    "matrix_columns",
    "image_rows",
    "image_columns",
    "number_of_slices",
    "repetition_time_ms",
    "echo_time_ms",
    "inversion_time_ms",
    "flip_angle_deg",
    "echo_train_length",
    "echo_numbers",
    "diffusion_b_value",
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
    ("fov_row_mm", "FOV row (mm)"),
    ("fov_col_mm", "FOV col (mm)"),
    ("matrix_rows", "Matrix rows"),
    ("matrix_columns", "Matrix columns"),
    ("number_of_slices", "Number of slices"),
    ("repetition_time_ms", "TR (ms)"),
    ("echo_time_ms", "TE (ms)"),
    ("inversion_time_ms", "TI (ms)"),
    ("flip_angle_deg", "Flip angle (deg)"),
]

SUMMARY_METRICS = [
    ("slice_thickness_mm", "slice_thickness_mm"),
    ("slice_spacing_mm", "space_between_slices_mm"),
    ("slice_gap_mm", "slice_gap_mm"),
    ("pixel_spacing_row_mm", "pixel_spacing_row_mm"),
    ("pixel_spacing_col_mm", "pixel_spacing_col_mm"),
    ("fov_row_mm", "fov_row_mm"),
    ("fov_col_mm", "fov_col_mm"),
    ("matrix_rows", "matrix_rows"),
    ("matrix_columns", "matrix_columns"),
    ("image_rows", "image_rows"),
    ("image_columns", "image_columns"),
    ("number_of_slices", "number_of_slices"),
    ("repetition_time_ms", "tr_ms"),
    ("echo_time_ms", "te_ms"),
    ("inversion_time_ms", "ti_ms"),
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


def build_search_text(
    row: pd.Series,
    columns: dict[str, str | None],
    keys: Iterable[str] | None = None,
) -> str:
    parts = []
    for key in keys or TEXT_KEYS:
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
                "phasecontrast",
                "pcangio",
                "pcmr",
                "flowencoded",
                "flowencoding",
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
        or has_compact(
            compact,
            ["dwi", "diffusion", "diffusionweighted", "traceweighted", "b1000", "b800", "b0"],
        )
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
            [
                "t2star",
                "t2s",
                "t2gre",
                "t2wgre",
                "swi",
                "swan",
                "susceptibilityweighted",
                "t2starweighted",
            ],
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


def get_number(row: pd.Series, columns: dict[str, str | None], key: str) -> float:
    return first_number(get_cell(row, columns, key))


def is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def metric_evidence(**values: float) -> str:
    parts = []
    for name, value in values.items():
        if is_finite(value):
            parts.append(f"{name}={value:g}")
    return ", ".join(parts)


def classify_condition_by_parameters(
    row: pd.Series,
    columns: dict[str, str | None],
) -> tuple[str, str]:
    """Fallback classifier that avoids SeriesDescription and ProtocolName.

    This uses numeric MR parameters and non-protocol DICOM tags. It intentionally
    stays conservative for MRA because short-TR/short-TE GRE can also be T1.
    """

    tr = get_number(row, columns, "repetition_time")
    te = get_number(row, columns, "echo_time")
    ti = get_number(row, columns, "inversion_time")
    fa = get_number(row, columns, "flip_angle")
    etl = get_number(row, columns, "echo_train_length")
    echo_numbers = get_number(row, columns, "echo_numbers")
    b_value = get_number(row, columns, "diffusion_b_value")
    thickness = get_number(row, columns, "slice_thickness")

    tag_text = build_search_text(row, columns, CLASSIFICATION_TEXT_KEYS)
    spaced, compact = text_forms(tag_text)

    has_diffusion_tag = any(
        get_cell(row, columns, key)
        for key in [
            "diffusion_directionality",
            "diffusion_anisotropy_type",
            "diffusion_gradient_orientation",
        ]
    )
    if is_finite(b_value) and (b_value > 0 or has_diffusion_tag):
        return (
            "DWI",
            "parameter heuristic: diffusion b-value/tag present; "
            + metric_evidence(b=b_value, TR=tr, TE=te),
        )

    has_gr = (
        has_word(spaced, ["gr", "gre", "spgr", "fspgr", "tfe", "ffe"])
        or has_compact(compact, ["gradient", "gradientecho", "spoiledgre", "fastspgr"])
    )
    has_se = has_word(spaced, ["se", "tse", "fse", "ir"]) or has_compact(
        compact, ["spinecho", "turbospinecho", "fastspinecho", "inversionrecovery"]
    )
    has_ep = has_word(spaced, ["ep", "epi"]) or has_compact(compact, ["echoplanar"])
    has_ir = has_word(spaced, ["ir"]) or has_compact(compact, ["inversionrecovery"])
    has_phase = has_word(spaced, ["phase"]) or has_compact(compact, ["phaseimage"])
    has_magnitude = has_word(spaced, ["magnitude", "mag"]) or has_compact(
        compact, ["magnitudeimage"]
    )

    if has_ep and is_finite(te) and te >= 45 and (is_finite(tr) and tr >= 1000):
        return (
            "DWI",
            "parameter heuristic: EPI with long TE/TR, review if this could be fMRI/perfusion; "
            + metric_evidence(TR=tr, TE=te, b=b_value),
        )

    if is_finite(ti) and ti >= 1400 and (
        (is_finite(tr) and tr >= 4000) or (is_finite(te) and te >= 60)
    ):
        return (
            "FLAIR",
            "parameter heuristic: long inversion time with long TR/TE; "
            + metric_evidence(TR=tr, TE=te, TI=ti),
        )

    if has_gr and (
        (is_finite(te) and te >= 12 and (not is_finite(fa) or fa <= 45))
        or (is_finite(echo_numbers) and echo_numbers >= 2 and is_finite(te) and te >= 8)
        or (has_phase and is_finite(te) and te >= 8)
    ):
        return (
            "T2star/SWI",
            "parameter heuristic: GRE/phase or multi-echo pattern; "
            + metric_evidence(TR=tr, TE=te, FA=fa, EchoNumbers=echo_numbers),
        )

    if is_finite(ti) and 450 <= ti < 1400 and is_finite(te) and te <= 20:
        if (is_finite(fa) and fa <= 30) or has_gr or has_ir:
            return (
                "T1",
                "parameter heuristic: T1 inversion-prepared short-TE pattern; "
                + metric_evidence(TR=tr, TE=te, TI=ti, FA=fa),
            )

    if is_finite(te) and te >= 60:
        if (is_finite(tr) and tr >= 1500) or (is_finite(etl) and etl >= 3) or has_se:
            return (
                "T2",
                "parameter heuristic: long TE with long TR/TSE-FSE pattern; "
                + metric_evidence(TR=tr, TE=te, ETL=etl),
            )

    if is_finite(te) and te <= 35:
        if is_finite(tr) and 100 <= tr <= 1200:
            return (
                "T1",
                "parameter heuristic: short TR and short TE; "
                + metric_evidence(TR=tr, TE=te, FA=fa),
            )
        if has_gr and is_finite(tr) and tr < 100 and is_finite(te) and te <= 10:
            if not is_finite(fa) or fa <= 15:
                return (
                    "T1",
                    "parameter heuristic: short-TR/short-TE low-flip-angle GRE; "
                    + metric_evidence(TR=tr, TE=te, FA=fa, SliceThickness=thickness),
                )

    if has_magnitude and has_gr and is_finite(te) and te >= 12:
        return (
            "T2star/SWI",
            "parameter heuristic: GRE magnitude image with T2star-range TE; "
            + metric_evidence(TR=tr, TE=te, FA=fa),
        )

    return "Other", "no non-description/protocol parameter rule matched"


def classify_condition_from_row(
    row: pd.Series,
    columns: dict[str, str | None],
    use_series_protocol_text: bool,
) -> tuple[str, str]:
    non_protocol_text = build_search_text(row, columns, CLASSIFICATION_TEXT_KEYS)
    condition, reason = classify_condition(non_protocol_text)
    if condition != "Other":
        return condition, "matched non-description/protocol tags: " + reason

    condition, reason = classify_condition_by_parameters(row, columns)
    if condition != "Other":
        return condition, reason

    if use_series_protocol_text:
        series_protocol_text = build_search_text(row, columns, SERIES_PROTOCOL_TEXT_KEYS)
        condition, text_reason = classify_condition(series_protocol_text)
        if condition != "Other":
            return condition, "matched SeriesDescription/ProtocolName fallback: " + text_reason

    return "Other", reason


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


def plane_from_normal(normal: np.ndarray, oblique_threshold: float = 0.80) -> tuple[str, float]:
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-6:
        return "Unknown", np.nan

    unit = np.asarray(normal, dtype=np.float64) / norm
    abs_unit = np.abs(unit)
    axis = int(np.argmax(abs_unit))
    max_component = float(abs_unit[axis])
    obliquity_deg = float(np.degrees(np.arccos(np.clip(max_component, -1.0, 1.0))))
    if max_component < oblique_threshold:
        return "Oblique", obliquity_deg
    if axis == 0:
        return "SAG", obliquity_deg
    if axis == 1:
        return "COR", obliquity_deg
    return "AX", obliquity_deg


def classify_plane_from_iop(value: object) -> tuple[str, str, float]:
    numbers = extract_numbers(value)
    if len(numbers) < 6:
        return "Unknown", "ImageOrientationPatient missing or incomplete", np.nan
    row_cosines = np.asarray(numbers[:3], dtype=np.float64)
    col_cosines = np.asarray(numbers[3:6], dtype=np.float64)
    normal = np.cross(row_cosines, col_cosines)
    plane, obliquity_deg = plane_from_normal(normal)
    if plane == "Unknown":
        return plane, "ImageOrientationPatient normal vector is invalid", obliquity_deg
    return (
        plane,
        "matched ImageOrientationPatient; obliquity_deg="
        + ("" if pd.isna(obliquity_deg) else f"{obliquity_deg:.1f}"),
        obliquity_deg,
    )


def orientation_axis(token: str) -> str | None:
    token = clean_text(token).upper()
    if not token:
        return None
    first = token[0]
    if first in {"L", "R"}:
        return "x"
    if first in {"A", "P"}:
        return "y"
    if first in {"H", "F", "S", "I"}:
        return "z"
    return None


def classify_plane_from_patient_orientation(value: object) -> tuple[str, str, float]:
    text = clean_text(value)
    if not text:
        return "Unknown", "PatientOrientation missing", np.nan
    tokens = [item for item in re.split(r"[^A-Za-z]+", text.upper()) if item]
    axes = []
    for token in tokens[:2]:
        axis = orientation_axis(token)
        if axis and axis not in axes:
            axes.append(axis)
    if len(axes) < 2:
        return "Unknown", "PatientOrientation could not be mapped to two in-plane axes", np.nan
    in_plane = set(axes)
    if in_plane == {"x", "y"}:
        return "AX", "matched PatientOrientation axes", 0.0
    if in_plane == {"x", "z"}:
        return "COR", "matched PatientOrientation axes", 0.0
    if in_plane == {"y", "z"}:
        return "SAG", "matched PatientOrientation axes", 0.0
    return "Unknown", "PatientOrientation axes were ambiguous", np.nan


def classify_plane_from_text(text: str) -> tuple[str, str, float]:
    spaced, compact = text_forms(text)
    if has_word(spaced, ["ax", "axi", "axial", "tra", "transverse"]) or has_compact(
        compact,
        ["axial", "transverse"],
    ):
        return "AX", "matched plane text", np.nan
    if has_word(spaced, ["sag", "sagittal"]):
        return "SAG", "matched plane text", np.nan
    if has_word(spaced, ["cor", "coronal"]):
        return "COR", "matched plane text", np.nan
    return "Unknown", "no plane text matched", np.nan


def classify_scan_plane(
    row: pd.Series,
    columns: dict[str, str | None],
) -> tuple[str, str, float]:
    plane, reason, obliquity = classify_plane_from_iop(
        get_cell(row, columns, "image_orientation_patient")
    )
    if plane != "Unknown":
        return plane, reason, obliquity

    plane, reason, obliquity = classify_plane_from_patient_orientation(
        get_cell(row, columns, "patient_orientation")
    )
    if plane != "Unknown":
        return plane, reason, obliquity

    plane, reason, obliquity = classify_plane_from_text(
        build_search_text(row, columns, TEXT_KEYS)
    )
    if plane != "Unknown":
        return plane, reason, obliquity

    return "Unknown", "no ImageOrientationPatient/PatientOrientation/plane text matched", np.nan


def series_from_column(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return df[column].map(first_number)


def add_derived_columns(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    infer_dim_from_geometry: bool,
    use_series_protocol_text: bool,
) -> pd.DataFrame:
    out = df.copy()

    classified = out.apply(
        lambda row: classify_condition_from_row(
            row,
            columns,
            use_series_protocol_text=use_series_protocol_text,
        ),
        axis=1,
    )
    out["image_condition"] = [item[0] for item in classified]
    out["condition_reason"] = [item[1] for item in classified]

    dimensionality = out.apply(
        lambda row: classify_dimensionality(
            row,
            columns,
            build_search_text(row, columns, CLASSIFICATION_TEXT_KEYS),
            infer_dim_from_geometry,
        ),
        axis=1,
    )
    out["dimensionality"] = [item[0] for item in dimensionality]
    out["dimensionality_reason"] = [item[1] for item in dimensionality]

    plane = out.apply(
        lambda row: classify_scan_plane(row, columns),
        axis=1,
    )
    out["scan_plane"] = [item[0] for item in plane]
    out["scan_plane_reason"] = [item[1] for item in plane]
    out["plane_obliquity_deg"] = [item[2] for item in plane]

    out["slice_thickness_mm"] = series_from_column(out, columns["slice_thickness"])
    out["slice_spacing_mm"] = series_from_column(out, columns["spacing_between_slices"])
    out["repetition_time_ms"] = series_from_column(out, columns["repetition_time"])
    out["echo_time_ms"] = series_from_column(out, columns["echo_time"])
    out["inversion_time_ms"] = series_from_column(out, columns["inversion_time"])
    out["flip_angle_deg"] = series_from_column(out, columns["flip_angle"])
    out["echo_train_length"] = series_from_column(out, columns["echo_train_length"])
    out["echo_numbers"] = series_from_column(out, columns["echo_numbers"])
    out["diffusion_b_value"] = series_from_column(out, columns["diffusion_b_value"])
    out["number_of_slices"] = series_from_column(out, columns["number_of_slices"])
    out["field_strength_t"] = series_from_column(out, columns["magnetic_field_strength"])
    out["field_strength_label"] = out["field_strength_t"].map(format_field_strength_label)

    if columns["pixel_spacing"] is not None:
        spacing_pairs = out[columns["pixel_spacing"]].map(two_numbers)
        out["pixel_spacing_row_mm"] = [item[0] for item in spacing_pairs]
        out["pixel_spacing_col_mm"] = [item[1] for item in spacing_pairs]
    else:
        out["pixel_spacing_row_mm"] = np.nan
        out["pixel_spacing_col_mm"] = np.nan

    out["image_rows"] = series_from_column(out, columns["rows"])
    out["image_columns"] = series_from_column(out, columns["columns"])
    out["matrix_rows"] = out["image_rows"].copy()
    out["matrix_columns"] = out["image_columns"].copy()
    out["matrix_source"] = np.where(
        out["matrix_rows"].notna() & out["matrix_columns"].notna(),
        "Rows/Columns",
        "",
    )

    if columns["acquisition_matrix"] is not None:
        acq_pairs = out[columns["acquisition_matrix"]].map(acquisition_matrix_numbers)
        acq_rows = pd.Series([item[0] for item in acq_pairs], index=out.index)
        acq_cols = pd.Series([item[1] for item in acq_pairs], index=out.index)
        has_acq_matrix = acq_rows.notna() & acq_cols.notna()
        out["acquisition_matrix_rows"] = acq_rows
        out["acquisition_matrix_columns"] = acq_cols
        out.loc[has_acq_matrix, "matrix_rows"] = acq_rows[has_acq_matrix]
        out.loc[has_acq_matrix, "matrix_columns"] = acq_cols[has_acq_matrix]
        out.loc[has_acq_matrix, "matrix_source"] = "AcquisitionMatrix"
    else:
        out["acquisition_matrix_rows"] = np.nan
        out["acquisition_matrix_columns"] = np.nan

    out["slice_gap_mm"] = out["slice_spacing_mm"] - out["slice_thickness_mm"]
    out.loc[out["slice_gap_mm"].abs() < 1e-6, "slice_gap_mm"] = 0.0

    fov_rows = out["image_rows"].fillna(out["matrix_rows"])
    fov_columns = out["image_columns"].fillna(out["matrix_columns"])
    out["fov_row_mm"] = fov_rows * out["pixel_spacing_row_mm"]
    out["fov_col_mm"] = fov_columns * out["pixel_spacing_col_mm"]
    out["voxel_volume_mm3"] = (
        out["pixel_spacing_row_mm"]
        * out["pixel_spacing_col_mm"]
        * out["slice_thickness_mm"]
    )

    out["matrix_size"] = [
        format_matrix_size(rows, cols)
        for rows, cols in zip(out["matrix_rows"], out["matrix_columns"])
    ]
    out["image_matrix_size"] = [
        format_matrix_size(rows, cols)
        for rows, cols in zip(out["image_rows"], out["image_columns"])
    ]
    out["acquisition_matrix_size"] = [
        format_matrix_size(rows, cols)
        for rows, cols in zip(
            out["acquisition_matrix_rows"],
            out["acquisition_matrix_columns"],
        )
    ]
    out["pixel_spacing_mm"] = [
        format_pair(row, col)
        for row, col in zip(out["pixel_spacing_row_mm"], out["pixel_spacing_col_mm"])
    ]
    out["fov_mm"] = [
        format_pair(row, col, decimals=1)
        for row, col in zip(out["fov_row_mm"], out["fov_col_mm"])
    ]
    out["slice_thickness_spacing_mm"] = [
        format_pair(thickness, spacing, decimals=2)
        for thickness, spacing in zip(out["slice_thickness_mm"], out["slice_spacing_mm"])
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


def format_field_strength_label(value: object) -> str:
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return "UnknownField"
    if not np.isfinite(strength) or strength <= 0:
        return "UnknownField"

    rounded = round(strength, 2)
    if abs(rounded - round(rounded)) < 1e-6:
        text = str(int(round(rounded)))
    else:
        text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return f"{text}T"


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


def format_analysis_group(
    condition: object,
    dimensionality: object,
    scan_plane: object | None = None,
    field_strength: object | None = None,
) -> str:
    condition_text = clean_text(condition) or "Other"
    dimension_text = clean_text(dimensionality) or "Unknown"
    plane_text = clean_text(scan_plane)
    field_text = clean_text(field_strength)
    group_parts = [dimension_text]
    if plane_text:
        group_parts.append(plane_text)
    if field_text:
        group_parts.append(field_text)
    if len(group_parts) > 1:
        return f"{condition_text}({','.join(group_parts)})"
    return f"{condition_text}({dimension_text})"


def ordered_condition_values(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna()]
    observed = [item for item in CONDITION_ORDER if item in set(values)]
    extra = sorted(set(values) - set(CONDITION_ORDER))
    return observed + extra


def ordered_dimension_values(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna()]
    observed = [item for item in DIMENSION_ORDER if item in set(values)]
    extra = sorted(set(values) - set(DIMENSION_ORDER))
    return observed + extra


def ordered_plane_values(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna()]
    observed = [item for item in PLANE_ORDER if item in set(values)]
    extra = sorted(set(values) - set(PLANE_ORDER))
    return observed + extra


def field_strength_sort_key(value: str) -> tuple[int, float, str]:
    if value == "UnknownField" or not value:
        return (2, float("inf"), value)
    numbers = extract_numbers(value)
    if numbers:
        return (0, numbers[0], value)
    return (1, float("inf"), value)


def ordered_field_strength_values(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna()]
    value_set = set(values)
    known = sorted(
        [value for value in value_set if value and value != "UnknownField"],
        key=field_strength_sort_key,
    )
    if "" in value_set:
        known.append("")
    if "UnknownField" in value_set:
        known.append("UnknownField")
    return known


def ordered_analysis_groups(df: pd.DataFrame) -> list[str]:
    if df.empty or "image_condition" not in df or "dimensionality" not in df:
        return []
    if "scan_plane" not in df:
        planes = pd.Series([""] * len(df), index=df.index)
    else:
        planes = df["scan_plane"]
    if "field_strength_label" not in df:
        field_strengths = pd.Series([""] * len(df), index=df.index)
    else:
        field_strengths = df["field_strength_label"]
    present = set(
        format_analysis_group(condition, dimensionality, plane, field_strength)
        for condition, dimensionality, plane, field_strength in zip(
            df["image_condition"],
            df["dimensionality"],
            planes,
            field_strengths,
        )
    )
    ordered = []
    for condition in ordered_condition_values(df["image_condition"]):
        for dimensionality in ordered_dimension_values(df["dimensionality"]):
            for plane in ordered_plane_values(planes):
                for field_strength in ordered_field_strength_values(field_strengths):
                    group = format_analysis_group(
                        condition,
                        dimensionality,
                        plane,
                        field_strength,
                    )
                    if group in present:
                        ordered.append(group)
    extra = sorted(present - set(ordered))
    return ordered + extra


def add_analysis_filter(
    df: pd.DataFrame,
    max_slice_mm: float | None,
    exclude_slices_le: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "scan_plane" not in out:
        out["scan_plane"] = "Unknown"
    if "field_strength_label" not in out:
        out["field_strength_label"] = "UnknownField"
    out["analysis_group"] = [
        format_analysis_group(condition, dimensionality, plane, field_strength)
        for condition, dimensionality, plane, field_strength in zip(
            out["image_condition"],
            out["dimensionality"],
            out["scan_plane"],
            out["field_strength_label"],
        )
    ]

    if (max_slice_mm is None or max_slice_mm <= 0) and (
        exclude_slices_le is None or exclude_slices_le <= 0
    ):
        out["analysis_excluded"] = False
        out["analysis_exclusion_reason"] = ""
        return out

    thickness = pd.to_numeric(
        out["slice_thickness_mm"]
        if "slice_thickness_mm" in out
        else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    spacing = pd.to_numeric(
        out["slice_spacing_mm"]
        if "slice_spacing_mm" in out
        else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    number_of_slices = pd.to_numeric(
        out["number_of_slices"]
        if "number_of_slices" in out
        else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    if max_slice_mm is None or max_slice_mm <= 0:
        thick_excluded = pd.Series(False, index=out.index)
        spacing_excluded = pd.Series(False, index=out.index)
    else:
        thick_excluded = thickness.ge(max_slice_mm).fillna(False)
        spacing_excluded = spacing.ge(max_slice_mm).fillna(False)
    if exclude_slices_le is None or exclude_slices_le <= 0:
        few_slices_excluded = pd.Series(False, index=out.index)
    else:
        few_slices_excluded = number_of_slices.le(exclude_slices_le).fillna(False)
    out["analysis_excluded"] = (
        thick_excluded | spacing_excluded | few_slices_excluded
    )

    reasons = []
    for thick, space, few_slices, thick_value, space_value, slice_count in zip(
        thick_excluded,
        spacing_excluded,
        few_slices_excluded,
        thickness,
        spacing,
        number_of_slices,
    ):
        row_reasons = []
        if thick:
            row_reasons.append(f"slice_thickness_mm={thick_value:g} >= {max_slice_mm:g}")
        if space:
            row_reasons.append(f"slice_spacing_mm={space_value:g} >= {max_slice_mm:g}")
        if few_slices:
            row_reasons.append(
                f"number_of_slices={slice_count:g} <= {exclude_slices_le:g}"
            )
        reasons.append("; ".join(row_reasons))
    out["analysis_exclusion_reason"] = reasons
    return out


def make_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    counts = (
        df.groupby(
            [
                "image_condition",
                "dimensionality",
                "scan_plane",
                "field_strength_label",
                "analysis_group",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="series_count")
    )
    group_order = {group: idx for idx, group in enumerate(ordered_analysis_groups(df))}
    counts["_order"] = counts["analysis_group"].map(group_order).fillna(len(group_order))
    counts = counts.sort_values(["_order", "image_condition", "dimensionality"]).drop(
        columns="_order"
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
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    rows = []
    group_cols = [
        "image_condition",
        "dimensionality",
        "scan_plane",
        "field_strength_label",
        "analysis_group",
    ]
    for group_values, group in df.groupby(group_cols, dropna=False):
        condition, dimensionality, scan_plane, field_strength, analysis_group = group_values
        total = len(group)
        for metric in NUMERIC_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row = {
                "image_condition": condition,
                "dimensionality": dimensionality,
                "scan_plane": scan_plane,
                "field_strength_label": field_strength,
                "analysis_group": analysis_group,
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
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    value_specs = {
        "slice_thickness_mm": 3,
        "slice_spacing_mm": 3,
        "slice_thickness_spacing_mm": None,
        "slice_gap_mm": 3,
        "pixel_spacing_row_mm": 3,
        "pixel_spacing_col_mm": 3,
        "matrix_rows": 0,
        "matrix_columns": 0,
        "matrix_size": None,
        "matrix_source": None,
        "image_matrix_size": None,
        "acquisition_matrix_size": None,
        "fov_row_mm": 1,
        "fov_col_mm": 1,
        "fov_mm": None,
        "pixel_spacing_mm": None,
        "number_of_slices": 0,
        "repetition_time_ms": 1,
        "echo_time_ms": 1,
        "inversion_time_ms": 1,
        "flip_angle_deg": 1,
        "echo_train_length": 0,
        "echo_numbers": 0,
        "diffusion_b_value": 1,
    }
    rows = []
    group_cols = [
        "image_condition",
        "dimensionality",
        "scan_plane",
        "field_strength_label",
        "analysis_group",
    ]

    for (
        condition,
        dimensionality,
        scan_plane,
        field_strength,
        analysis_group,
    ), group in df.groupby(group_cols, dropna=False):
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
                        "scan_plane": scan_plane,
                        "field_strength_label": field_strength,
                        "analysis_group": analysis_group,
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
        ["analysis_group", "metric", "series_count"],
        ascending=[True, True, False],
    )


def summarize_numeric(series: pd.Series, decimals: int = 2) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return f"{median:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"


def numeric_valid_count(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def top_text_values(series: pd.Series, max_items: int = 5) -> str:
    clean = series.map(clean_text).replace("", np.nan).dropna()
    if clean.empty:
        return ""
    counts = clean.value_counts(dropna=True).head(max_items)
    return "; ".join(f"{value} (n={count})" for value, count in counts.items())


def make_condition_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)

    decimals = {
        "slice_thickness_mm": 2,
        "slice_spacing_mm": 2,
        "slice_gap_mm": 2,
        "pixel_spacing_row_mm": 3,
        "pixel_spacing_col_mm": 3,
        "fov_row_mm": 1,
        "fov_col_mm": 1,
        "matrix_rows": 0,
        "matrix_columns": 0,
        "number_of_slices": 0,
        "repetition_time_ms": 1,
        "echo_time_ms": 1,
        "inversion_time_ms": 1,
    }

    rows = []
    group_cols = [
        "image_condition",
        "dimensionality",
        "scan_plane",
        "field_strength_label",
        "analysis_group",
    ]
    for (
        condition,
        dimensionality,
        scan_plane,
        field_strength,
        analysis_group,
    ), group in df.groupby(
        group_cols,
        dropna=False,
    ):
        row = {
            "image_condition": condition,
            "dimensionality": dimensionality,
            "scan_plane": scan_plane,
            "field_strength_label": field_strength,
            "analysis_group": analysis_group,
            "series_count": len(group),
        }
        for metric, output_name in SUMMARY_METRICS:
            if metric not in group:
                row[output_name] = ""
                row[f"{output_name}_n"] = 0
                continue
            row[output_name] = summarize_numeric(group[metric], decimals.get(metric, 2))
            row[f"{output_name}_n"] = numeric_valid_count(group[metric])

        row["common_matrix_size"] = top_text_values(group.get("matrix_size", pd.Series(dtype=str)))
        row["common_slice_thickness_spacing_mm"] = top_text_values(
            group.get("slice_thickness_spacing_mm", pd.Series(dtype=str))
        )
        row["common_fov_mm"] = top_text_values(group.get("fov_mm", pd.Series(dtype=str)))
        row["common_pixel_spacing_mm"] = top_text_values(
            group.get("pixel_spacing_mm", pd.Series(dtype=str))
        )
        row["matrix_source"] = top_text_values(group.get("matrix_source", pd.Series(dtype=str)))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    group_order = {group: idx for idx, group in enumerate(ordered_analysis_groups(df))}
    summary["_order"] = summary["analysis_group"].map(group_order).fillna(len(group_order))
    return summary.sort_values("_order").drop(columns="_order")


def make_slice_thickness_spacing_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    required = ["slice_thickness_mm", "slice_spacing_mm"]
    if any(column not in df for column in required):
        return pd.DataFrame(
            columns=[
                "image_condition",
                "dimensionality",
                "scan_plane",
                "field_strength_label",
                "analysis_group",
                "slice_thickness_mm",
                "slice_spacing_mm",
                "slice_thickness_spacing_mm",
                "series_count",
                "group_percent",
            ]
        )

    work = df.copy()
    work["slice_thickness_mm"] = pd.to_numeric(
        work["slice_thickness_mm"],
        errors="coerce",
    ).round(3)
    work["slice_spacing_mm"] = pd.to_numeric(
        work["slice_spacing_mm"],
        errors="coerce",
    ).round(3)
    work = work.dropna(subset=required)
    if work.empty:
        return pd.DataFrame()

    work["slice_thickness_spacing_mm"] = [
        format_pair(thickness, spacing, decimals=2)
        for thickness, spacing in zip(work["slice_thickness_mm"], work["slice_spacing_mm"])
    ]
    group_total = work.groupby("analysis_group", dropna=False).size()
    counts = (
        work.groupby(
            [
                "image_condition",
                "dimensionality",
                "scan_plane",
                "field_strength_label",
                "analysis_group",
                "slice_thickness_mm",
                "slice_spacing_mm",
                "slice_thickness_spacing_mm",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="series_count")
    )
    counts["group_percent"] = [
        count / max(group_total.get(group, 0), 1) * 100.0
        for group, count in zip(counts["analysis_group"], counts["series_count"])
    ]
    group_order = {group: idx for idx, group in enumerate(ordered_analysis_groups(work))}
    counts["_order"] = counts["analysis_group"].map(group_order).fillna(len(group_order))
    return counts.sort_values(
        ["_order", "series_count", "slice_thickness_mm", "slice_spacing_mm"],
        ascending=[True, False, True, True],
    ).drop(columns="_order")


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
    if "analysis_group" not in plot_df:
        plot_df["analysis_group"] = [
            format_analysis_group(condition, dimensionality)
            for condition, dimensionality in zip(
                plot_df["image_condition"],
                plot_df["dimensionality"],
            )
        ]
    group_order = list(plot_df["analysis_group"])
    plot_df["analysis_group"] = pd.Categorical(
        plot_df["analysis_group"],
        categories=group_order,
        ordered=True,
    )

    plt.figure(figsize=(max(10, 0.65 * len(plot_df) + 3), 5.8))
    if sns is not None:
        kwargs = {
            "data": plot_df,
            "x": "analysis_group",
            "y": "series_count",
        }
        if seaborn_at_least(0, 12):
            kwargs["errorbar"] = None
        else:
            kwargs["ci"] = None
        ax = sns.barplot(**kwargs)
    else:
        ax = plot_df.plot(
            kind="bar",
            x="analysis_group",
            y="series_count",
            legend=False,
            figsize=(max(10, 0.65 * len(plot_df) + 3), 5.8),
        ).axes
    ax.set_title("Series count by condition, dimensionality, plane, and field strength")
    ax.set_xlabel("Condition (dimensionality, plane, field strength)")
    ax.set_ylabel("Series count")
    ax.tick_params(axis="x", rotation=45)
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
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    if "field_strength_label" not in df:
        df = df.copy()
        df["field_strength_label"] = "UnknownField"
    values = pd.to_numeric(df.get(metric), errors="coerce")
    plot_df = df.loc[
        values.notna(),
        [
            "image_condition",
            "dimensionality",
            "scan_plane",
            "field_strength_label",
            "analysis_group",
        ],
    ].copy()
    plot_df[metric] = values[values.notna()]
    if not include_other:
        plot_df = plot_df[plot_df["image_condition"] != "Other"]
    if plot_df.empty or plot_df["image_condition"].nunique() == 0:
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"box_{safe_filename(metric)}.png"
    group_order = ordered_analysis_groups(plot_df)
    plot_df["analysis_group"] = pd.Categorical(
        plot_df["analysis_group"],
        categories=group_order,
        ordered=True,
    )

    plt.figure(figsize=(max(11, 0.72 * len(group_order) + 3), 5.8))
    if sns is not None:
        ax = sns.boxplot(
            data=plot_df,
            x="analysis_group",
            y=metric,
            showfliers=False,
            order=group_order,
        )
        sns.stripplot(
            data=plot_df,
            x="analysis_group",
            y=metric,
            order=group_order,
            color="black",
            alpha=0.22,
            size=2,
        )
    else:
        ax = plot_df.boxplot(
            column=metric,
            by="analysis_group",
            figsize=(max(11, 0.72 * len(group_order) + 3), 5.8),
        )
        plt.suptitle("")
    ax.set_title(label)
    ax.set_xlabel("Condition (dimensionality, plane, field strength)")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_slice_thickness_spacing_bubble_plots(
    df: pd.DataFrame,
    figures_dir: Path,
    include_other: bool,
) -> list[Path]:
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    required = ["slice_thickness_mm", "slice_spacing_mm"]
    if any(col not in df for col in required):
        return []
    if "scan_plane" not in df:
        df = df.copy()
        df["scan_plane"] = "Unknown"
    if "field_strength_label" not in df:
        df = df.copy()
        df["field_strength_label"] = "UnknownField"

    plot_df = df.copy()
    plot_df["slice_thickness_mm"] = pd.to_numeric(
        plot_df["slice_thickness_mm"], errors="coerce"
    ).round(3)
    plot_df["slice_spacing_mm"] = pd.to_numeric(
        plot_df["slice_spacing_mm"], errors="coerce"
    ).round(3)
    plot_df = plot_df.dropna(subset=required)
    if not include_other:
        plot_df = plot_df[plot_df["image_condition"] != "Other"]
    if plot_df.empty:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    split_cols = ["field_strength_label", "scan_plane", "dimensionality"]
    split_df = plot_df[split_cols].drop_duplicates()
    split_df["field_strength_label"] = pd.Categorical(
        split_df["field_strength_label"],
        categories=ordered_field_strength_values(plot_df["field_strength_label"]),
        ordered=True,
    )
    split_df["scan_plane"] = pd.Categorical(
        split_df["scan_plane"],
        categories=ordered_plane_values(plot_df["scan_plane"]),
        ordered=True,
    )
    split_df["dimensionality"] = pd.Categorical(
        split_df["dimensionality"],
        categories=ordered_dimension_values(plot_df["dimensionality"]),
        ordered=True,
    )
    split_df = split_df.sort_values(split_cols)

    for _, split_values in split_df.iterrows():
        field_strength = clean_text(split_values["field_strength_label"]) or "UnknownField"
        scan_plane = clean_text(split_values["scan_plane"]) or "Unknown"
        dimensionality = clean_text(split_values["dimensionality"]) or "Unknown"
        subset = plot_df[
            (plot_df["field_strength_label"] == field_strength)
            & (plot_df["scan_plane"] == scan_plane)
            & (plot_df["dimensionality"] == dimensionality)
        ]
        if subset.empty:
            continue

        counts = (
            subset.groupby(
                ["image_condition", "slice_thickness_mm", "slice_spacing_mm"],
                dropna=False,
            )
            .size()
            .reset_index(name="series_count")
        )
        if counts.empty:
            continue

        condition_order = ordered_condition_values(counts["image_condition"])
        n_conditions = len(condition_order)
        n_cols = min(3, max(n_conditions, 1))
        n_rows = int(math.ceil(n_conditions / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(8, 3.2 * n_cols), max(4.2, 3.0 * n_rows)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        max_count = max(int(counts["series_count"].max()), 1)
        x_min = float(counts["slice_thickness_mm"].min())
        x_max = float(counts["slice_thickness_mm"].max())
        y_min = float(counts["slice_spacing_mm"].min())
        y_max = float(counts["slice_spacing_mm"].max())
        x_margin = max((x_max - x_min) * 0.08, 0.15)
        y_margin = max((y_max - y_min) * 0.08, 0.15)
        if sns is not None:
            colors = sns.color_palette("tab10", n_colors=max(n_conditions, 1))
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_conditions, 1)))

        for condition_index, condition in enumerate(condition_order):
            ax = axes[condition_index // n_cols][condition_index % n_cols]
            condition_counts = counts[counts["image_condition"] == condition]
            sizes = 90 + 620 * (
                condition_counts["series_count"] / max_count
            ).pow(0.72)
            ax.scatter(
                condition_counts["slice_thickness_mm"],
                condition_counts["slice_spacing_mm"],
                s=sizes,
                color=colors[condition_index],
                alpha=0.58,
                edgecolors="black",
                linewidths=0.5,
            )
            for _, row in condition_counts.iterrows():
                ax.text(
                    row["slice_thickness_mm"],
                    row["slice_spacing_mm"],
                    str(int(row["series_count"])),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
            ax.set_title(clean_text(condition) or "Other")
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.grid(True, alpha=0.28)

        for empty_index in range(n_conditions, n_rows * n_cols):
            axes[empty_index // n_cols][empty_index % n_cols].axis("off")

        fig.suptitle(
            f"Slice thickness vs spacing counts: "
            f"{field_strength}, {scan_plane}, {dimensionality}"
        )
        fig.supxlabel("Slice thickness (mm)")
        fig.supylabel("Slice spacing (mm)")
        path = figures_dir / (
            "bubble_slice_thickness_vs_spacing"
            f"_field_{safe_filename(field_strength)}"
            f"_plane_{safe_filename(scan_plane)}"
            f"_dim_{safe_filename(dimensionality)}.png"
        )
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def save_categorical_count_heatmaps(
    df: pd.DataFrame,
    metric: str,
    label: str,
    figures_dir: Path,
    include_other: bool,
    max_values: int = 12,
) -> list[Path]:
    if "analysis_group" not in df:
        df = add_analysis_filter(df, None)
    if metric not in df:
        return []

    if "scan_plane" not in df:
        df = df.copy()
        df["scan_plane"] = "Unknown"
    if "field_strength_label" not in df:
        df = df.copy()
        df["field_strength_label"] = "UnknownField"
    plot_df = df[
        [
            "image_condition",
            "dimensionality",
            "scan_plane",
            "field_strength_label",
            "analysis_group",
            metric,
        ]
    ].copy()
    plot_df[metric] = plot_df[metric].map(clean_text)
    plot_df = plot_df[plot_df[metric] != ""]
    if not include_other:
        plot_df = plot_df[plot_df["image_condition"] != "Other"]
    if plot_df.empty:
        return []

    top_values_for_metric = plot_df[metric].value_counts().head(max_values).index
    plot_df = plot_df[plot_df[metric].isin(top_values_for_metric)]
    if plot_df.empty:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    split_cols = ["field_strength_label", "scan_plane", "dimensionality"]
    split_df = plot_df[split_cols].drop_duplicates()
    split_df["field_strength_label"] = pd.Categorical(
        split_df["field_strength_label"],
        categories=ordered_field_strength_values(plot_df["field_strength_label"]),
        ordered=True,
    )
    split_df["scan_plane"] = pd.Categorical(
        split_df["scan_plane"],
        categories=ordered_plane_values(plot_df["scan_plane"]),
        ordered=True,
    )
    split_df["dimensionality"] = pd.Categorical(
        split_df["dimensionality"],
        categories=ordered_dimension_values(plot_df["dimensionality"]),
        ordered=True,
    )
    split_df = split_df.sort_values(split_cols)

    for _, split_values in split_df.iterrows():
        field_strength = clean_text(split_values["field_strength_label"]) or "UnknownField"
        scan_plane = clean_text(split_values["scan_plane"]) or "Unknown"
        dimensionality = clean_text(split_values["dimensionality"]) or "Unknown"
        subset = plot_df[
            (plot_df["field_strength_label"] == field_strength)
            & (plot_df["scan_plane"] == scan_plane)
            & (plot_df["dimensionality"] == dimensionality)
        ]
        if subset.empty:
            continue

        condition_order = ordered_condition_values(subset["image_condition"])
        pivot = (
            subset.pivot_table(
                index="image_condition",
                columns=metric,
                values="analysis_group",
                aggfunc="count",
                fill_value=0,
            )
            .reindex(index=condition_order, columns=top_values_for_metric, fill_value=0)
            .astype(int)
        )
        if pivot.empty:
            continue

        path = figures_dir / (
            f"heatmap_{safe_filename(metric)}"
            f"_field_{safe_filename(field_strength)}"
            f"_plane_{safe_filename(scan_plane)}"
            f"_dim_{safe_filename(dimensionality)}.png"
        )
        plt.figure(
            figsize=(
                max(8, 0.7 * pivot.shape[1] + 3),
                max(4.2, 0.45 * pivot.shape[0] + 2),
            )
        )
        if sns is not None:
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar_kws={"label": "Series count"},
            )
        else:
            ax = plt.gca()
            image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
            plt.colorbar(image, ax=ax, label="Series count")
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            for row_index in range(pivot.shape[0]):
                for col_index in range(pivot.shape[1]):
                    ax.text(
                        col_index,
                        row_index,
                        str(pivot.iat[row_index, col_index]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
        ax.set_title(f"{label}: {field_strength}, {scan_plane}, {dimensionality}")
        ax.set_xlabel(label)
        ax.set_ylabel("Condition")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)

    return paths


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

    bubble_paths = save_slice_thickness_spacing_bubble_plots(
        df,
        figures_dir,
        include_other,
    )
    paths.extend(bubble_paths)

    matrix_paths = save_categorical_count_heatmaps(
        df,
        "matrix_size",
        "Matrix size",
        figures_dir,
        include_other,
    )
    paths.extend(matrix_paths)

    thickness_spacing_paths = save_categorical_count_heatmaps(
        df,
        "slice_thickness_spacing_mm",
        "Slice thickness x spacing (mm)",
        figures_dir,
        True,
    )
    paths.extend(thickness_spacing_paths)

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
    analysis_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    columns: dict[str, str | None],
    counts: pd.DataFrame,
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    figures: list[Path],
    encoding: str,
    max_slice_mm: float | None,
    exclude_slices_le: int | None,
) -> str:
    detected_rows = [
        {"logical_field": key, "csv_column": value or ""}
        for key, value in columns.items()
    ]
    detected = pd.DataFrame(detected_rows)

    condition_counts = (
        analysis_df["image_condition"]
        .value_counts(dropna=False)
        .rename_axis("image_condition")
        .reset_index(name="series_count")
    )
    dimension_counts = (
        analysis_df["dimensionality"]
        .value_counts(dropna=False)
        .rename_axis("dimensionality")
        .reset_index(name="series_count")
    )
    plane_counts = (
        analysis_df["scan_plane"]
        .value_counts(dropna=False)
        .rename_axis("scan_plane")
        .reset_index(name="series_count")
        if "scan_plane" in analysis_df
        else pd.DataFrame(columns=["scan_plane", "series_count"])
    )
    field_strength_counts = (
        analysis_df["field_strength_label"]
        .value_counts(dropna=False)
        .rename_axis("field_strength_label")
        .reset_index(name="series_count")
        if "field_strength_label" in analysis_df
        else pd.DataFrame(columns=["field_strength_label", "series_count"])
    )

    missing = []
    for metric in NUMERIC_METRICS:
        if metric in analysis_df:
            valid = pd.to_numeric(analysis_df[metric], errors="coerce").notna().sum()
            missing.append(
                {
                    "metric": metric,
                    "valid_count": int(valid),
                    "missing_percent": (len(analysis_df) - valid)
                    / max(len(analysis_df), 1)
                    * 100.0,
                }
            )
    if missing:
        missing_df = pd.DataFrame(missing).sort_values("missing_percent")
    else:
        missing_df = pd.DataFrame(columns=["metric", "valid_count", "missing_percent"])

    if stats.empty or "metric" not in stats:
        key_stats = pd.DataFrame()
    else:
        key_stats = stats[
            stats["metric"].isin(
                [
                    "slice_thickness_mm",
                    "slice_spacing_mm",
                    "slice_gap_mm",
                    "fov_row_mm",
                    "fov_col_mm",
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
                "scan_plane",
                "field_strength_label",
                "analysis_group",
                "metric",
                "valid_count",
                "median",
                "q1",
                "q3",
                "top_values",
            ]
        ]

    summary_display_columns = [
        "analysis_group",
        "scan_plane",
        "field_strength_label",
        "series_count",
        "slice_thickness_mm",
        "space_between_slices_mm",
        "common_slice_thickness_spacing_mm",
        "fov_row_mm",
        "fov_col_mm",
        "common_matrix_size",
        "matrix_source",
        "common_fov_mm",
        "common_pixel_spacing_mm",
    ]
    summary_display = summary[
        [column for column in summary_display_columns if column in summary]
    ].copy()

    figure_lines = "\n".join(
        f"- `{path.relative_to(output_dir)}`" for path in figures
    ) or "- No figures generated."
    if max_slice_mm is None or max_slice_mm <= 0:
        exclusion_line = "No slice-thickness/spacing exclusion threshold was applied."
    else:
        exclusion_line = (
            f"Rows with SliceThickness or SpacingBetweenSlices >= {max_slice_mm:g} mm "
            "were excluded from counts, statistics, and figures."
        )
    if exclude_slices_le is None or exclude_slices_le <= 0:
        slice_count_exclusion_line = "No low-slice-count exclusion threshold was applied."
    else:
        slice_count_exclusion_line = (
            f"Rows with NumberOfSlices <= {exclude_slices_le:g} were excluded from "
            "counts, statistics, and figures when that tag is available."
        )

    report = f"""# DICOM Tag Analysis Report

Input: `{input_path}`

Output directory: `{output_dir}`

Input rows: {len(df)}

Rows used for analysis: {len(analysis_df)}

Rows excluded from analysis: {len(excluded_df)}

Slice exclusion rule: {exclusion_line}

Low slice-count exclusion rule: {slice_count_exclusion_line}

CSV encoding: `{encoding}`

## Generated files

- `series_with_derived_columns.csv`
- `included_series_for_analysis.csv`
- `excluded_series_from_analysis.csv`
- `condition_dimension_counts.csv`
- `summary_by_condition_dimension.csv`
- `slice_thickness_spacing_counts.csv`
- `metric_stats_by_condition_dimension.csv`
- `metric_value_counts_by_condition_dimension.csv`
- `detected_columns.json`
- `analysis_report.md`
{figure_lines}

## Image condition counts

{markdown_table(condition_counts)}

## Dimensionality counts

{markdown_table(dimension_counts)}

## Plane counts

{markdown_table(plane_counts)}

## Field strength counts

{markdown_table(field_strength_counts)}

## Condition x dimensionality x plane x field strength counts

{markdown_table(counts)}

## Geometry Summary By Condition

{markdown_table(summary_display, max_rows=40)}

## Key metric summary

{markdown_table(key_stats, max_rows=40)}

## Metric missingness

{markdown_table(missing_df, max_rows=40)}

## Detected input columns

{markdown_table(detected, max_rows=80)}

## Notes

- Image condition labels are heuristic classifications based on non-description
  tags first, especially SequenceName, ScanningSequence, SequenceVariant,
  ScanOptions, ImageType, AcquisitionContrast, diffusion tags, TR, TE, TI, and
  FlipAngle. SeriesDescription and ProtocolName are ignored unless
  `--use-series-protocol-text` is used.
- 2D/3D labels are based mainly on MRAcquisitionType and non-description
  sequence/acquisition tags when present.
- Scan plane labels are based mainly on ImageOrientationPatient. PatientOrientation
  and AX/SAG/COR text are used as fallbacks; if these tags are absent, scan_plane
  will often be `Unknown`.
- Magnetic field strength groups are based on MagneticFieldStrength and formatted
  as labels such as `1.5T` or `3T`; missing values are grouped as `UnknownField`.
- Review `condition_reason` and `dimensionality_reason` in
  `series_with_derived_columns.csv` for questionable series.
- Rows with `analysis_excluded=True` are kept in `series_with_derived_columns.csv`
  but are not used for counts, statistics, or figures.
- Slice thickness vs spacing is shown as split bubble plots. Each point is one
  unique thickness/spacing pair, the label and bubble size show the series count,
  and panels are separated by condition.
- Heatmaps for matrix size and slice thickness x spacing are split into separate
  files by field strength, scan plane, and 2D/3D label. Slice thickness x spacing
  heatmaps include `Other` groups because they are intended as geometry QC views.
- If many rows are `Other` or `Unknown`, add more descriptive DICOM tag columns to
  the input CSV, especially SequenceName, ScanningSequence, SequenceVariant,
  ScanOptions, ImageType, AcquisitionContrast, DiffusionBValue, EchoTrainLength,
  EchoNumbers, and MRAcquisitionType.
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
    max_slice_mm: float | None,
    exclude_slices_le: int | None,
) -> dict[str, Path | list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "analysis_excluded" not in df:
        df = add_analysis_filter(df, max_slice_mm, exclude_slices_le)

    analysis_df = df[~df["analysis_excluded"]].copy()
    excluded_df = df[df["analysis_excluded"]].copy()

    counts = make_counts(analysis_df)
    stats = make_metric_stats(analysis_df)
    value_counts = make_value_counts(analysis_df)
    summary = make_condition_dimension_summary(analysis_df)
    thickness_spacing_counts = make_slice_thickness_spacing_counts(analysis_df)

    derived_path = output_dir / "series_with_derived_columns.csv"
    included_path = output_dir / "included_series_for_analysis.csv"
    excluded_path = output_dir / "excluded_series_from_analysis.csv"
    counts_path = output_dir / "condition_dimension_counts.csv"
    summary_path = output_dir / "summary_by_condition_dimension.csv"
    thickness_spacing_counts_path = output_dir / "slice_thickness_spacing_counts.csv"
    stats_path = output_dir / "metric_stats_by_condition_dimension.csv"
    value_counts_path = output_dir / "metric_value_counts_by_condition_dimension.csv"
    columns_path = output_dir / "detected_columns.json"
    report_path = output_dir / "analysis_report.md"

    write_csv(df, derived_path)
    write_csv(analysis_df, included_path)
    write_csv(excluded_df, excluded_path)
    write_csv(counts, counts_path)
    write_csv(summary, summary_path)
    write_csv(thickness_spacing_counts, thickness_spacing_counts_path)
    write_csv(stats, stats_path)
    write_csv(value_counts, value_counts_path)

    columns_path.write_text(
        json.dumps(columns, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figures: list[Path] = []
    if not no_plots:
        figures = make_figures(analysis_df, counts, output_dir, include_other_plots)

    report = make_report(
        input_path=input_path,
        output_dir=output_dir,
        df=df,
        analysis_df=analysis_df,
        excluded_df=excluded_df,
        columns=columns,
        counts=counts,
        stats=stats,
        summary=summary,
        figures=figures,
        encoding=encoding,
        max_slice_mm=max_slice_mm,
        exclude_slices_le=exclude_slices_le,
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "derived": derived_path,
        "included": included_path,
        "excluded": excluded_path,
        "counts": counts_path,
        "summary": summary_path,
        "thickness_spacing_counts": thickness_spacing_counts_path,
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
        "--exclude-slice-mm",
        type=float,
        default=DEFAULT_EXCLUDE_SLICE_MM,
        help=(
            "Exclude rows from counts, statistics, and figures when SliceThickness "
            "or SpacingBetweenSlices is greater than or equal to this value in mm. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--exclude-slices-le",
        type=int,
        default=DEFAULT_EXCLUDE_SLICES_LE,
        help=(
            "Exclude rows from counts, statistics, and figures when NumberOfSlices "
            "is less than or equal to this value. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--infer-dim-from-geometry",
        action="store_true",
        help=(
            "If textual 2D/3D hints are missing, infer 3D for thin contiguous "
            "slices and 2D for thicker slices."
        ),
    )
    parser.add_argument(
        "--use-series-protocol-text",
        action="store_true",
        help=(
            "Also use SeriesDescription and ProtocolName as a last-resort image "
            "condition fallback. By default, condition classification ignores them."
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
        use_series_protocol_text=args.use_series_protocol_text,
    )
    derived = add_analysis_filter(
        derived,
        args.exclude_slice_mm,
        args.exclude_slices_le,
    )
    outputs = save_outputs(
        input_path=input_path,
        df=derived,
        columns=columns,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
        include_other_plots=args.include_other_plots,
        encoding=encoding,
        max_slice_mm=args.exclude_slice_mm,
        exclude_slices_le=args.exclude_slices_le,
    )

    analyzed_count = int((~derived["analysis_excluded"]).sum())
    excluded_count = int(derived["analysis_excluded"].sum())
    print(f"Loaded {len(derived)} series")
    print(f"Analyzed {analyzed_count} series; excluded {excluded_count}")
    print(f"Report: {outputs['report']}")
    print(f"Derived CSV: {outputs['derived']}")
    print(f"Summary CSV: {outputs['summary']}")
    print(f"Thickness/spacing counts CSV: {outputs['thickness_spacing_counts']}")
    print(f"Stats CSV: {outputs['stats']}")
    if not args.no_plots:
        print(f"Figures: {len(outputs['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
