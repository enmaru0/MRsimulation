#!/usr/bin/env python3
"""
dicom_to_raw.py
===============
DICOM が保存されたフォルダ階層を再帰的にサーチし、**最下層（葉）フォルダごと**に
含まれる DICOM を 1 つの 3D 画像として読み込み、`.raw` / `.hdr` で出力する。
同時に、出力した各画像の DICOM タグを 1 行ずつ `summary_dicom_tag.csv` にまとめる。

出力
----
- <out_root>/患者ID_シリーズ_撮影日_時刻.raw : 3D ボリューム（2byte バイナリ、x最速→y→z）
- <out_root>/患者ID_シリーズ_撮影日_時刻.hdr : 'X Y Z 2 dx dy dz'（mm。recon_motion と同形式）
- <out_root>/summary_dicom_tag.csv          : 全出力画像のDICOMタグ（1画像=1行）

仕様
----
- 最下層フォルダ = サブフォルダを持たないフォルダ（os.walk で dirnames が空）。
- 1 フォルダに複数シリーズが混在する場合は SeriesInstanceUID ごとに分けて各々出力。
- スライス順は ImageOrientationPatient×ImagePositionPatient（法線方向の位置）で決定。
  無ければ SliceLocation → InstanceNumber の順でフォールバック。
- 画素は DICOM の格納値をそのまま 2byte（PixelRepresentation により int16/uint16）で保存。
  実値は magnitude/HU = 格納値×RescaleSlope + RescaleIntercept（CSV に記録）。
- raw ビューア向けに既定で行(y)を反転（--no-flip-y で無効化）。

使い方
------
    python dicom_to_raw.py <DICOMルート> [--out-root raw_out] [--no-flip-y]
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np
import pydicom


# CSV に出す DICOM タグ（列名, DICOM属性名）。全行同じ列順で揃える。
TAG_COLUMNS = [
    ("PatientID", "PatientID"),
    ("PatientName", "PatientName"),
    ("PatientBirthDate", "PatientBirthDate"),
    ("PatientSex", "PatientSex"),
    ("PatientAge", "PatientAge"),
    ("StudyDate", "StudyDate"),
    ("StudyTime", "StudyTime"),
    ("StudyDescription", "StudyDescription"),
    ("AccessionNumber", "AccessionNumber"),
    ("StudyInstanceUID", "StudyInstanceUID"),
    ("SeriesNumber", "SeriesNumber"),
    ("SeriesDescription", "SeriesDescription"),
    ("SeriesInstanceUID", "SeriesInstanceUID"),
    ("Modality", "Modality"),
    ("BodyPartExamined", "BodyPartExamined"),
    ("Manufacturer", "Manufacturer"),
    ("ManufacturerModelName", "ManufacturerModelName"),
    ("InstitutionName", "InstitutionName"),
    ("MagneticFieldStrength", "MagneticFieldStrength"),
    ("ProtocolName", "ProtocolName"),
    ("SequenceName", "SequenceName"),
    ("ScanningSequence", "ScanningSequence"),
    ("SequenceVariant", "SequenceVariant"),
    ("MRAcquisitionType", "MRAcquisitionType"),
    ("RepetitionTime", "RepetitionTime"),
    ("EchoTime", "EchoTime"),
    ("InversionTime", "InversionTime"),
    ("FlipAngle", "FlipAngle"),
    ("EchoTrainLength", "EchoTrainLength"),
    ("PixelBandwidth", "PixelBandwidth"),
    ("SliceThickness", "SliceThickness"),
    ("SpacingBetweenSlices", "SpacingBetweenSlices"),
    ("AcquisitionMatrix", "AcquisitionMatrix"),
    ("ImageOrientationPatient", "ImageOrientationPatient"),
    ("PatientPosition", "PatientPosition"),
    ("RescaleSlope", "RescaleSlope"),
    ("RescaleIntercept", "RescaleIntercept"),
    ("BitsStored", "BitsStored"),
    ("PixelRepresentation", "PixelRepresentation"),
    ("WindowCenter", "WindowCenter"),
    ("WindowWidth", "WindowWidth"),
]

# 出力で算出して付け足す列（DICOMタグ以外の派生情報）
EXTRA_COLUMNS = ["output_file", "source_folder", "n_slices", "rows", "columns",
                 "pixel_spacing_row_mm", "pixel_spacing_col_mm", "slice_spacing_mm",
                 "image_position_first", "raw_dtype", "y_flipped",
                 "absolute_zyx", "reverse_z", "axis_order"]


def _san(s, default="NA") -> str:
    """ファイル名用にサニタイズ（英数._- 以外を _ に。空なら default）。"""
    s = str(s).strip()
    if not s:
        return default
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s.strip("_") or default


def _tag(ds, name, default=""):
    v = getattr(ds, name, default)
    return default if v is None else v


def find_leaf_dirs(root: str):
    """サブフォルダを持たない（最下層の）ディレクトリを列挙。"""
    leaves = []
    for dirpath, dirnames, _ in os.walk(root):
        if not dirnames:
            leaves.append(dirpath)
    return sorted(leaves)


def read_dicoms(folder: str):
    """フォルダ内の DICOM を読み、SeriesInstanceUID ごとにグループ化して返す。"""
    groups: dict[str, list] = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path, force=True)
        except Exception:  # noqa: BLE001
            continue
        if "PixelData" not in ds or not hasattr(ds, "Rows"):
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "noseries"))
        groups.setdefault(uid, []).append(ds)
    return groups


def slice_position(ds) -> float:
    """スライス法線方向の位置を返す（ソート用）。"""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None and len(iop) == 6:
        r = np.array(iop[:3], float)
        c = np.array(iop[3:], float)
        n = np.cross(r, c)
        return float(np.dot(n, np.array(ipp, float)))
    sl = getattr(ds, "SliceLocation", None)
    if sl is not None:
        return float(sl)
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def build_volume(dsets: list):
    """シリーズの DICOM 群 → スライス順ソート・3Dスタック（格納値のまま）。"""
    dsets = sorted(dsets, key=slice_position)
    rows = int(dsets[0].Rows)
    cols = int(dsets[0].Columns)
    dsets = [d for d in dsets if int(d.Rows) == rows and int(d.Columns) == cols]
    signed = int(getattr(dsets[0], "PixelRepresentation", 0)) == 1
    dtype = "<i2" if signed else "<u2"
    planes = []
    for d in dsets:
        try:
            px = d.pixel_array
        except Exception:  # noqa: BLE001
            continue
        planes.append(np.asarray(px))
    vol = np.stack(planes, axis=0)
    # スライス間隔(mm): 位置の中央差 → 無ければ SpacingBetweenSlices/厚
    pos = np.array([slice_position(d) for d in dsets], float)
    if len(pos) > 1:
        dz = float(np.median(np.abs(np.diff(pos))))
    else:
        dz = 0.0
    if dz <= 0:
        dz = float(getattr(dsets[0], "SpacingBetweenSlices", 0) or
                   getattr(dsets[0], "SliceThickness", 0) or 1.0)

    # 患者座標(LPS)での各配列軸の方向ベクトル（絶対ZYX再配置用）。
    iop = getattr(dsets[0], "ImageOrientationPatient", None)
    geom = None
    if iop is not None and len(iop) == 6:
        row_dir = np.array(iop[0:3], float)    # 列index増加方向（=axis2/x）
        col_dir = np.array(iop[3:6], float)    # 行index増加方向（=axis1/y）
        n = np.cross(row_dir, col_dir)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 0 else n
        # スライス昇順ソート済み → slice index 増加方向は +n
        ps = [float(x) for x in getattr(dsets[0], "PixelSpacing", [1.0, 1.0])]
        geom = {
            "axis_dirs": np.array([n, col_dir, row_dir]),   # axis0(slice),1(row),2(col)
            "axis_sp": np.array([dz, ps[0], ps[1]]),        # 同順の間隔[mm]
        }
    return vol, dsets, dtype, dz, geom


# LPS 絶対座標の正方向: x=L=[1,0,0], y=P=[0,1,0], z=S=[0,0,1]
# 出力軸順 (axis0=Z=S, axis1=Y=P, axis2=X=L)
_LPS_TARGETS = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)


def reorient_absolute_zyx(vol: np.ndarray, geom: dict):
    """配列を患者絶対座標 LPS の ZYX 順（軸0=S-I, 軸1=A-P, 軸2=L-R, index増=+S/+P/+L）へ
    並べ替え（permute）＋反転（flip）する。リサンプルはしない（最近接軸へスナップ）。
    戻り値: (vol2, (sx, sy, sz))  ※sx=axis2間隔, sy=axis1, sz=axis0。"""
    dirs = geom["axis_dirs"]            # (3,3) 各入力軸の LPS 方向
    sps = geom["axis_sp"]
    perm, flips, used = [], [], set()
    for t in _LPS_TARGETS:             # 出力 axis0->S, axis1->P, axis2->L
        scores = [abs(float(np.dot(dirs[i], t))) if i not in used else -1.0
                  for i in range(3)]
        i = int(np.argmax(scores))
        used.add(i)
        perm.append(i)
        flips.append(float(np.dot(dirs[i], t)) < 0)
    vol2 = np.transpose(vol, perm)
    out_sp = sps[perm]
    for ax, fl in enumerate(flips):
        if fl:
            vol2 = np.flip(vol2, axis=ax)
    sz, sy, sx = float(out_sp[0]), float(out_sp[1]), float(out_sp[2])
    return np.ascontiguousarray(vol2), (sx, sy, sz)


def output_name(ds) -> str:
    """患者ID_シリーズ_撮影日_時刻 を作る。"""
    pid = _san(_tag(ds, "PatientID"), "noID")
    series = _san(_tag(ds, "SeriesNumber") or _tag(ds, "SeriesDescription"), "S")
    date = _san(_tag(ds, "StudyDate") or _tag(ds, "SeriesDate"), "nodate")
    time = _san(str(_tag(ds, "StudyTime") or _tag(ds, "SeriesTime")).split(".")[0], "notime")
    return f"{pid}_{series}_{date}_{time}"


def write_raw(vol: np.ndarray, dtype: str, out_base: str,
              sx: float, sy: float, sz: float):
    """向き補正済みボリュームを書き出す。sx=axis2(列), sy=axis1(行), sz=axis0(スライス)間隔[mm]。"""
    nz, ny, nx = vol.shape
    arr = np.ascontiguousarray(vol.astype(dtype))
    with open(out_base + ".raw", "wb") as f:
        f.write(arr.tobytes())
    with open(out_base + ".hdr", "w") as f:
        f.write(f"{nx} {ny} {nz} 2 {sx:g} {sy:g} {sz:g}")


def csv_row(ds, out_file, src, vol, sx, sy, sz, dtype, info: dict) -> dict:
    nz, ny, nx = vol.shape
    row = {}
    for col, attr in TAG_COLUMNS:
        v = getattr(ds, attr, "")
        if isinstance(v, (list, pydicom.multival.MultiValue)):
            v = "\\".join(str(x) for x in v)
        row[col] = "" if v is None else str(v)
    ipp = getattr(ds, "ImagePositionPatient", "")
    row.update({
        "output_file": out_file,
        "source_folder": src,
        "n_slices": nz, "rows": ny, "columns": nx,
        "pixel_spacing_row_mm": f"{sy:g}", "pixel_spacing_col_mm": f"{sx:g}",
        "slice_spacing_mm": f"{sz:g}",
        "image_position_first": "\\".join(str(x) for x in ipp) if ipp != "" else "",
        "raw_dtype": "int16" if dtype == "<i2" else "uint16",
        "y_flipped": info.get("y_flipped", False),
        "absolute_zyx": info.get("absolute_zyx", False),
        "reverse_z": info.get("reverse_z", False),
        "axis_order": info.get("axis_order", ""),
    })
    return row


def process(root: str, out_root: str, flip_y: bool,
            absolute_zyx: bool = False, reverse_z: bool = False) -> None:
    os.makedirs(out_root, exist_ok=True)
    leaves = find_leaf_dirs(root)
    print(f"[start] {len(leaves)} 最下層フォルダ -> {out_root}"
          f"{'  [absolute ZYX]' if absolute_zyx else ''}{'  [reverse-z]' if reverse_z else ''}")
    records, used = [], {}
    n_vol = 0
    for leaf in leaves:
        groups = read_dicoms(leaf)
        if not groups:
            continue
        for uid, dsets in groups.items():
            try:
                vol, sorted_ds, dtype, dz, geom = build_volume(dsets)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {leaf} ({uid[:12]}…): {e}")
                continue
            tmpl = sorted_ds[0]
            ps = [float(x) for x in getattr(tmpl, "PixelSpacing", [1.0, 1.0])]
            sx, sy, sz = ps[1], ps[0], dz          # 列(x), 行(y), スライス(z) 間隔
            info = {"y_flipped": False, "absolute_zyx": False,
                    "reverse_z": reverse_z, "axis_order": "slice,row,col(取得順)"}

            if absolute_zyx:
                if geom is None:
                    print(f"[warn] {leaf}: IOP/IPP無し → 絶対ZYX再配置できず取得順で出力")
                else:
                    vol, (sx, sy, sz) = reorient_absolute_zyx(vol, geom)
                    info.update(absolute_zyx=True, axis_order="Z(S-I),Y(A-P),X(L-R) LPS+")
            else:
                # 従来の raw ビューア向け y(行)反転（既定ON）
                if flip_y:
                    vol = vol[:, ::-1, :]
                    info["y_flipped"] = True

            if reverse_z:                          # スライス(z)方向を反転
                vol = vol[::-1, :, :]

            vol = np.ascontiguousarray(vol)
            stem = output_name(tmpl)
            n = used.get(stem, 0)
            used[stem] = n + 1
            if n:                                  # 同名衝突は連番付与
                stem = f"{stem}_{n+1}"
            out_base = os.path.join(out_root, stem)
            write_raw(vol, dtype, out_base, sx, sy, sz)
            records.append(csv_row(tmpl, stem + ".raw", leaf, vol, sx, sy, sz, dtype, info))
            n_vol += 1
            print(f"[ok] {os.path.relpath(leaf, root)}  -> {stem}.raw  "
                  f"({vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}, {dtype})"
                  f"{'  ' + info['axis_order'] if absolute_zyx else ''}")

    if records:
        cols = [c for c, _ in TAG_COLUMNS] + EXTRA_COLUMNS
        csv_path = os.path.join(out_root, "summary_dicom_tag.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(records)
        print(f"[csv] {csv_path}  ({len(records)} 画像)")
    print(f"[done] {n_vol} 画像 -> {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="DICOM を含むルートフォルダ（再帰探索）")
    ap.add_argument("--out-root", default="raw_out", help="出力先（既定 raw_out）")
    ap.add_argument("--no-flip-y", action="store_true",
                    help="raw の行(y)反転をしない（既定は反転。--absolute-zyx 時は無効）")
    ap.add_argument("--reverse-z", action="store_true",
                    help="出力スライス(z)方向を反転する")
    ap.add_argument("--absolute-zyx", action="store_true",
                    help="患者絶対座標 LPS の ZYX 順（軸0=S-I, 軸1=A-P, 軸2=L-R）へ"
                         "並べ替え＋反転（permute/flipのみ、リサンプル無し）。"
                         "撮像面(AX/COR/SAG)に依らず一定の向きで出力。"
                         "この時 y反転は適用しない（向きは座標で確定）")
    args = ap.parse_args()
    process(args.input, args.out_root, flip_y=not args.no_flip_y,
            absolute_zyx=args.absolute_zyx, reverse_z=args.reverse_z)


if __name__ == "__main__":
    main()
