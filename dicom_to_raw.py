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

# 出力で算出して付け足す列（DICOMタグ以外の派生情報）。
# 先頭列 stem は保存した .raw/.hdr の名称（拡張子なし）と一致。
LEAD_COLUMNS = ["stem", "body_part"]
EXTRA_COLUMNS = ["output_file", "source_folder", "n_slices", "rows", "columns",
                 "pixel_spacing_row_mm", "pixel_spacing_col_mm", "slice_spacing_mm",
                 "image_position_first", "raw_dtype", "y_flipped",
                 "absolute_zyx", "reverse_z", "axis_order",
                 "split_label", "diffusion_b_value", "volume_in_series",
                 "raw_values", "raw_rescale_slope"]


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


# 同一シリーズ内で別ボリュームに分けるタグ（keyword と ファイル名ラベル）。
# これらは「1ボリューム内では一定・ボリューム間で変わる」量。--split-tags で追加可。
# "@bvalue" は複数ベンダーの b 値タグを横断検出する仮想スペック。
DEFAULT_SPLIT_TAGS = [
    ("@bvalue", "b"),                          # 拡散 b 値（標準/シーケンス/私的タグ横断）
    ("DiffusionGradientOrientation", "dir"),   # (0018,9089) 拡散傾斜方向（ベクトル）
    ("EchoNumbers", "e"),                      # (0018,0086) エコー番号（マルチエコー）
    ("TemporalPositionIdentifier", "t"),       # (0020,0100) 時相（ダイナミック）
]


def _num(v):
    """数値化（'\\'区切りの先頭値にも対応）。失敗時 None。"""
    try:
        return round(float(v), 4)
    except Exception:  # noqa: BLE001
        try:
            return round(float(str(v).split("\\")[0]), 4)
        except Exception:  # noqa: BLE001
            return None


def diffusion_bvalue(ds):
    """複数ベンダーの b 値タグを横断して b 値を取得する。無ければ None。
    標準(0018,9087) → MRDiffusionSequence(0018,9117) → Siemens(0019,100C)
    → Philips(2001,1003) → GE(0043,1039) の順。"""
    v = getattr(ds, "DiffusionBValue", None)
    n = _num(v) if v not in (None, "") else None
    if n is not None:
        return n
    seq = getattr(ds, "MRDiffusionSequence", None)
    if seq:
        for it in seq:
            v = getattr(it, "DiffusionBValue", None)
            n = _num(v) if v not in (None, "") else None
            if n is not None:
                return n
    for tag in [(0x0019, 0x100C), (0x2001, 0x1003)]:   # Siemens, Philips
        if tag in ds:
            n = _num(ds[tag].value)
            if n is not None:
                return n
    if (0x0043, 0x1039) in ds:                          # GE: 先頭値、+1e9オフセット除去
        val = ds[0x0043, 0x1039].value
        first = val[0] if isinstance(val, (list, pydicom.multival.MultiValue)) else val
        n = _num(first)
        if n is not None:
            if n >= 1e9:
                n -= 1e9
            return round(n, 4) if n >= 0 else None
    return None


def _resolve_tag(ds, spec):
    """spec（DICOMキーワード / 'gggg,eeee' / '@bvalue'）の値を取り出す。無ければ None。
    ベクトルは丸めた tuple、数値は丸めた float、その他は文字列で返す。"""
    if spec == "@bvalue":
        return diffusion_bvalue(ds)
    v = None
    if "," in spec:                            # 'gggg,eeee' 形式（私的タグ等）
        try:
            g, e = spec.split(",")
            tag = (int(g, 16), int(e, 16))
            if tag in ds:
                v = ds[tag].value
        except Exception:  # noqa: BLE001
            v = None
    else:
        v = getattr(ds, spec, None)
    if v is None or v == "":
        return None
    if isinstance(v, (list, pydicom.multival.MultiValue)):
        try:
            return tuple(round(float(x), 4) for x in v)
        except Exception:  # noqa: BLE001
            return tuple(str(x) for x in v)
    try:
        return round(float(v), 4)
    except Exception:  # noqa: BLE001
        return str(v)


def frame_key(ds, split_specs):
    """ボリュームを区別するキー（存在する split タグの (spec, value) タプル）。"""
    key = []
    for spec, _label in split_specs:
        val = _resolve_tag(ds, spec)
        if val is not None:
            key.append((spec, val))
    return tuple(key)


def split_by_frame(dsets, split_specs):
    """シリーズの DICOM 群を frame_key（b値/方向/エコー/時相）でサブグループ化。"""
    subs: dict = {}
    for d in dsets:
        subs.setdefault(frame_key(d, split_specs), []).append(d)
    return subs


def split_by_position_repeat(dsets):
    """同一スライス位置が N 回現れる＝N ボリューム、として分割（タグで分けきれない時の安全網）。
    各位置のフレームを b値→エコー→時相→取得番号→Instance番号 で並べ、i 番目を vol i に割当てる。
    戻り値: ボリュームごとの DICOM リスト。分割不要なら 1 要素。"""
    bypos: dict = {}
    for d in dsets:
        bypos.setdefault(round(slice_position(d), 3), []).append(d)
    if not bypos:
        return [list(dsets)]
    v = max(len(x) for x in bypos.values())
    if v <= 1:
        return [list(dsets)]

    def okey(d):
        b = diffusion_bvalue(d)
        return (b if b is not None else 0.0,
                _num(getattr(d, "EchoNumbers", 0)) or 0,
                _num(getattr(d, "TemporalPositionIdentifier", 0)) or 0,
                _num(getattr(d, "AcquisitionNumber", 0)) or 0,
                _num(getattr(d, "InstanceNumber", 0)) or 0)

    vols = [[] for _ in range(v)]
    for frames in bypos.values():
        for i, fr in enumerate(sorted(frames, key=okey)):
            vols[min(i, v - 1)].append(fr)
    return [x for x in vols if x]


def frame_suffix(fkey, split_specs, dir_index) -> str:
    """frame_key からファイル名サフィックス（例 _b1000, _e2, _t3, _dir5）を作る。"""
    label = {spec: lab for spec, lab in split_specs}
    parts = []
    for spec, val in fkey:
        lab = label.get(spec, spec.replace(",", ""))
        if isinstance(val, tuple):                 # 方向ベクトル等 → 通し番号
            parts.append(f"{lab}{dir_index.get((spec, val), 0)}")
        elif isinstance(val, float) and val == int(val):
            parts.append(f"{lab}{int(val)}")
        else:
            parts.append(f"{lab}{str(val).replace('.', 'p')}")
    return "_" + "_".join(parts) if parts else ""


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


def _quantize_real(vol: np.ndarray):
    """実値(float)ボリュームを 2byte 整数へ。値域に応じ符号と 10のべき乗スケールを決める。
    戻り値 (stored, dtype, raw_slope)。real = stored × raw_slope。
    通常データ（int16/uint16範囲）では raw_slope=1（raw=実値そのもの）。"""
    vmin, vmax = float(np.min(vol)), float(np.max(vol))
    signed = vmin < -0.5
    hi = 32000.0 if signed else 60000.0
    absmax = max(abs(vmin), abs(vmax))
    gain = 1.0
    while absmax * gain > hi:           # 桁が大きすぎる時だけ 1/10 ずつ縮小
        gain /= 10.0
    raw_slope = 1.0 / gain
    if signed:
        stored = np.clip(np.round(vol * gain), -32768, 32767).astype("<i2")
        dtype = "<i2"
    else:
        stored = np.clip(np.round(vol * gain), 0, 65535).astype("<u2")
        dtype = "<u2"
    return stored, dtype, raw_slope


def build_volume(dsets: list, apply_rescale: bool = True):
    """シリーズの DICOM 群 → スライス順ソート・3Dスタック。
    apply_rescale=True: 各スライスに RescaleSlope/Intercept を適用して実値(HU/信号)にし、
    2byte整数へ量子化（real = 格納値 × raw_slope）。False: DICOM格納値のまま。"""
    dsets = sorted(dsets, key=slice_position)
    rows = int(dsets[0].Rows)
    cols = int(dsets[0].Columns)
    dsets = [d for d in dsets if int(d.Rows) == rows and int(d.Columns) == cols]
    planes = []
    for d in dsets:
        try:
            px = np.asarray(d.pixel_array)
        except Exception:  # noqa: BLE001
            continue
        if apply_rescale:
            sl = float(getattr(d, "RescaleSlope", 1) or 1)
            ic = float(getattr(d, "RescaleIntercept", 0) or 0)
            px = px.astype(np.float64) * sl + ic       # 実値（per-slice 係数）
        planes.append(px)
    vol = np.stack(planes, axis=0)
    if apply_rescale:
        vol, dtype, raw_slope = _quantize_real(vol)
    else:
        signed = int(getattr(dsets[0], "PixelRepresentation", 0)) == 1
        dtype = "<i2" if signed else "<u2"
        raw_slope = 1.0
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
    return vol, dsets, dtype, dz, geom, raw_slope


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


def csv_row(ds, stem, out_file, src, vol, sx, sy, sz, dtype, info: dict) -> dict:
    nz, ny, nx = vol.shape
    row = {}
    for col, attr in TAG_COLUMNS:
        v = getattr(ds, attr, "")
        if isinstance(v, (list, pydicom.multival.MultiValue)):
            v = "\\".join(str(x) for x in v)
        row[col] = "" if v is None else str(v)
    ipp = getattr(ds, "ImagePositionPatient", "")
    row.update({
        "stem": stem,                              # 保存した .raw/.hdr の名称（拡張子なし）
        "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
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
        "split_label": info.get("split_label", ""),
        "diffusion_b_value": info.get("diffusion_b_value", ""),
        "volume_in_series": info.get("volume_in_series", ""),
        "raw_values": info.get("raw_values", ""),
        "raw_rescale_slope": info.get("raw_rescale_slope", ""),
    })
    return row


def _emit_volume(dsets, leaf, root, flip_y, absolute_zyx, reverse_z,
                 stem_extra, out_root, used, records, apply_rescale=True):
    """1 サブグループ（=1ボリューム）を向き補正して .raw/.hdr 出力＋CSV行を作る。"""
    vol, sorted_ds, dtype, dz, geom, raw_slope = build_volume(dsets, apply_rescale)
    tmpl = sorted_ds[0]
    ps = [float(x) for x in getattr(tmpl, "PixelSpacing", [1.0, 1.0])]
    sx, sy, sz = ps[1], ps[0], dz                  # 列(x), 行(y), スライス(z) 間隔
    info = {"y_flipped": False, "absolute_zyx": False, "reverse_z": reverse_z,
            "axis_order": "slice,row,col(取得順)",
            "raw_rescale_slope": f"{raw_slope:g}",
            "raw_values": "rescaled(real)" if apply_rescale else "stored(as-is)",
            **stem_extra.get("info", {})}

    if absolute_zyx:
        if geom is None:
            print(f"[warn] {leaf}: IOP/IPP無し → 絶対ZYX再配置できず取得順で出力")
        else:
            vol, (sx, sy, sz) = reorient_absolute_zyx(vol, geom)
            info.update(absolute_zyx=True, axis_order="Z(S-I),Y(A-P),X(L-R) LPS+")
    elif flip_y:
        vol = vol[:, ::-1, :]
        info["y_flipped"] = True
    if reverse_z:
        vol = vol[::-1, :, :]

    vol = np.ascontiguousarray(vol)
    stem = output_name(tmpl) + stem_extra.get("suffix", "")
    n = used.get(stem, 0)
    used[stem] = n + 1
    if n:
        stem = f"{stem}_{n+1}"
    out_base = os.path.join(out_root, stem)
    write_raw(vol, dtype, out_base, sx, sy, sz)
    records.append(csv_row(tmpl, stem, stem + ".raw", leaf, vol, sx, sy, sz, dtype, info))
    print(f"[ok] {os.path.relpath(leaf, root)}  -> {stem}.raw  "
          f"({vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}, {dtype})"
          f"{'  ' + info['split_label'] if info.get('split_label') else ''}"
          f"{'  ' + info['axis_order'] if absolute_zyx else ''}")


def process(root: str, out_root: str, flip_y: bool, absolute_zyx: bool = False,
            reverse_z: bool = False, split_specs=None, apply_rescale: bool = True) -> None:
    os.makedirs(out_root, exist_ok=True)
    split_specs = DEFAULT_SPLIT_TAGS if split_specs is None else split_specs
    leaves = find_leaf_dirs(root)
    print(f"[start] {len(leaves)} 最下層フォルダ -> {out_root}"
          f"{'  [absolute ZYX]' if absolute_zyx else ''}{'  [reverse-z]' if reverse_z else ''}"
          f"{'  [split-by:' + ','.join(s for s, _ in split_specs) + ']' if split_specs else '  [no-split]'}")
    records, used = [], {}
    for leaf in leaves:
        groups = read_dicoms(leaf)
        if not groups:
            continue
        for uid, dsets in groups.items():
            subs = split_by_frame(dsets, split_specs) if split_specs else {(): dsets}
            # frame_key で分けた各群を、さらに「位置リピート」で分割（安全網）。
            volumes = []                           # (fkey, dsets, posrep_idx, posrep_count)
            for fkey, sub in sorted(subs.items(), key=lambda kv: str(kv[0])):
                parts = split_by_position_repeat(sub) if split_specs else [sub]
                for pi, part in enumerate(parts):
                    volumes.append((fkey, part, pi, len(parts)))
            multi = len(volumes) > 1
            if multi:
                tag_v = len({fk for fk, _, _, _ in volumes})
                pos_v = any(pc > 1 for _, _, _, pc in volumes)
                print(f"[split] {os.path.relpath(leaf, root)} ({uid[:12]}…): "
                      f"1シリーズ → {len(volumes)} ボリューム"
                      f"{'（位置リピートで分割。タグ未検出のためラベルは v番号）' if pos_v and tag_v <= 1 else ''}")
            dir_index = {}
            if multi:
                dirvals = sorted({(s, v) for fk, _, _, _ in volumes for (s, v) in fk
                                  if isinstance(v, tuple)}, key=lambda x: str(x))
                dir_index = {sv: i for i, sv in enumerate(dirvals)}
            for vi, (fkey, sub, pi, pc) in enumerate(volumes):
                suffix = frame_suffix(fkey, split_specs, dir_index) if multi else ""
                if pc > 1:                         # 位置リピート分割分の通し番号
                    suffix += f"_v{pi}"
                bval = diffusion_bvalue(sub[0])    # 複数ベンダー横断で b 値取得
                extra = {"suffix": suffix, "info": {
                    "split_label": suffix.lstrip("_"),
                    "diffusion_b_value": "" if bval is None else
                    (int(bval) if bval == int(bval) else bval),
                    "volume_in_series": vi if multi else "",
                }}
                try:
                    _emit_volume(sub, leaf, root, flip_y, absolute_zyx, reverse_z,
                                 extra, out_root, used, records, apply_rescale)
                except Exception as e:  # noqa: BLE001
                    print(f"[skip] {leaf} ({uid[:12]}…){suffix}: {e}")

    n_vol = len(records)

    if records:
        cols = LEAD_COLUMNS + [c for c, _ in TAG_COLUMNS] + EXTRA_COLUMNS
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
    ap.add_argument("--split-tags", default=None,
                    help="同一シリーズ内を別ボリュームに分けるタグを追加（カンマ区切り）。"
                         "DICOMキーワード or 'gggg,eeee' 16進。既定の "
                         "b値/拡散方向/エコー/時相 に追加される。例 EchoTime,0019,100c")
    ap.add_argument("--no-split", action="store_true",
                    help="シリーズ内のサブグループ分割をしない（1シリーズ=1ボリューム）")
    ap.add_argument("--no-rescale", action="store_true",
                    help="RescaleSlope/Intercept を適用せず DICOM 格納値のまま出力"
                         "（既定は適用して実値=HU/信号で出力）")
    args = ap.parse_args()

    split_specs = [] if args.no_split else list(DEFAULT_SPLIT_TAGS)
    if args.split_tags and not args.no_split:
        toks = [t.strip() for t in args.split_tags.split(",") if t.strip()]
        i = 0
        while i < len(toks):
            # 'gggg,eeee' は2トークンに割れるので再結合
            if re.fullmatch(r"[0-9A-Fa-fx]{4}", toks[i]) and i + 1 < len(toks) \
                    and re.fullmatch(r"[0-9A-Fa-fx]{4}", toks[i + 1]):
                spec = f"{toks[i]},{toks[i+1]}"
                i += 2
            else:
                spec = toks[i]
                i += 1
            split_specs.append((spec, _san(spec.replace(",", ""))[:6] or "v"))

    process(args.input, args.out_root, flip_y=not args.no_flip_y,
            absolute_zyx=args.absolute_zyx, reverse_z=args.reverse_z,
            split_specs=split_specs, apply_rescale=not args.no_rescale)


if __name__ == "__main__":
    main()
