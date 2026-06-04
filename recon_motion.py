#!/usr/bin/env python3
"""
recon_motion.py
===============
fastMRI 形式マルチコイル k-space (.h5) を再構成し、PNG または DICOM で出力する。
motion/ ・ multicoil_test/ ・ gre_data/ など、同形式のフォルダすべてに使える。

データ形式（fastMRI brain multicoil）
------------------------------------
- kspace: (slices, coils, H, W) complex64
- reconstruction_rss: (slices, H, W) float32  ← 正解再構成（検証用）
- ismrmrd_header: 撮像ジオメトリ/シーケンス情報（FOV, 厚, TR/TE, 磁場強度, UID 等）

再構成
------
各コイルを中心化 IFFT（norm="ortho"）→ コイル方向に root-sum-of-squares (RSS)。
これは fastMRI 標準で reconstruction_rss を厳密に再現する（検証で相対誤差 ~1e-7）。
モーションアーチファクト（位相エンコード方向のゴースト/ブレ）はそのまま画像に現れる。

出力（--format）
----------------
- png    : <out_root>/<basename>/sl00.png ...（8bit グレースケール、ボリューム最大値で正規化）
- dicom  : <out_root>/<basename>/sl00.dcm ...（MR Image Storage。ismrmrd_header の
           ジオメトリ＝PixelSpacing/SliceThickness/SpacingBetweenSlices/IPP/IOP と、
           シーケンス情報＝TR/TE/TI/FlipAngle/MagneticFieldStrength を埋め込む。
           画素=round(magnitude/RescaleSlope) の控えめスケール、RescaleSlope で magnitude 復元）
- binary : <out_root>/<basename>.raw/.hdr/.tag（3D ボリュームを 2byte 生バイナリ）
           .raw = int16 LE（x 最速→y→z、行yは既定で反転=上下を合わせる）、
           .hdr = 'X Y Z 2 dx dy dz'、.tag = 各種メタ情報（rescale_slope 含む）
- both   : png + dicom
- all    : png + dicom + binary

信号スケール: 画素は magnitude を過大にせず round(magnitude/rescale_slope) で格納し、
rescale_slope(10のべき乗) で magnitude=stored*rescale_slope と正確復元する（DICOMは
RescaleSlope タグ、binary は .tag）。

変換後、出力ルートに summary.csv（症例ごと1行：タグ/ジオメトリ/シーケンス/rescale_slope/
低磁場設定 等）を出力する。

対応形式
--------
- マルチコイル(brain): kspace (slices, coils, H, W)。コイル RSS。
- 単コイル(knee): kspace (slices, H, W)。|IFFT|。reconSpace へ中央クロップ、
  アンダーサンプリング(mask)はゼロ詰め再構成。
ヘッダの reconSpace マトリクスに合わせて IFFT 後に中央クロップする（例 640x368→320x320）。

低磁場シミュレーション（任意・k空間ドメイン・再構成前。出力の画素数は不変）
----------------------------------------------------------------------
- --acq-matrix N : 取得マトリクスを N×N（または R,C）へ。k空間中央をクロップして
  元グリッドへゼロ詰め＝解像度だけ落とし、出力画像の画素数は変えない。
- --lowfield-snr S : 複素ガウシアンノイズを付加して目標SNRへ（RSS後 Rician）。

使い方
------
    # PNG（既定）
    python recon_motion.py --in-root motion --out-root motion_png
    # 単コイル膝データ
    python recon_motion.py --in-root singlecoil_test --out-root singlecoil_test_png
    # DICOM
    python recon_motion.py --in-root gre_data --out-root gre_data_dicom --format dicom
    # binary(.raw/.hdr/.tag)
    python recon_motion.py --in-root gre_data --out-root gre_data_raw --format binary
    # 低磁場シミュレーション: 取得192×192へ落として再構成（出力画素数は不変）
    python recon_motion.py --in-root singlecoil_test --out-root singlecoil_lf \
        --acq-matrix 192
    # 全形式・先頭2ファイルだけ試す
    python recon_motion.py --in-root motion --out-root out --format all --limit 2

※ これらの k-space と再構成画像は患者データのため git に push しない（.gitignore 済み）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
STORE_MAX = 32000          # 格納整数の上限（int16安全域 ±32767 内）


def rescale_slope_for(data_max: float, lo: float = 1000.0,
                      hi: float = float(STORE_MAX)) -> float:
    """ボリュームの実最大値から rescale_slope(=10のべき乗) を決める。

    格納値 = round(magnitude / slope) が [lo, hi] に収まるよう slope を選ぶ。
    これにより magnitude = 格納値 * slope で実信号を復元でき、かつ格納値が
    過大(従来の65000等)にならない。データセット間で振幅が桁違い
    （ブレイン ~1e2、単コイル膝 ~1e-4）でも自動で合う。
    """
    if data_max <= 0:
        return 1.0
    gain = 1.0
    while data_max * gain < lo:
        gain *= 10.0
    while data_max * gain > hi:
        gain /= 10.0
    return 1.0 / gain


# ----------------------------- 再構成 -----------------------------
def ifft2c(x: np.ndarray) -> np.ndarray:
    """中心化 2D 逆FFT（fastMRI 標準、norm="ortho"）。最後の2軸に作用。"""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def center_crop(arr: np.ndarray, out_hw) -> np.ndarray:
    """最後の2軸を中央 (th, tw) にクロップ（fastMRI の reconSpace への切り出し）。"""
    if out_hw is None:
        return arr
    H, W = arr.shape[-2:]
    th, tw = min(int(out_hw[0]), H), min(int(out_hw[1]), W)
    r0, c0 = (H - th) // 2, (W - tw) // 2
    return arr[..., r0:r0 + th, c0:c0 + tw]


def kspace_crop_zerofill(ks: np.ndarray, acq_hw) -> np.ndarray:
    """k空間中央を (tr, tc) にクロップ→元グリッドへゼロ詰め。
    取得マトリクスを (tr,tc) に落としつつ、出力画像のグリッド/解像度(画素数)は不変。
    （DC は中心レイアウト前提。本データは fastMRI 規約で中心 DC）
    """
    H, W = ks.shape[-2:]
    tr, tc = min(int(acq_hw[0]), H), min(int(acq_hw[1]), W)
    r0, c0 = (H - tr) // 2, (W - tc) // 2
    out = np.zeros_like(ks)
    out[..., r0:r0 + tr, c0:c0 + tc] = ks[..., r0:r0 + tr, c0:c0 + tc]
    return out


def reconstruct(kspace: np.ndarray, recon_size=None, acq_matrix=None,
                snr: float | None = None, rng=None) -> np.ndarray:
    """k空間 → magnitude 画像。マルチコイル(S,C,H,W)はRSS、単コイル(S,H,W)は|.|。

    低磁場シミュレーション（任意・k空間ドメイン、出力画素数は不変）:
    - acq_matrix=(tr,tc): 取得マトリクスを中央クロップで落とす（解像度低下）
    - snr: 複素ガウシアンノイズを付加して目標SNRへ（RSS後 Rician/noncentral-chi）
    recon_size=(rows,cols): IFFT後に中央クロップ（fastMRI reconSpace。Noneで無し）
    """
    ks = kspace
    if acq_matrix is not None:
        ks = kspace_crop_zerofill(ks, acq_matrix)

    imgs = ifft2c(ks)
    multicoil = (kspace.ndim == 4)

    def to_mag(im):
        return np.sqrt((np.abs(im) ** 2).sum(axis=1)) if multicoil else np.abs(im)

    if snr is not None and snr > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        mag0 = to_mag(imgs)
        fg = mag0[mag0 > mag0.mean()]
        sig = float(np.median(fg)) if fg.size else float(mag0.max())
        n_coil = kspace.shape[1] if multicoil else 1
        # 単コイルでは magnitude ノイズ std ≈ σ。狙い SNR=sig/σ。
        # マルチコイルRSSは結合で SNR が上がるため 1/√C で一次近似（近似である旨を注記）。
        sigma = sig / (snr * np.sqrt(2.0) * np.sqrt(n_coil))
        imgs = imgs + rng.normal(0.0, sigma, imgs.shape) \
                    + 1j * rng.normal(0.0, sigma, imgs.shape)

    mag = to_mag(imgs)
    return center_crop(mag, recon_size)


# ----------------------------- ヘッダ解析 -----------------------------
def _local(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _find(parent, tag):
    if parent is None:
        return None
    for el in parent.iter():
        if _local(el) == tag:
            return el
    return None


def _ftext(parent, tag, default=None):
    el = _find(parent, tag)
    if el is not None and el.text and el.text.strip():
        return el.text.strip()
    return default


def parse_header(raw) -> dict:
    """ismrmrd_header(XML) から DICOM 生成に必要な情報を抽出。失敗時は既定値。"""
    meta = {
        "rows": None, "cols": None, "dx": None, "dy": None,
        "thickness": 5.0, "spacing": 6.0,
        "patient_position": "HFS", "field_strength": None,
        "tr": None, "te": None, "ti": None, "flip": None,
        "vendor": None, "model": None, "institution": None,
        "protocol": None, "sequence": None,
        "series_uid": None, "study_uid": None, "for_uid": None,
        "study_time": None,
    }
    if raw is None:
        return meta
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return meta

    recon = _find(root, "reconSpace")
    if recon is None:
        recon = _find(root, "encodedSpace")
    mat = _find(recon, "matrixSize")
    fov = _find(recon, "fieldOfView_mm")
    if mat is not None:
        mx, my = _ftext(mat, "x"), _ftext(mat, "y")
        meta["cols"] = int(mx) if mx else None
        meta["rows"] = int(my) if my else None
    if fov is not None and mat is not None:
        fx, fy = _ftext(fov, "x"), _ftext(fov, "y")
        if fx and meta["cols"]:
            meta["dx"] = float(fx) / meta["cols"]
        if fy and meta["rows"]:
            meta["dy"] = float(fy) / meta["rows"]

    enc = _find(root, "encoding")
    meta["thickness"] = float(_ftext(enc, "sliceThickness", meta["thickness"]))
    meta["spacing"] = float(_ftext(enc, "spacingBetweenSlices", meta["spacing"]))

    meta["patient_position"] = _ftext(root, "patientPosition", meta["patient_position"])
    fs = _ftext(root, "systemFieldStrength_T")
    meta["field_strength"] = float(fs) if fs else None
    meta["vendor"] = _ftext(root, "systemVendor")
    meta["model"] = _ftext(root, "systemModel")
    meta["institution"] = _ftext(root, "institutionName")
    meta["protocol"] = _ftext(root, "protocolName")

    seq = _find(root, "sequenceParameters")
    for key, tag in (("tr", "TR"), ("te", "TE"), ("ti", "TI"), ("flip", "flipAngle_deg")):
        v = _ftext(seq, tag)
        meta[key] = float(v) if v else None
    meta["sequence"] = _ftext(seq, "sequence_type")

    meta["series_uid"] = _ftext(root, "seriesInstanceUID")
    meta["study_uid"] = _ftext(root, "studyUID")
    meta["for_uid"] = _ftext(root, "frameOfReferenceUID")
    meta["study_time"] = _ftext(root, "studyTime")
    return meta


def _uid_or_new(s):
    return s if (s and _UID_RE.match(s) and len(s) <= 64) else generate_uid()


# ----------------------------- 出力 -----------------------------
def to_uint8(vol: np.ndarray, vmax: float) -> np.ndarray:
    """[0, vmax] を [0,255] にクリップ・量子化。"""
    v = np.clip(vol / (vmax + 1e-9), 0.0, 1.0)
    return (v * 255.0 + 0.5).astype(np.uint8)


def save_png(vol: np.ndarray, vmax: float, out_dir: str) -> None:
    u8 = to_uint8(vol, vmax)
    for i in range(u8.shape[0]):
        Image.fromarray(u8[i], mode="L").save(os.path.join(out_dir, f"sl{i:02d}.png"))


def save_dicom(vol: np.ndarray, rescale_slope: float, data_max: float, meta: dict,
               basename: str, patient_id: str, acquisition: str, out_dir: str,
               flip_y: bool = True, reverse_slices: bool = True) -> None:
    """再構成 magnitude を MR Image Storage DICOM 群として書き出す。

    信号スケール: 格納画素 = round(magnitude / rescale_slope)（控えめな整数）。
    RescaleSlope を設定するので pixel_array * RescaleSlope = 元 magnitude。
    向き補正（既定ON）:
    - flip_y: 行(y)を反転（上下を合わせる。raw と同方向）
    - reverse_slices: スライス積層方向を反転（InstanceNumber/IPP-z の向きを合わせる）
    """
    if flip_y:
        vol = vol[:, ::-1, :]
    if reverse_slices:
        vol = vol[::-1, :, :]
    vol = np.ascontiguousarray(vol)

    n, nrows, ncols = vol.shape
    dx = meta["dx"] or 1.0          # 列方向(x)画素間隔 mm
    dy = meta["dy"] or 1.0          # 行方向(y)画素間隔 mm
    spacing = meta["spacing"] or float(meta["thickness"])
    thickness = meta["thickness"]

    # 格納整数 = magnitude / rescale_slope（過大スケールにしない）
    inv = 1.0 / (rescale_slope + 1e-30)

    series_uid = _uid_or_new(meta["series_uid"])
    study_uid = _uid_or_new(meta["study_uid"])
    for_uid = _uid_or_new(meta["for_uid"])

    # 軸位(HFS)前提のジオメトリ。FOV 中心を原点に配置。
    fovx, fovy = dx * ncols, dy * nrows
    x0 = -fovx / 2.0 + dx / 2.0
    y0 = -fovy / 2.0 + dy / 2.0
    z0 = -(n - 1) * spacing / 2.0
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    date_part = basename.split("_")[0]
    study_date = date_part[:8] if date_part[:8].isdigit() else ""
    m = re.search(r"(\d+)$", basename)
    series_number = int(m.group(1)) if m else 1
    series_desc = f"{meta.get('protocol') or acquisition} (RSS recon)"

    for k in range(n):
        stored = np.clip(np.round(vol[k] * inv), 0, STORE_MAX).astype("<u2")
        sop_uid = generate_uid()

        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
        fm.MediaStorageSOPInstanceUID = sop_uid
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = generate_uid()

        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = MR_IMAGE_STORAGE
        ds.SOPInstanceUID = sop_uid
        ds.Modality = "MR"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.PatientName = patient_id or basename
        ds.PatientID = patient_id or basename
        ds.StudyID = (meta.get("study_uid") or "")[:16]
        ds.StudyDate = study_date
        ds.StudyTime = meta.get("study_time") or ""
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1
        ds.ImageType = ["DERIVED", "SECONDARY", "MFSPLIT"]

        if meta.get("vendor"):
            ds.Manufacturer = meta["vendor"].strip()
        if meta.get("model"):
            ds.ManufacturerModelName = meta["model"]
        if meta.get("institution"):
            ds.InstitutionName = meta["institution"]
        ds.SeriesDescription = series_desc
        if meta.get("protocol"):
            ds.ProtocolName = meta["protocol"]
        if meta.get("sequence"):
            ds.SequenceName = meta["sequence"][:16]
        if meta.get("field_strength") is not None:
            ds.MagneticFieldStrength = meta["field_strength"]
        for tag, key in (("RepetitionTime", "tr"), ("EchoTime", "te"),
                         ("InversionTime", "ti"), ("FlipAngle", "flip")):
            if meta.get(key) is not None:
                setattr(ds, tag, meta[key])
        ds.PatientPosition = meta.get("patient_position") or "HFS"

        # ジオメトリ
        ds.PixelSpacing = [float(dy), float(dx)]      # [行間隔, 列間隔]
        ds.SliceThickness = float(thickness)
        ds.SpacingBetweenSlices = float(spacing)
        ds.ImageOrientationPatient = iop
        z = z0 + k * spacing
        ds.ImagePositionPatient = [float(x0), float(y0), float(z)]
        ds.SliceLocation = float(z)

        # 画素
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = int(nrows)
        ds.Columns = int(ncols)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = float(rescale_slope)        # pixel*slope = 元 magnitude
        ds.RescaleIntercept = 0.0
        ds.RescaleType = "US"
        # Window/Level は rescale 後（magnitude）の単位で指定
        ds.WindowCenter = float(data_max * 0.5)
        ds.WindowWidth = float(max(data_max, 1e-6))
        ds.PixelData = stored.tobytes()

        ds.save_as(os.path.join(out_dir, f"sl{k:02d}.dcm"), enforce_file_format=True)


def save_binary(vol: np.ndarray, rescale_slope: float, data_max: float, meta: dict,
                basename: str, patient_id: str, acquisition: str, out_base: str,
                flip_y: bool = True) -> None:
    """3D ボリュームを 2byte の .raw として保存し、.hdr / .tag を併記する。

    - <out_base>.raw : signed int16 リトルエンディアン。並びは x が最速→ y → z（C順, shape (z,y,x)）
    - <out_base>.hdr : 'xサイズ yサイズ zサイズ 2 x_spacing y_spacing z_spacing'
    - <out_base>.tag : 撮像/再構成の各種メタ情報（key: value）

    信号スケール: 格納値 = round(magnitude / rescale_slope)。過大スケールにせず、
    `.tag` の rescale_slope を使って magnitude = 格納値 * rescale_slope で復元する。
    2byte を符号付き short(int16) として読むビューア向けに int16範囲(±32767)に収める。
    向き: 多くの .raw ビューアは行原点が下(bottom-up)のため、既定で y(行) を反転して
    上下が正しく表示されるようにする（flip_y=False で無効化）。
    """
    nz, ny, nx = vol.shape
    dx = meta["dx"] or 1.0                       # 列(x)方向の物理スペーシング mm
    dy = meta["dy"] or 1.0                       # 行(y)方向の物理スペーシング mm
    dz = meta["spacing"] or meta["thickness"] or 1.0   # スライス(z)方向 mm

    inv = 1.0 / (rescale_slope + 1e-30)          # 格納値 = magnitude / slope
    stored = np.clip(np.round(vol * inv), 0, STORE_MAX).astype("<i2")  # signed int16 (z,y,x)

    if flip_y:
        stored = stored[:, ::-1, :]              # 行(y)を反転 → 原点を下に（上下を合わせる）

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    # .raw （x 最速で連続）
    with open(out_base + ".raw", "wb") as f:
        f.write(np.ascontiguousarray(stored).tobytes())   # C順: 最終軸 x が最速

    # .hdr （サイズ x y z, 2byte, 物理スペーシング x y z。末尾スペースなし）
    with open(out_base + ".hdr", "w") as f:
        f.write(f"{nx} {ny} {nz} 2 {dx:g} {dy:g} {dz:g}")

    # .tag （各種情報）
    slope = rescale_slope                        # stored*slope = 元 magnitude
    lines = [
        f"basename: {basename}",
        f"acquisition: {acquisition}",
        f"patient_id: {patient_id}",
        "modality: MR",
        "recon: coil-RSS (IFFT norm=ortho, fastMRI)",
        f"dims_xyz: {nx} {ny} {nz}",
        f"voxel_spacing_mm_xyz: {dx:g} {dy:g} {dz:g}",
        f"slice_thickness_mm: {meta['thickness']:g}",
        f"spacing_between_slices_mm: {meta['spacing']:g}",
        "data_type: int16",
        "byte_order: little_endian",
        "bytes_per_voxel: 2",
        f"voxel_order: x_fastest_then_y_then_z{'  (y flipped: origin bottom-left)' if flip_y else ''}",
        f"intensity_max: {data_max:g}",
        f"stored_max: {int(np.max(stored))}",
        f"rescale_slope: {slope:g}",
        "rescale_note: magnitude = stored_value * rescale_slope",
        f"patient_position: {meta.get('patient_position') or ''}",
        f"field_strength_T: {meta['field_strength'] if meta['field_strength'] is not None else ''}",
        f"TR_ms: {meta['tr'] if meta['tr'] is not None else ''}",
        f"TE_ms: {meta['te'] if meta['te'] is not None else ''}",
        f"TI_ms: {meta['ti'] if meta['ti'] is not None else ''}",
        f"flip_deg: {meta['flip'] if meta['flip'] is not None else ''}",
        f"protocol: {meta.get('protocol') or ''}",
        f"sequence: {meta.get('sequence') or ''}",
        f"manufacturer: {(meta.get('vendor') or '').strip()}",
        f"model: {meta.get('model') or ''}",
        f"institution: {meta.get('institution') or ''}",
        f"series_uid: {meta.get('series_uid') or ''}",
        f"study_uid: {meta.get('study_uid') or ''}",
        f"frame_of_reference_uid: {meta.get('for_uid') or ''}",
    ]
    with open(out_base + ".tag", "w") as f:
        f.write("\n".join(lines) + "\n")


def process_file(path: str, in_root: str, out_root: str, fmts: set,
                 acq_matrix=None, snr=None, rng=None, raw_flip_y=True,
                 dicom_flip_y=True, dicom_reverse_slices=True) -> int:
    rel = os.path.relpath(path, in_root)
    base = os.path.splitext(rel)[0]            # 例 inter-scan_motion/2022061401_T101

    with h5py.File(path, "r") as h:
        ks = h["kspace"][:]
        attr_max = float(h.attrs["max"]) if "max" in h.attrs else None
        acq = h.attrs.get("acquisition", "?")
        pid = h.attrs.get("patient_id", "")
        raw_hdr = h["ismrmrd_header"][()] if "ismrmrd_header" in h else None

    # ヘッダは常に解析（recon サイズ＝reconSpace への中央クロップに必要。単コイル膝など）
    meta = parse_header(raw_hdr)
    recon_size = (meta["rows"], meta["cols"]) if meta["rows"] and meta["cols"] else None

    rss = reconstruct(ks, recon_size=recon_size, acq_matrix=acq_matrix,
                      snr=snr, rng=rng)
    data_max = float(rss.max())
    vmax = attr_max if attr_max is not None else data_max     # PNG 正規化用
    rescale_slope = rescale_slope_for(data_max)               # magnitude = stored*slope
    lf = ""
    if acq_matrix is not None:
        lf += f" acq={acq_matrix[0]}x{acq_matrix[1]}"
    if snr is not None:
        lf += f" snr={snr}"

    # png / dicom はスライス画像 → <out_root>/<base>/slNN.*
    if fmts & {"png", "dicom"}:
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)
        if "png" in fmts:
            save_png(rss, vmax, out_dir)
        if "dicom" in fmts:
            save_dicom(rss, rescale_slope, data_max, meta, os.path.basename(base),
                       pid, acq, out_dir,
                       flip_y=dicom_flip_y, reverse_slices=dicom_reverse_slices)

    # binary は 1 ボリューム = <out_root>/<base>.raw/.hdr/.tag
    if "binary" in fmts:
        save_binary(rss, rescale_slope, data_max, meta, os.path.basename(base),
                    pid, acq, os.path.join(out_root, base), flip_y=raw_flip_y)

    print(f"[ok] {rel}  ({acq})  {rss.shape[0]} slices {rss.shape[1]}x{rss.shape[2]}  "
          f"[{'+'.join(sorted(fmts))}]{lf}")

    nz, ny, nx = rss.shape
    return {
        "file": rel,
        "basename": os.path.basename(base),
        "acquisition": acq,
        "patient_id": pid,
        "n_slices": nz,
        "nx": nx, "ny": ny, "nz": nz,
        "dx_mm": round(meta["dx"], 5) if meta["dx"] else "",
        "dy_mm": round(meta["dy"], 5) if meta["dy"] else "",
        "dz_mm": round(meta["spacing"], 5) if meta["spacing"] else "",
        "slice_thickness_mm": meta["thickness"],
        "spacing_mm": meta["spacing"],
        "field_strength_T": meta["field_strength"] if meta["field_strength"] is not None else "",
        "TR_ms": meta["tr"] if meta["tr"] is not None else "",
        "TE_ms": meta["te"] if meta["te"] is not None else "",
        "TI_ms": meta["ti"] if meta["ti"] is not None else "",
        "flip_deg": meta["flip"] if meta["flip"] is not None else "",
        "protocol": meta.get("protocol") or "",
        "sequence": meta.get("sequence") or "",
        "manufacturer": (meta.get("vendor") or "").strip(),
        "model": meta.get("model") or "",
        "institution": meta.get("institution") or "",
        "intensity_max": round(data_max, 6),
        "rescale_slope": rescale_slope,
        "stored_max": int(round(data_max / rescale_slope)),
        "formats": "+".join(sorted(fmts)),
        "acq_matrix": f"{acq_matrix[0]}x{acq_matrix[1]}" if acq_matrix else "",
        "lowfield_snr": snr if snr is not None else "",
        "series_uid": meta.get("series_uid") or "",
        "study_uid": meta.get("study_uid") or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-root", default="motion", help="入力 .h5 ルート")
    ap.add_argument("--out-root", default="motion_png", help="出力ルート")
    ap.add_argument("--format", choices=["png", "dicom", "binary", "both", "all"],
                    default="png",
                    help="出力形式（png/dicom/binary、both=png+dicom、all=全部。既定 png）")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N ファイルのみ（0=全部）")
    # 低磁場シミュレーション（k空間ドメイン・再構成前。出力の画素数は不変）
    ap.add_argument("--acq-matrix", default=None,
                    help="取得マトリクスを N または R,C に落とす（k空間中央クロップ＋ゼロ詰め）。"
                         "解像度低下のみで出力画素数は不変。例: 192 / 192,192")
    ap.add_argument("--lowfield-snr", type=float, default=None,
                    help="低磁場ノイズを付加し目標SNRへ（複素ガウシアン。単コイルで厳密、"
                         "マルチコイルRSSは近似）")
    ap.add_argument("--seed", type=int, default=0, help="ノイズ乱数シード")
    ap.add_argument("--raw-no-flip-y", action="store_true",
                    help="binary(.raw)出力で行(y)反転をしない（既定は反転して上下を合わせる）")
    ap.add_argument("--dicom-no-flip-y", action="store_true",
                    help="DICOM出力で行(y)反転をしない（既定は反転して上下を合わせる）")
    ap.add_argument("--dicom-no-reverse-slices", action="store_true",
                    help="DICOM出力でスライス方向を反転しない（既定は反転して積層方向を合わせる）")
    args = ap.parse_args()

    fmt_map = {
        "png": {"png"}, "dicom": {"dicom"}, "binary": {"binary"},
        "both": {"png", "dicom"}, "all": {"png", "dicom", "binary"},
    }
    fmts = fmt_map[args.format]

    acq_matrix = None
    if args.acq_matrix:
        parts = [int(v) for v in str(args.acq_matrix).replace("x", ",").split(",")]
        acq_matrix = (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])
    rng = np.random.default_rng(args.seed) if args.lowfield_snr else None

    files = sorted(glob.glob(os.path.join(args.in_root, "**", "*.h5"), recursive=True))
    if args.limit:
        files = files[:args.limit]
    lf_desc = ""
    if acq_matrix:
        lf_desc += f"  lowfield: acq-matrix={acq_matrix[0]}x{acq_matrix[1]}"
    if args.lowfield_snr:
        lf_desc += f" snr={args.lowfield_snr}"
    print(f"[start] {len(files)} files -> {args.out_root}  format={args.format}{lf_desc}")

    records = []
    n_files = n_slices = 0
    for p in files:
        try:
            rec = process_file(p, args.in_root, args.out_root, fmts,
                               acq_matrix=acq_matrix, snr=args.lowfield_snr, rng=rng,
                               raw_flip_y=not args.raw_no_flip_y,
                               dicom_flip_y=not args.dicom_no_flip_y,
                               dicom_reverse_slices=not args.dicom_no_reverse_slices)
            records.append(rec)
            n_slices += rec["n_slices"]
            n_files += 1
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {p}: {e}")

    # summary.csv（症例ごとに1行）
    if records:
        os.makedirs(args.out_root, exist_ok=True)
        csv_path = os.path.join(args.out_root, "summary.csv")
        cols = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(records)
        print(f"[csv] {csv_path}  ({len(records)} cases)")

    print(f"[done] {n_files} files, {n_slices} slices -> {args.out_root} ({args.format})")


if __name__ == "__main__":
    main()
