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

向き: 簡易 fastMRI 形式にはスライス毎の IOP/IPP が無く、参照できる向き情報は
patientPosition(HFS/FFS)のみ。これでスライス積層方向を決め、raw/DICOM/PNG を一貫させる
（HFS/FFS で S-I/L-R が反転するため入力依存の反転を解消）。--flip-y/--flip-x/--reverse-slices
の auto/on/off で上書き可。

変換後、出力ルートに summary.csv（症例ごと1行：タグ/ジオメトリ/シーケンス/rescale_slope/
向き/低磁場設定 等）を出力する。

2D 厚スライスのシミュレーション: --slab N で連続 N 枚の薄スライスを矩形プロファイルで
合成し厚 2D スライス化する（3D薄スライス積層→擬似2D。mri_slice_sim.py の方針に準拠）。

対応形式
--------
- マルチコイル(brain/knee): kspace (slices, coils, H, W)。コイル RSS。
- 単コイル(knee): kspace (slices, H, W)。|IFFT|。reconSpace へ中央クロップ、
  アンダーサンプリング(mask)はゼロ詰め再構成。
- 真の3D取得: kspace (partitions, coils, H, W)。パーティション(kz)方向も IFFT する
  3D再構成。ヘッダ(encodedSpace.z>1 / kspace_encoding_step_2>0)で自動判定（--recon-3d）。
  既定レイアウトは (kz,coil,ky,kx)＝軸0がパーティション（--part-axis で変更可）。
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


# ----------------------------- 向き（patientPosition 由来） -----------------------------
def resolve_orient(meta: dict, flip_x="auto", flip_y="auto", reverse="auto"):
    """出力の向き(flip_x, flip_y, reverse_slices)を決める。

    簡易 fastMRI 形式にはスライス毎の ImageOrientationPatient/ImagePositionPatient が
    無く、参照できる向き情報は `patientPosition`(HFS/FFS 等)のみ。これでスライス積層方向と
    左右を一貫させる（HFS と FFS で S-I / L-R が反転するため）。auto 時の既定:
      - patientPosition あり: flip_y=True、reverse_slices=Head-First?True:False、flip_x=False
      - patientPosition 無し（ヘッダ無し。Calgary-Campinas 等）: いずれも False
        （向き情報が無いので素の配列順を尊重し、勝手な反転をしない）
    on/off で明示上書きできる。
    """
    pos = meta.get("patient_position")
    if pos:
        head_first = pos.upper().startswith("HF")
        auto = {"flip_x": False, "flip_y": True, "reverse_slices": head_first}
    else:
        auto = {"flip_x": False, "flip_y": False, "reverse_slices": False}

    def pick(opt, key):
        return auto[key] if opt == "auto" else (opt == "on")

    return (pick(flip_x, "flip_x"), pick(flip_y, "flip_y"), pick(reverse, "reverse_slices"))


def apply_orient(vol: np.ndarray, flip_x: bool, flip_y: bool, reverse_slices: bool):
    """(z,y,x) ボリュームに向き補正を適用（全出力形式で共通に使う）。"""
    if reverse_slices:
        vol = vol[::-1, :, :]
    if flip_y:
        vol = vol[:, ::-1, :]
    if flip_x:
        vol = vol[:, :, ::-1]
    return np.ascontiguousarray(vol)


def combine_slab(vol: np.ndarray, n: int, step: int | None = None) -> np.ndarray:
    """N 枚の薄スライスを矩形(rect)プロファイルで合成し、厚い 2D スライスを作る。

    3D ボリューム（薄スライス積層）から擬似 2D 厚スライスをシミュレートする。
    各出力スライス = 連続 N 枚の等重み平均（rect スライスプロファイル、総和1で正規化）。
    本プロジェクト mri_slice_sim.py の方針（magnitude・同コントラスト・矩形プロファイル）に準拠。
    step を指定すると重なりスラブも可（既定 step=N で非重複）。
    """
    if n <= 1:
        return vol
    step = step or n
    nz = vol.shape[0]
    out = [vol[s:s + n].mean(axis=0) for s in range(0, nz - n + 1, step)]
    if not out:                                   # N がスライス数より大きい場合は全平均1枚
        out = [vol.mean(axis=0)]
    return np.ascontiguousarray(np.stack(out, axis=0))


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


def ifftc_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """指定軸に沿った中心化 1D 逆FFT（norm="ortho"）。"""
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm="ortho"), axes=axis)


def deinterleave_complex(arr: np.ndarray, axis: int) -> np.ndarray:
    """指定軸の実/虚インターリーブ（[r0,i0,r1,i1,...]）を複素に変換。
    その軸サイズ 2N → N（複素）。Calgary-Campinas 等の float 実数格納に使う。
    """
    axis = axis % arr.ndim
    if arr.shape[axis] % 2 != 0:
        raise ValueError(f"real/imag axis {axis} size {arr.shape[axis]} must be even")
    sr = [slice(None)] * arr.ndim
    si = [slice(None)] * arr.ndim
    sr[axis] = slice(0, None, 2)
    si[axis] = slice(1, None, 2)
    return (arr[tuple(sr)].astype(np.float32)
            + 1j * arr[tuple(si)].astype(np.float32)).astype(np.complex64)


def detect_layout(shape, enc_x, enc_y, enc_z):
    """4D k空間の軸配置を、ヘッダの encodedSpace(x,y,z) と各軸サイズの一致で推定し、
    標準順 (kz, coil, ky, kx) への並べ替え permutation を返す。判定不能なら None。

    kx=enc.x(readout), ky=enc.y(phase), kz=enc.z(partition), 残り=coil。
    in-plane(ky,kx)が最後の2軸でない/パーティションが軸0でない場合の自動補正に使う。
    """
    if len(shape) != 4:
        return None
    ex = int(enc_x) if enc_x else None
    ey = int(enc_y) if enc_y else None
    ez = int(enc_z) if enc_z else None
    if not ex or not ey or not ez or ez <= 1:
        return None
    axes, used = list(range(4)), []

    def match(target):
        for a in axes:
            if a not in used and abs(shape[a] - target) <= 1:
                used.append(a)
                return a
        return None

    kx = match(ex)
    ky = match(ey)
    kz = match(ez)
    rest = [a for a in axes if a not in used]
    if kx is None or ky is None or kz is None or len(rest) != 1:
        return None
    return (kz, rest[0], ky, kx)         # → (kz, coil, ky, kx)


def reconstruct(kspace: np.ndarray, recon_size=None, acq_matrix=None,
                snr: float | None = None, rng=None,
                partition_3d: bool = False, recon_z=None,
                part_axis: int = 0, centered_kspace: bool = True) -> np.ndarray:
    """k空間 → magnitude 画像。マルチコイル(...,C,H,W)はRSS、単コイル(...,H,W)は|.|。

    既定は面内(最後の2軸)のみ IFFT（2Dマルチスライス）。
    真の 3D 取得（partition_3d=True）では、パーティション軸 part_axis（既定 0）にも
    1D 逆FFT を掛ける＝スライス方向も k空間エンコードを復元する。recon_z 指定時は
    パーティション軸をその枚数へ中央クロップ（スライスオーバーサンプリング除去）。

    centered_kspace: True=DCが中心（fastMRI規約。ifftshift/fftshift 付き）。
      False=DCが端[0,0]（標準FFT配置、Calgary-Campinas 等）→ シフト無しの素の ifft。
      端DCに中心化IFFTを使うと像が半周ずれ「四隅に分裂」するため要切替。

    低磁場シミュレーション（任意・k空間ドメイン、出力画素数は不変）:
    - acq_matrix=(tr,tc): 取得マトリクスを中央クロップで落とす（中心DC前提）
    - snr: 複素ガウシアンノイズを付加して目標SNRへ（RSS後 Rician/noncentral-chi）
    recon_size=(rows,cols): IFFT後に中央クロップ（fastMRI reconSpace。Noneで無し）
    """
    ks = kspace
    if acq_matrix is not None:
        ks = kspace_crop_zerofill(ks, acq_matrix)

    if centered_kspace:
        imgs = ifft2c(ks)                          # 面内（中心DC）
        if partition_3d:
            imgs = ifftc_axis(imgs, part_axis)     # スライス(kz)方向も復元 → 3D IFFT
    else:                                          # 端DC: シフト無しの素の ifft
        imgs = np.fft.ifft2(ks, axes=(-2, -1), norm="ortho")
        if partition_3d:
            imgs = np.fft.ifft(imgs, axis=part_axis, norm="ortho")
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

    mag = to_mag(imgs)                             # (partitions/slices, H, W)
    mag = center_crop(mag, recon_size)             # 面内クロップ
    if partition_3d and recon_z and 0 < recon_z < mag.shape[0]:
        z0 = (mag.shape[0] - recon_z) // 2         # パーティション中央クロップ
        mag = mag[z0:z0 + recon_z]
    return mag


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
        "patient_position": None, "field_strength": None,
        "tr": None, "te": None, "ti": None, "flip": None,
        "vendor": None, "model": None, "institution": None,
        "protocol": None, "sequence": None,
        "series_uid": None, "study_uid": None, "for_uid": None,
        "study_time": None,
        "enc_x": None, "enc_y": None, "enc_z": None,
        "recon_z": None, "is_3d": False,
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

    # 3D 取得の判定（パーティション方向 kz もエンコードされているか）
    enc_sp = _find(enc, "encodedSpace")
    enc_mat = _find(enc_sp, "matrixSize")
    rec_mat = mat
    ex, ey = (_ftext(enc_mat, "x"), _ftext(enc_mat, "y")) if enc_mat is not None else (None, None)
    ez = _ftext(enc_mat, "z") if enc_mat is not None else None
    rz = _ftext(rec_mat, "z") if rec_mat is not None else None
    meta["enc_x"] = int(ex) if ex else None
    meta["enc_y"] = int(ey) if ey else None
    meta["enc_z"] = int(ez) if ez else None
    meta["recon_z"] = int(rz) if rz else None
    # kspace_encoding_step_2（パーティション位相エンコード）の最大値も併用
    step2 = _find(enc, "kspace_encoding_step_2")
    s2max = _ftext(step2, "maximum") if step2 is not None else None
    s2max = int(s2max) if s2max else 0
    meta["is_3d"] = bool((meta["enc_z"] or 1) > 1 or s2max > 0)

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
               basename: str, patient_id: str, acquisition: str, out_dir: str) -> None:
    """再構成 magnitude を MR Image Storage DICOM 群として書き出す。

    vol は呼び出し側で向き補正済み（apply_orient）であることを前提とする。
    信号スケール: 格納画素 = round(magnitude / rescale_slope)（控えめな整数）。
    RescaleSlope を設定するので pixel_array * RescaleSlope = 元 magnitude。
    """
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
                orient_note: str = "") -> None:
    """3D ボリュームを 2byte の .raw として保存し、.hdr / .tag を併記する。

    - <out_base>.raw : signed int16 リトルエンディアン。並びは x が最速→ y → z（C順, shape (z,y,x)）
    - <out_base>.hdr : 'xサイズ yサイズ zサイズ 2 x_spacing y_spacing z_spacing'
    - <out_base>.tag : 撮像/再構成の各種メタ情報（key: value）

    vol は呼び出し側で向き補正済み（apply_orient）であることを前提とする。
    信号スケール: 格納値 = round(magnitude / rescale_slope)。過大スケールにせず、
    `.tag` の rescale_slope を使って magnitude = 格納値 * rescale_slope で復元する。
    2byte を符号付き short(int16) として読むビューア向けに int16範囲(±32767)に収める。
    """
    nz, ny, nx = vol.shape
    dx = meta["dx"] or 1.0                       # 列(x)方向の物理スペーシング mm
    dy = meta["dy"] or 1.0                       # 行(y)方向の物理スペーシング mm
    dz = meta["spacing"] or meta["thickness"] or 1.0   # スライス(z)方向 mm

    inv = 1.0 / (rescale_slope + 1e-30)          # 格納値 = magnitude / slope
    stored = np.clip(np.round(vol * inv), 0, STORE_MAX).astype("<i2")  # signed int16 (z,y,x)

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
        "voxel_order: x_fastest_then_y_then_z",
        f"orientation: {orient_note}",
        f"patient_position: {meta.get('patient_position') or ''}",
        f"intensity_max: {data_max:g}",
        f"stored_max: {int(np.max(stored))}",
        f"rescale_slope: {slope:g}",
        "rescale_note: magnitude = stored_value * rescale_slope",
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
                 acq_matrix=None, snr=None, rng=None,
                 flip_x="auto", flip_y="auto", reverse="auto",
                 slab=1, slab_step=None, recon_3d="auto", part_axis=0,
                 transpose=None, real_imag_axis=None, kspace_dc="center",
                 pixel_spacing=None, slice_spacing=None, slice_thickness=None) -> int:
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

    # ジオメトリの明示上書き（ヘッダ無しデータ＝Calgary-Campinas 等の voxel 間隔指定）
    if pixel_spacing is not None:
        meta["dy"], meta["dx"] = pixel_spacing            # [row, col] mm
    if slice_spacing is not None:
        meta["spacing"] = slice_spacing
    if slice_thickness is not None:
        meta["thickness"] = slice_thickness

    # 3D取得か（auto=ヘッダの encodedSpace.z / kspace_encoding_step_2 から判定）
    is3d = meta["is_3d"] if recon_3d == "auto" else (recon_3d == "on")

    # 実/虚インターリーブ（実数格納）→ 複素化（Calgary-Campinas 等）。transpose より先。
    if real_imag_axis is not None:
        ks = deinterleave_complex(ks, real_imag_axis)

    # k空間の軸配置補正: --transpose 明示 > 3D時の自動推定（ヘッダのマトリクスサイズ一致）
    layout_note = ""
    if transpose is not None:
        ks = np.ascontiguousarray(np.transpose(ks, transpose))
        layout_note = f"transpose={transpose}"
    elif is3d:
        perm = detect_layout(ks.shape, meta.get("enc_x"), meta.get("enc_y"),
                             meta.get("enc_z"))
        if perm and perm != tuple(range(ks.ndim)):
            ks = np.ascontiguousarray(np.transpose(ks, perm))
            layout_note = f"auto-layout {tuple(ks.shape)} (perm={perm})"

    rss = reconstruct(ks, recon_size=recon_size, acq_matrix=acq_matrix,
                      snr=snr, rng=rng, partition_3d=is3d,
                      recon_z=meta.get("recon_z"), part_axis=part_axis,
                      centered_kspace=(kspace_dc == "center"))

    # 向き補正（patientPosition 由来。全形式に共通適用して一貫させる）
    fx, fy, rev = resolve_orient(meta, flip_x, flip_y, reverse)
    rss = apply_orient(rss, fx, fy, rev)
    orient_note = (f"patient_position={meta.get('patient_position') or '?'}; "
                   f"flip_x={fx} flip_y={fy} reverse_slices={rev}")

    # 2D 厚スライスのシミュレーション（薄スライスを矩形プロファイルで合成）
    if slab and slab > 1:
        n0 = rss.shape[0]
        rss = combine_slab(rss, slab, slab_step)
        step = slab_step or slab
        if meta.get("spacing"):
            meta = dict(meta)
            meta["thickness"] = float(meta["thickness"]) * slab     # 厚み≈N×元厚
            meta["spacing"] = float(meta["spacing"]) * step         # 間隔≈step×元間隔
        lf_slab = f" slab={slab}({n0}->{rss.shape[0]}枚)"
    else:
        lf_slab = ""

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
                       pid, acq, out_dir)

    # binary は 1 ボリューム = <out_root>/<base>.raw/.hdr/.tag
    if "binary" in fmts:
        save_binary(rss, rescale_slope, data_max, meta, os.path.basename(base),
                    pid, acq, os.path.join(out_root, base), orient_note=orient_note)

    print(f"[ok] {rel}  ({acq}, {meta.get('patient_position') or '?'}"
          f"{', 3D' if is3d else ''})  "
          f"{rss.shape[0]} slices {rss.shape[1]}x{rss.shape[2]}  "
          f"[{'+'.join(sorted(fmts))}]{lf}{lf_slab}  rev_slices={rev}"
          f"{('  ' + layout_note) if layout_note else ''}")

    nz, ny, nx = rss.shape
    return {
        "file": rel,
        "basename": os.path.basename(base),
        "acquisition": acq,
        "patient_id": pid,
        "patient_position": meta.get("patient_position") or "",
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
        "is_3d": is3d,
        "flip_x": fx, "flip_y": fy, "reverse_slices": rev,
        "slab": slab if slab and slab > 1 else "",
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
    # 向き（patientPosition 由来で自動。raw/DICOM/PNG 共通。auto/on/off で上書き）
    ap.add_argument("--flip-y", choices=["auto", "on", "off"], default="auto",
                    help="行(上下)反転。auto=ON（撮像面に依らず上下を合わせる）")
    ap.add_argument("--flip-x", choices=["auto", "on", "off"], default="auto",
                    help="列(左右)反転。auto=OFF")
    ap.add_argument("--reverse-slices", choices=["auto", "on", "off"], default="auto",
                    help="スライス積層方向の反転。auto=patientPosition由来"
                         "（Head-First:ON / Feet-First:OFF）")
    # 2D 厚スライスのシミュレーション（3D薄スライス積層 → 厚2D）
    ap.add_argument("--slab", type=int, default=1,
                    help="N枚の薄スライスを矩形プロファイルで合成し厚2Dスライス化（既定1=無し）")
    ap.add_argument("--slab-step", type=int, default=None,
                    help="スラブの送り（既定=--slab で非重複）")
    # 真の3D取得の再構成（パーティション方向 kz にも IFFT）
    ap.add_argument("--recon-3d", choices=["auto", "on", "off"], default="auto",
                    help="3D取得の再構成。auto=ヘッダ(encodedSpace.z/step_2)で判定、"
                         "on=強制3D(スライス方向もIFFT)、off=面内のみ(2Dマルチスライス)")
    ap.add_argument("--part-axis", type=int, default=0,
                    help="3D時のパーティション(kz)軸（既定0。kspace=(kz,coil,ky,kx)前提）")
    ap.add_argument("--transpose", default=None,
                    help="k空間の軸を明示的に並べ替える permutation（例 '2,0,3,1'）。"
                         "標準順 (kz/slice, coil, ky, kx) へ。inspect_h5.py で軸を確認")
    ap.add_argument("--real-imag-axis", type=int, default=None,
                    help="実/虚インターリーブ軸（実数格納の複素データ）。その軸 2N→N に複素化。"
                         "Calgary-Campinas は最後の軸（例 -1）。--transpose より先に適用")
    ap.add_argument("--kspace-dc", choices=["center", "corner"], default="center",
                    help="k空間のDC位置。center=中心(fastMRI規約,既定)、"
                         "corner=端[0,0](標準FFT配置/Calgary-Campinas)。"
                         "corner を center で再構成すると像が四隅に分裂する")
    # ジオメトリの上書き（ヘッダ無しデータ用。例 Calgary-Campinas は 1mm 等方）
    ap.add_argument("--pixel-spacing", default=None,
                    help="面内画素間隔[mm]を上書き。'v'(等方) か 'row,col'。例 1 / 0.9,0.9")
    ap.add_argument("--slice-spacing", type=float, default=None,
                    help="スライス間隔(SpacingBetweenSlices)[mm]を上書き。例 1")
    ap.add_argument("--slice-thickness", type=float, default=None,
                    help="スライス厚(SliceThickness)[mm]を上書き。例 1")
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
    transpose = tuple(int(v) for v in args.transpose.split(",")) if args.transpose else None
    pixel_spacing = None
    if args.pixel_spacing:
        pv = [float(v) for v in str(args.pixel_spacing).split(",")]
        pixel_spacing = (pv[0], pv[0]) if len(pv) == 1 else (pv[0], pv[1])   # (row, col)

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
                               flip_x=args.flip_x, flip_y=args.flip_y,
                               reverse=args.reverse_slices,
                               slab=args.slab, slab_step=args.slab_step,
                               recon_3d=args.recon_3d, part_axis=args.part_axis,
                               transpose=transpose, real_imag_axis=args.real_imag_axis,
                               kspace_dc=args.kspace_dc, pixel_spacing=pixel_spacing,
                               slice_spacing=args.slice_spacing,
                               slice_thickness=args.slice_thickness)
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
