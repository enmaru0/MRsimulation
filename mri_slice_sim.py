#!/usr/bin/env python3
"""
mri_slice_sim.py
================
3DのMRI(または任意)DICOMシリーズから、擬似的な2D厚スライスDICOMを合成する。

物理モデル
---------
厚いスライスのMR信号は、連続体の横磁化 M(z) をスライス感度プロファイル P(z) で
重み付き積分したものに等しい:

    S(c) = ∫ P(z - c) · M(z) dz

本スクリプトが対象とするケース（magnitudeのみ / 出力コントラストは入力と同一 /
矩形プロファイル）では M(z) ≈ 入力画素値 とみなせるため、スライス方向の
プロファイル重み付き加算に帰着する。CTのガウシアン平滑化との違いは

  1) カーネルがガウシアンではなく矩形(FWHM = スライス厚)であること
  2) 強度スケールを保つため重みを総和1に正規化すること

の2点。複素データや別コントラストが必要な場合は、複素加算 / 信号方程式での
再計算が必要になる（本スクリプトの範囲外。--profile を差し替える等で拡張可能）。

注意
----
- magnitude かつ 同一コントラスト のときに物理的に妥当。
- 位相情報が無いため部分容積での位相打ち消しは再現されない（その分、境界の
  信号低下を過小評価しうる）。
- 面内解像度は変更しない（through-plane のみ）。面内を変える場合はk空間
  トランケーション(Gibbsリンギング)が別途必要。
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates


# --------------------------------------------------------------------------- #
# スライスプロファイル
# --------------------------------------------------------------------------- #
def rect_profile(fwhm: float) -> Callable[[np.ndarray], np.ndarray]:
    """矩形プロファイル。windowed-sinc RFパルスの第一近似。"""
    half = fwhm / 2.0
    return lambda z: (np.abs(z) <= half).astype(float)


def gaussian_profile(fwhm: float) -> Callable[[np.ndarray], np.ndarray]:
    """ガウシアンプロファイル（CT相当の比較用）。"""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return lambda z: np.exp(-0.5 * (z / sigma) ** 2)


def trapezoid_profile(fwhm: float, ramp: float) -> Callable[[np.ndarray], np.ndarray]:
    """台形プロファイル。端の鈍り(ramp mm)を持つより現実的な近似。"""
    half = fwhm / 2.0

    def f(z: np.ndarray) -> np.ndarray:
        a = np.abs(z)
        out = np.ones_like(a, dtype=float)
        if ramp > 0:
            edge = (a > half - ramp) & (a < half)
            out[edge] = (half - a[edge]) / ramp
        out[a >= half] = 0.0
        return out

    return f


PROFILES = {
    "rect": rect_profile,
    "gaussian": gaussian_profile,
}


def measured_profile(ssp_file: str):
    """calibrate.py が保存した実測SSP(offset,weight)を補間する callable を返す。"""
    arr = np.load(ssp_file)          # shape (2, M): [offsets; weights]
    off, w = arr[0], arr[1]
    span = float(off.max() - off.min())
    f = lambda z: np.interp(z, off, w, left=0.0, right=0.0)
    return f, span


# --------------------------------------------------------------------------- #
# シリーズ読み込みとジオメトリ
# --------------------------------------------------------------------------- #
@dataclass
class Series:
    datasets: list            # スライス順にソート済みの pydicom.Dataset
    positions: np.ndarray     # スライス法線方向の位置 s_i [mm] (昇順)
    normal: np.ndarray        # スライス法線の単位ベクトル (LPS)
    volume: np.ndarray        # (n_slices, rows, cols) float, real-world値(HU/信号)
    template: pydicom.Dataset


def load_series(folder: str, pattern: str = "*") -> Series:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue
        if "PixelData" not in ds or "ImagePositionPatient" not in ds:
            continue
        datasets.append(ds)
    if not datasets:
        raise RuntimeError(f"有効なDICOM画像が見つかりません: {folder}")

    iop = np.array(datasets[0].ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[0:3], iop[3:6])
    normal /= np.linalg.norm(normal)

    s = np.array([np.dot(np.array(ds.ImagePositionPatient, float), normal)
                  for ds in datasets])
    order = np.argsort(s)
    datasets = [datasets[i] for i in order]
    s = s[order]

    # real-world値に変換して積層（線形領域で加算するため）
    vol = []
    for ds in datasets:
        px = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        vol.append(px * slope + intercept)
    volume = np.stack(vol, axis=0)

    return Series(datasets, s, normal, volume, datasets[0])


# --------------------------------------------------------------------------- #
# 重み行列の構築
# --------------------------------------------------------------------------- #
def voronoi_edges(positions: np.ndarray) -> np.ndarray:
    """各サンプルの受け持ち区間境界(隣接中点)。長さ n+1。"""
    mids = (positions[:-1] + positions[1:]) / 2.0
    first = positions[0] - (mids[0] - positions[0])
    last = positions[-1] + (positions[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def build_weights(positions: np.ndarray,
                  centers: np.ndarray,
                  profile: Callable[[np.ndarray], np.ndarray],
                  support: float,
                  oversample: int = 64) -> np.ndarray:
    """
    重み行列 W (n_out, n_in) を返す。
    W[o, i] = ∫_{区間 i} P(z - centers[o]) dz  (各出力で総和1に正規化)

    矩形プロファイルでは区間と窓の重なり長に一致する。任意プロファイルにも
    対応するため、各入力区間を oversample 分割して数値積分する。
    """
    edges = voronoi_edges(positions)           # (n_in+1,)
    a = edges[:-1]                              # 区間下端 (n_in,)
    b = edges[1:]                               # 区間上端 (n_in,)
    n_in = positions.size
    n_out = centers.size
    W = np.zeros((n_out, n_in))

    for o, c in enumerate(centers):
        lo, hi = c - support / 2.0, c + support / 2.0
        # サポート外の区間はスキップ
        active = np.where((b > lo) & (a < hi))[0]
        for i in active:
            seg_lo = max(a[i], lo)
            seg_hi = min(b[i], hi)
            if seg_hi <= seg_lo:
                continue
            zs = np.linspace(seg_lo, seg_hi, oversample)
            w = np.trapezoid(profile(zs - c), zs)
            W[o, i] = w
        ssum = W[o].sum()
        if ssum > 0:
            W[o] /= ssum
    return W


def output_centers(positions: np.ndarray,
                   spacing: float,
                   thickness: float,
                   start: float | None) -> np.ndarray:
    """出力スライス中心を spacing 間隔で生成。デフォルトは入力範囲を覆う。"""
    z0 = positions[0] if start is None else start
    z1 = positions[-1]
    n = int(np.floor((z1 - z0) / spacing)) + 1
    return z0 + spacing * np.arange(max(n, 1))


# --------------------------------------------------------------------------- #
# 任意面へのリスライス (AX / COR / SAG)
# --------------------------------------------------------------------------- #
# 患者座標系(LPS): +X=左, +Y=後, +Z=頭。各面の (列方向u, 行方向v, 法線n)。
# u,v,n は正規直交基底。n は「スライス番号が増える向き」。
ORIENTATIONS = {
    "axial":    (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
    "coronal":  (np.array([1., 0., 0.]), np.array([0., 0., -1.]), np.array([0., 1., 0.])),
    "sagittal": (np.array([0., 1., 0.]), np.array([0., 0., -1.]), np.array([1., 0., 0.])),
}


def input_affine(series: Series) -> np.ndarray:
    """入力ボクセル添字 [col, row, slice, 1] -> 世界座標(LPS) のアフィン(4x4)。"""
    tmpl = series.template
    iop = np.array(tmpl.ImageOrientationPatient, float)
    x_dir, y_dir = iop[0:3], iop[3:6]              # 列方向, 行方向
    ps = [float(v) for v in tmpl.PixelSpacing]     # [row spacing, col spacing]
    ps_row, ps_col = ps[0], ps[1]
    dz = float(np.median(np.diff(series.positions)))
    ipp0 = np.array(series.datasets[0].ImagePositionPatient, float)

    A = np.eye(4)
    A[:3, 0] = x_dir * ps_col      # col 添字
    A[:3, 1] = y_dir * ps_row      # row 添字
    A[:3, 2] = series.normal * dz  # slice 添字
    A[:3, 3] = ipp0
    return A


def reslice_volume(series: Series,
                   u: np.ndarray, v: np.ndarray, n: np.ndarray,
                   in_plane_spacing: float,
                   recon_step: float):
    """
    入力3Dボリュームを、出力面 (u,v,n) の正則格子へトリリニア再標本化する。
    返り値: (V_fine[K,R,C], d_positions[K], a_min, b_min)
      - K は法線方向の細かい刻み(recon_step)で刻んだ枚数
      - 後段で build_weights によりプロファイル積分→出力スライスに集約する
    """
    A = input_affine(series)
    Ainv = np.linalg.inv(A)
    K_in, R_in, C_in = series.volume.shape

    # 入力ボリューム8隅を世界座標へ→出力軸へ射影して範囲を決める
    corners = []
    for ck in (0, K_in - 1):
        for cr in (0, R_in - 1):
            for cc in (0, C_in - 1):
                corners.append(A @ np.array([cc, cr, ck, 1.0]))
    corners = np.array(corners)[:, :3]
    a = corners @ u; b = corners @ v; d = corners @ n
    a_min, a_max = a.min(), a.max()
    b_min, b_max = b.min(), b.max()
    d_min, d_max = d.min(), d.max()

    C_out = int(np.ceil((a_max - a_min) / in_plane_spacing)) + 1
    R_out = int(np.ceil((b_max - b_min) / in_plane_spacing)) + 1
    K_out = int(np.ceil((d_max - d_min) / recon_step)) + 1

    cc = a_min + np.arange(C_out) * in_plane_spacing      # (C,)
    rr = b_min + np.arange(R_out) * in_plane_spacing      # (R,)
    d_positions = d_min + np.arange(K_out) * recon_step   # (K,)

    # 面内の世界座標(法線成分を除く)を事前計算: (R, C, 3)
    base = cc[None, :, None] * u[None, None, :] + rr[:, None, None] * v[None, None, :]
    V_fine = np.empty((K_out, R_out, C_out), dtype=np.float32)
    vol = series.volume  # (K_in, R_in, C_in) = (slice, row, col)

    for ki, dval in enumerate(d_positions):
        world = base + dval * n[None, None, :]            # (R, C, 3)
        homog = np.concatenate([world, np.ones((R_out, C_out, 1))], axis=-1)
        vox = homog @ Ainv.T                              # (R, C, 4) -> [col,row,slice]
        coords = np.stack([vox[..., 2], vox[..., 1], vox[..., 0]], axis=0)  # (slice,row,col)
        V_fine[ki] = map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)

    return V_fine, d_positions, a_min, b_min


# --------------------------------------------------------------------------- #
# DICOM書き出し
# --------------------------------------------------------------------------- #
def write_output(series: Series,
                 centers: np.ndarray,
                 out_volume: np.ndarray,
                 thickness: float,
                 spacing: float,
                 out_dir: str,
                 series_desc: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    new_series_uid = generate_uid()
    tmpl = series.template
    iop = np.array(tmpl.ImageOrientationPatient, float)
    normal = series.normal

    # 出力IPPは、入力先頭スライスのIPPを基準に法線方向へ centers ぶん移動
    base_ipp = np.array(series.datasets[0].ImagePositionPatient, float)
    base_s = series.positions[0]

    slope = float(getattr(tmpl, "RescaleSlope", 1) or 1)
    intercept = float(getattr(tmpl, "RescaleIntercept", 0) or 0)

    for k, c in enumerate(centers):
        ds = pydicom.dcmread(series.datasets[0].filename)  # ヘッダ雛形を複製
        ipp = base_ipp + normal * (c - base_s)

        # real-world値 → 格納画素値に戻し、dtypeにクリップ
        px = (out_volume[k] - intercept) / slope
        bits = int(getattr(ds, "BitsStored", 16))
        signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
        if signed:
            lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
            dtype = np.int16
        else:
            lo, hi = 0, 2 ** bits - 1
            dtype = np.uint16
        px = np.clip(np.round(px), lo, hi).astype(dtype)

        ds.PixelData = px.tobytes()
        ds.Rows, ds.Columns = px.shape
        ds.SliceThickness = float(thickness)
        ds.SpacingBetweenSlices = float(spacing)
        ds.ImagePositionPatient = [float(x) for x in ipp]
        ds.SliceLocation = float(c)
        ds.InstanceNumber = k + 1
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid()
        if hasattr(ds, "file_meta"):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = series_desc
        # 派生画像であることを明示
        it = list(getattr(ds, "ImageType", ["DERIVED", "SECONDARY"]))
        if it and it[0] == "ORIGINAL":
            it[0] = "DERIVED"
        ds.ImageType = it

        ds.save_as(os.path.join(out_dir, f"{k + 1:05d}.DCM"))


def _encode_px(ds: pydicom.Dataset, real: np.ndarray,
               slope: float, intercept: float) -> np.ndarray:
    """real-world値 -> 格納画素値に戻し、dtype範囲にクリップ。"""
    px = (real - intercept) / slope
    bits = int(getattr(ds, "BitsStored", 16))
    signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
    if signed:
        lo, hi, dtype = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1, np.int16
    else:
        lo, hi, dtype = 0, 2 ** bits - 1, np.uint16
    return np.clip(np.round(px), lo, hi).astype(dtype)


def write_resliced(series: Series,
                   u: np.ndarray, v: np.ndarray, n: np.ndarray,
                   a_min: float, b_min: float,
                   in_plane_spacing: float,
                   centers: np.ndarray,
                   out_volume: np.ndarray,
                   thickness: float, spacing: float,
                   out_dir: str, series_desc: str) -> None:
    """AX/COR/SAG リスライス結果を、出力面のジオメトリでDICOM書き出し。"""
    os.makedirs(out_dir, exist_ok=True)
    new_series_uid = generate_uid()
    tmpl = series.template
    slope = float(getattr(tmpl, "RescaleSlope", 1) or 1)
    intercept = float(getattr(tmpl, "RescaleIntercept", 0) or 0)
    iop = [float(x) for x in (*u, *v)]

    for k, c in enumerate(centers):
        ds = pydicom.dcmread(series.datasets[0].filename)
        # 出力スライスの (row0,col0) 世界座標 = a_min*u + b_min*v + c*n
        ipp = a_min * u + b_min * v + c * n
        px = _encode_px(ds, out_volume[k], slope, intercept)

        ds.PixelData = px.tobytes()
        ds.Rows, ds.Columns = px.shape
        ds.PixelSpacing = [float(in_plane_spacing), float(in_plane_spacing)]
        ds.ImageOrientationPatient = iop
        ds.ImagePositionPatient = [float(x) for x in ipp]
        ds.SliceThickness = float(thickness)
        ds.SpacingBetweenSlices = float(spacing)
        ds.SliceLocation = float(c)
        ds.InstanceNumber = k + 1
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid()
        if hasattr(ds, "file_meta"):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = series_desc
        it = list(getattr(ds, "ImageType", ["DERIVED", "SECONDARY"]))
        if it and it[0] == "ORIGINAL":
            it[0] = "DERIVED"
        ds.ImageType = it
        ds.save_as(os.path.join(out_dir, f"{k + 1:05d}.DCM"))


# --------------------------------------------------------------------------- #
def simulate(in_dir: str,
             out_dir: str,
             thickness: float,
             spacing: float,
             profile_name: str,
             support: float | None,
             start: float | None,
             pattern: str,
             series_desc: str,
             orientation: str = "native",
             in_plane_spacing: float | None = None,
             recon_step: float | None = None,
             ssp_file: str | None = None) -> None:
    series = load_series(in_dir, pattern)
    dz_in = float(np.median(np.diff(series.positions)))
    ps_in = [float(x) for x in series.template.PixelSpacing]
    print(f"[load] {len(series.datasets)} slices, in-plane "
          f"{series.template.Rows}x{series.template.Columns}, "
          f"slice spacing ~{dz_in:.3f} mm, normal={np.round(series.normal,3)}")

    if ssp_file:
        # calibrate.py の実測SSPを使用（thickness/profile指定より優先）
        profile, span = measured_profile(ssp_file)
        profile_name = f"measured({os.path.basename(ssp_file)})"
        if support is None:
            support = span
    else:
        profile = PROFILES[profile_name](thickness)
        if support is None:
            # 矩形は thickness、裾を持つプロファイルは余裕を見て広めに取る
            support = thickness if profile_name == "rect" else thickness * 3.0

    if orientation == "native":
        # 入力スライス方向に沿って集約（リスライスなし）
        centers = output_centers(series.positions, spacing, thickness, start)
        W = build_weights(series.positions, centers, profile, support)
        nz, ny, nx = series.volume.shape
        out = (W @ series.volume.reshape(nz, ny * nx)).reshape(-1, ny, nx)
        print(f"[sim ] native profile={profile_name} FWHM={thickness}mm "
              f"support={support}mm -> {centers.size} slices @ {spacing}mm spacing")
        write_output(series, centers, out, thickness, spacing, out_dir, series_desc)
    else:
        # AX/COR/SAG: 患者座標系でリスライスしてからプロファイル積分
        u, v, n = ORIENTATIONS[orientation]
        ips = in_plane_spacing or min(ps_in)
        step = recon_step or min(*ps_in, dz_in)
        V_fine, d_pos, a_min, b_min = reslice_volume(series, u, v, n, ips, step)
        centers = output_centers(d_pos, spacing, thickness, start)
        W = build_weights(d_pos, centers, profile, support)
        K, R, C = V_fine.shape
        out = (W @ V_fine.reshape(K, R * C)).reshape(-1, R, C)
        print(f"[sim ] {orientation} profile={profile_name} FWHM={thickness}mm "
              f"support={support}mm in-plane={ips:.3f}mm recon-step={step:.3f}mm "
              f"-> {centers.size} slices {R}x{C} @ {spacing}mm spacing")
        write_resliced(series, u, v, n, a_min, b_min, ips,
                       centers, out, thickness, spacing, out_dir, series_desc)
    print(f"[save] {centers.size} files -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="入力DICOMシリーズのフォルダ")
    ap.add_argument("output", help="出力フォルダ")
    ap.add_argument("--thickness", type=float, default=5.0, help="出力スライス厚 [mm] (default 5)")
    ap.add_argument("--spacing", type=float, default=6.0, help="出力スライス間隔(中心間) [mm] (default 6)")
    ap.add_argument("--profile", choices=list(PROFILES), default="rect",
                    help="スライス感度プロファイル (default rect)")
    ap.add_argument("--support", type=float, default=None,
                    help="プロファイルの積分範囲 [mm] (default: rect=thickness, それ以外=3*thickness)")
    ap.add_argument("--start", type=float, default=None,
                    help="先頭出力スライス中心の法線方向位置 [mm] (default: 入力先頭)")
    ap.add_argument("--pattern", default="*", help="入力ファイルのglobパターン (default *)")
    ap.add_argument("--desc", default="Simulated 2D thick-slice", help="SeriesDescription")
    ap.add_argument("--orientation", choices=["native", "axial", "coronal", "sagittal"],
                    default="native",
                    help="出力面 (default native=入力スライス方向)。axial/coronal/sagittal は"
                         "患者座標系でリスライスして生成")
    ap.add_argument("--in-plane-spacing", type=float, default=None,
                    help="リスライス時の面内画素間隔 [mm] (default: 入力の最小画素間隔)")
    ap.add_argument("--recon-step", type=float, default=None,
                    help="リスライス時の法線方向の細刻み [mm] (default: 入力の最小間隔)")
    ap.add_argument("--ssp-file", default=None,
                    help="calibrate.py が出力した実測SSP(.npy)。指定時は --profile/--thickness より優先")
    args = ap.parse_args()

    simulate(args.input, args.output, args.thickness, args.spacing,
             args.profile, args.support, args.start, args.pattern, args.desc,
             args.orientation, args.in_plane_spacing, args.recon_step, args.ssp_file)


if __name__ == "__main__":
    main()
