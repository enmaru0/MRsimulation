#!/usr/bin/env python3
"""
lowfield_sim.py
===============
高磁場(1.5T/3T)の高画質MRI DICOMから、物理的に整合した低磁場(0.3-0.5T)風の
劣化画像を合成する。教師あり学習の (入力=低磁場風 / 正解=高磁場) ペア生成用。

モデル化する劣化（magnitude画像から物理的に妥当に作れるもの）
--------------------------------------------------------------
1. SNR低下 (主役): SNR ∝ B0^p (既定 p=1)。低磁場ほどノイズ増。
   MR magnitudeのノイズは Rician（複素ガウシアン→絶対値）。低SNRでは
   ノイズフロアと信号バイアスが現れる。ガウシアン加算では不正確。
2. 解像度低下/ボケ: 低磁場はSNR確保のため大ボクセル・低マトリクス・強い
   再構成フィルタ → 面内ガウシアンPSF / k空間トランケーション(Gibbs) /
   ダウンサンプルで再現。
3. (任意・近似) T1短縮によるコントラスト変化: 厳密には定量マップが要る。

ノイズ量の決め方
----------------
既存ノイズ σ_high を背景(空気)領域の Rayleigh 統計から実測し、磁場比で
スケールして目標 σ_low を決める:
    σ_low = σ_high · (B0_high / B0_low)^p
追加する複素ノイズ:  σ_add = sqrt(σ_low^2 − σ_high^2)
実際の低磁場画像があれば --ref-low でその背景から σ_low を直接実測できる
（「実画像と同等」に最も近い）。--target-snr / --noise-sigma で直接指定も可。

注意
----
- 単一コイル magnitude を仮定（Rician）。パラレルイメージング/多コイルは
  noncentral-chi + 空間変動g-factorになるため、本ツールは近似。
- T1コントラスト変化はオプションの粗い近似。T2強調など磁場ロバストな
  コントラストでは省略しても実用的。
- through-plane(スライス厚)の変更は mri_slice_sim.py の担当。本ツールは
  面内の解像度・ノイズに集中する（各スライスを独立処理）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import gaussian_filter, zoom

from mri_slice_sim import load_series, _encode_px, Series


# --------------------------------------------------------------------------- #
# ノイズ推定・付加
# --------------------------------------------------------------------------- #
def estimate_sigma_rayleigh(vol: np.ndarray, corner_frac: float = 0.0625) -> float:
    """画像コーナー(空気領域)の magnitude から Rician の素ガウシアン σ を推定。

    背景 magnitude は Rayleigh 分布: E[M^2] = 2σ^2 ⇒ σ = sqrt(E[M^2]/2)。
    パーセンタイルで分布を「切る」と過小評価になるため、コーナー全画素の
    二次モーメントを用いる。4コーナーのうち最も空気らしい(平均最小)ものを採用し、
    組織が入り込んだコーナーの混入を避ける。
    """
    R, C = vol.shape[-2:]
    fr = max(4, int(R * corner_frac))
    fc = max(4, int(C * corner_frac))
    corners = [vol[..., :fr, :fc], vol[..., :fr, -fc:],
               vol[..., -fr:, :fc], vol[..., -fr:, -fc:]]
    m2 = [float(np.mean(np.clip(c, 0, None) ** 2)) for c in corners]
    return float(np.sqrt(min(m2) / 2.0))


def signal_level(vol: np.ndarray, fg_percentile: float = 60.0) -> float:
    """前景の代表信号レベル（SNR表示用）。"""
    fg = vol[vol > np.percentile(vol, fg_percentile)]
    return float(np.median(fg)) if fg.size else float(vol.max())


def add_rician(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """magnitude img に σ の複素ガウシアンを加え Rician magnitude を返す。"""
    if sigma <= 0:
        return img
    m = np.clip(img, 0, None)
    n1 = rng.normal(0.0, sigma, m.shape)
    n2 = rng.normal(0.0, sigma, m.shape)
    return np.sqrt((m + n1) ** 2 + n2 ** 2)


# --------------------------------------------------------------------------- #
# 解像度劣化
# --------------------------------------------------------------------------- #
def inplane_blur(img: np.ndarray, sigma_mm: float, ps: list) -> np.ndarray:
    if sigma_mm <= 0:
        return img
    return gaussian_filter(img, (sigma_mm / float(ps[0]), sigma_mm / float(ps[1])))


def kspace_truncate(img: np.ndarray, keep: float) -> np.ndarray:
    """k空間中央の keep 割合のみ残す（低マトリクス取得→Gibbsリンギング/解像度低下）。"""
    if keep >= 1.0:
        return img
    F = np.fft.fftshift(np.fft.fft2(img))
    R, C = img.shape
    kr, kc = int(R * keep / 2), int(C * keep / 2)
    cr, cc = R // 2, C // 2
    mask = np.zeros_like(F, dtype=bool)
    mask[cr - kr:cr + kr, cc - kc:cc + kc] = True
    F[~mask] = 0
    return np.abs(np.fft.ifft2(np.fft.ifftshift(F)))


def resolution_downup(img: np.ndarray, factor: float,
                      rng: np.random.Generator, sigma: float) -> np.ndarray:
    """factor倍に縮小→(低解像で)Ricianノイズ付加→元サイズへ拡大。

    低磁場の大ボクセル取得を模す。ノイズは取得解像度で乗るので縮小後に付加し、
    拡大で空間相関（実機の低解像ノイズらしさ）が入る。
    """
    if factor >= 1.0:
        return add_rician(img, sigma, rng)
    small = zoom(img, factor, order=1)
    small = add_rician(small, sigma, rng)
    back = zoom(small, (img.shape[0] / small.shape[0],
                        img.shape[1] / small.shape[1]), order=1)
    return back


# --------------------------------------------------------------------------- #
# 実低磁場プロファイル較正・適用の共有ヘルパ（unpaired・スケール不変量）
# --------------------------------------------------------------------------- #
def inplane_resolution_mm(ds) -> tuple[float, float]:
    """(再構成, 取得) 面内解像度[mm] を返す。

    再構成 = min(PixelSpacing)。取得 = FOV / AcquisitionMatrix（最も粗い方向）で、
    ゼロフィル補間で見かけ細かい場合でも真の取得解像度を反映する。タグが無ければ
    取得=再構成にフォールバック。
    """
    ps = [float(x) for x in ds.PixelSpacing]
    recon = float(min(ps))
    rows, cols = int(ds.Rows), int(ds.Columns)
    fov = max(ps[0] * rows, ps[1] * cols)
    acq = recon
    am = getattr(ds, "AcquisitionMatrix", None)
    if am:
        vals = [int(v) for v in am if int(v) > 0]
        if vals:
            acq = float(fov / min(vals))      # 最も粗い方向 ≈ 実効解像度
    return recon, max(acq, recon)


def foreground_mask(img: np.ndarray, pct: float = 55.0) -> np.ndarray:
    return img > np.percentile(img, pct)


def res_to_blur_sigma(res_low: float, res_high: float) -> float:
    """取得面内解像度の差を等価ガウシアンσ[mm]に換算（FWHM差→σ）。

    解像度は実低磁場/高磁場の DICOM PixelSpacing（取得面内解像度）から取る。
    スペクトルからの解像度推定は強ノイズで不安定なため、安定なこの方式を主とする。
    """
    fwhm_add = float(np.sqrt(max(res_low ** 2 - res_high ** 2, 0.0)))
    return fwhm_add / 2.3548


def fg_quantiles(vol: np.ndarray, probs: np.ndarray, pct: float = 55.0) -> np.ndarray:
    """前景輝度の分位値。コントラスト（組織輝度関係）の記述子。"""
    fg = vol[vol > np.percentile(vol, pct)]
    return np.quantile(fg, probs)


def histogram_match(img: np.ndarray, src_q: np.ndarray,
                    tgt_q: np.ndarray) -> np.ndarray:
    """src分位→tgt分位 の単調写像で輝度を変換（ヒストグラムマッチング）。"""
    return np.interp(img, src_q, tgt_q)


# --------------------------------------------------------------------------- #
# T1コントラスト（任意・近似）
# --------------------------------------------------------------------------- #
def approx_t1_contrast(img: np.ndarray, strength: float) -> np.ndarray:
    """低磁場のT1短縮を粗く近似（コントラスト圧縮）。strength∈[0,1]。

    定量マップが無いため、輝度ヒストグラムを中央値方向へ圧縮する経験的処理。
    厳密でないので既定OFF。T2強調などでは不要。
    """
    if strength <= 0:
        return img
    med = np.median(img[img > np.percentile(img, 60)])
    return med + (img - med) * (1.0 - 0.5 * strength)


# --------------------------------------------------------------------------- #
def simulate_lowfield(in_dir: str, out_dir: str, pattern: str,
                      field_high: float, field_low: float, snr_exp: float,
                      target_snr: float | None, noise_sigma: float | None,
                      ref_low: str | None,
                      blur_mm: float, in_plane_res: float | None,
                      kspace_keep: float, downsample: float,
                      t1_strength: float, seed: int, desc: str,
                      profile: str | None = None) -> None:
    s = load_series(in_dir, pattern)
    vol = s.volume.copy()
    ps = [float(x) for x in s.template.PixelSpacing]

    # 実低磁場プロファイル（lowfield_calibrate.py 出力）を適用
    if profile is not None:
        prof = json.load(open(profile))
        # コントラスト: 高磁場前景を低磁場の正規化分位へヒストグラムマッチング
        nq = len(prof["intensity_quantiles"])
        probs = np.linspace(0.0, 1.0, nq)
        src_q = fg_quantiles(vol, probs)
        scale = signal_level(vol)                       # 高磁場の代表レベルで再スケール
        tgt_q = np.asarray(prof["intensity_quantiles"]) * scale
        # 0アンカー: 背景(空気)を0付近に保つ（前景分位だけだと空気が持ち上がる）
        src_q = np.concatenate([[0.0], src_q])
        tgt_q = np.concatenate([[0.0], tgt_q])
        vol = histogram_match(vol, src_q, tgt_q)
        # ノイズ・解像度の目標を上書き（解像度は取得PixelSpacing差から安定に換算）
        target_snr = float(prof["target_snr"])
        res_low = float(prof["resolution_mm"])
        blur_mm = max(blur_mm, res_to_blur_sigma(res_low, min(ps)))
        t1_strength = 0.0                               # コントラストは適用済み
        print(f"[prof] {profile} name={prof.get('name','?')} "
              f"target_snr={target_snr:.1f} res_low={res_low:.2f}mm "
              f"res_high={min(ps):.2f}mm -> blur={blur_mm:.2f}mm")

    sigma_high = estimate_sigma_rayleigh(vol)
    sig = signal_level(vol)

    # 目標ノイズ σ_low の決定（優先順位: noise_sigma > target_snr > ref_low > field比）
    if noise_sigma is not None:
        sigma_low = float(noise_sigma)
        how = "noise-sigma"
    elif target_snr is not None:
        sigma_low = sig / target_snr
        how = "target-snr"
    elif ref_low is not None:
        sref = load_series(ref_low, pattern)
        sigma_low = estimate_sigma_rayleigh(sref.volume)
        # 信号スケールが違う可能性 → 参照のSNRを高磁場の信号レベルへ換算
        sref_sig = signal_level(sref.volume)
        sigma_low = sigma_low * (sig / (sref_sig + 1e-9))
        how = f"ref-low({os.path.basename(ref_low)})"
    else:
        sigma_low = sigma_high * (field_high / field_low) ** snr_exp
        how = f"field {field_high}T->{field_low}T ^{snr_exp}"

    sigma_add = float(np.sqrt(max(sigma_low ** 2 - sigma_high ** 2, 0.0)))

    # 解像度ボケσの決定（in_plane_res 指定があれば等価ガウシアン幅へ換算）
    if in_plane_res is not None and in_plane_res > min(ps):
        # 取得解像度をFWHMとみなし、追加ボケ = sqrt(res_low^2 - res_high^2)
        fwhm_add = np.sqrt(max(in_plane_res ** 2 - min(ps) ** 2, 0.0))
        blur_eff = blur_mm + fwhm_add / 2.3548
    else:
        blur_eff = blur_mm

    snr_before = sig / (sigma_high + 1e-9)
    snr_after = sig / (sigma_low + 1e-9)
    print(f"[load] {vol.shape[0]} slices {vol.shape[1]}x{vol.shape[2]}, ps={ps}mm")
    print(f"[noise] σ_high≈{sigma_high:.2f} (SNR≈{snr_before:.1f}) -> "
          f"σ_low={sigma_low:.2f} (SNR≈{snr_after:.1f}) via {how}; σ_add={sigma_add:.2f}")
    print(f"[res ] blur={blur_eff:.2f}mm kspace_keep={kspace_keep} downsample={downsample}")

    os.makedirs(out_dir, exist_ok=True)
    new_uid = generate_uid()
    rng = np.random.default_rng(seed)
    slope = float(getattr(s.template, "RescaleSlope", 1) or 1)
    intercept = float(getattr(s.template, "RescaleIntercept", 0) or 0)

    for k, ds in enumerate(s.datasets):
        img = vol[k].copy()
        if t1_strength > 0:
            img = approx_t1_contrast(img, t1_strength)
        img = inplane_blur(img, blur_eff, ps)
        if kspace_keep < 1.0:
            img = kspace_truncate(img, kspace_keep)
        if downsample < 1.0:
            img = resolution_downup(img, downsample, rng, sigma_add)
        else:
            img = add_rician(img, sigma_add, rng)

        px = _encode_px(ds, img, slope, intercept)
        ds = pydicom.dcmread(ds.filename)
        ds.PixelData = px.tobytes()
        ds.Rows, ds.Columns = px.shape
        ds.SeriesInstanceUID = new_uid
        ds.SOPInstanceUID = generate_uid()
        if hasattr(ds, "file_meta"):
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.SeriesDescription = desc
        it = list(getattr(ds, "ImageType", ["DERIVED", "SECONDARY"]))
        if it and it[0] == "ORIGINAL":
            it[0] = "DERIVED"
        ds.ImageType = it
        ds.save_as(os.path.join(out_dir, f"{k + 1:05d}.DCM"))

    print(f"[save] {len(s.datasets)} files -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="高磁場DICOMシリーズのフォルダ")
    ap.add_argument("output", help="出力フォルダ")
    ap.add_argument("--pattern", default="*", help="globパターン")
    # SNR / ノイズ
    ap.add_argument("--field-high", type=float, default=3.0, help="入力の磁場強度[T]")
    ap.add_argument("--field-low", type=float, default=0.5, help="目標の磁場強度[T]")
    ap.add_argument("--snr-exponent", type=float, default=1.0, help="SNR∝B0^p の p")
    ap.add_argument("--target-snr", type=float, default=None, help="目標SNRを直接指定")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="追加前の目標σを実値で直接指定")
    ap.add_argument("--ref-low", default=None,
                    help="実低磁場シリーズ。背景からσを実測して合わせる")
    # 解像度
    ap.add_argument("--blur-mm", type=float, default=0.0, help="面内ガウシアンPSF σ[mm]")
    ap.add_argument("--in-plane-res", type=float, default=None,
                    help="目標面内解像度[mm]（等価ボケに換算して適用）")
    ap.add_argument("--kspace-keep", type=float, default=1.0,
                    help="k空間中央の保持割合(0-1]。<1でGibbs/解像度低下")
    ap.add_argument("--downsample", type=float, default=1.0,
                    help="取得解像度の縮小率(0-1]。<1で縮小→ノイズ→拡大")
    # コントラスト
    ap.add_argument("--t1-strength", type=float, default=0.0,
                    help="低磁場T1短縮の近似強度[0-1]（既定0=OFF, 近似なので注意）")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード")
    ap.add_argument("--desc", default="Simulated low-field", help="SeriesDescription")
    ap.add_argument("--profile", default=None,
                    help="lowfield_calibrate.py が出力したコントラスト別プロファイル(.json)。"
                         "ノイズ/解像度/コントラストを実低磁場に合わせて上書き")
    args = ap.parse_args()

    simulate_lowfield(args.input, args.output, args.pattern,
                      args.field_high, args.field_low, args.snr_exponent,
                      args.target_snr, args.noise_sigma, args.ref_low,
                      args.blur_mm, args.in_plane_res, args.kspace_keep,
                      args.downsample, args.t1_strength, args.seed, args.desc,
                      args.profile)


if __name__ == "__main__":
    main()
