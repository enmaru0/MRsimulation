#!/usr/bin/env python3
"""
measure_noise.py
================
DICOM画像の指定ROIから、ノイズσ・信号・SNRを手動計測する。
自動推定(コーナー/Laplacian)が当てにならない時に、目で見て選んだ領域で測るためのツール。

計測方法（放射線で標準的な手法）
--------------------------------
- 背景(空気)ROI: magnitudeは Rayleigh 分布 ⇒ σ = sqrt(mean(M^2)/2)。
  （単純な std だと Rayleigh では std=0.655σ と過小になるので二次モーメントを使う）
- 均一組織ROI: そのROIの std を直接ノイズとみなす（構造が無い平坦部を選ぶこと）。
- SNR: 信号ROIの mean / ノイズσ。

ROI 指定
--------
"x,y,w,h"（左上のx=列, y=行, 幅, 高さ; 画素単位）。ビューアで読み取った座標を渡す。

出力は lowfield_sim にそのまま渡せる:
    --target-snr <SNR>     （スケール不変。推奨。実低磁場で測って高磁場へ適用）
    --noise-sigma <σ>      （同一スケールで使う時のみ）

例
--
    # 実低磁場で 信号ROIと背景ROIからSNRを測る
    python measure_noise.py MRP/hitachi025/T2AX --slice 10 \
        --signal-roi 220,210,30,30 --noise-roi 10,10,40,40

    # 均一組織ROIだけでσを測る
    python measure_noise.py MRP/hitachi025/T2AX --noise-roi 200,200,20,20 \
        --noise-region tissue

    # ROI未指定なら自動推定(コーナー/Laplacian)を表示
    python measure_noise.py MRP/hitachi025/T2AX
"""
from __future__ import annotations

import argparse

import numpy as np

from mri_slice_sim import load_series
from lowfield_sim import estimate_noise_sigma, signal_level


def parse_roi(text: str):
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def roi_pixels(imgs: np.ndarray, roi):
    x, y, w, h = roi
    R, C = imgs.shape[-2:]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(C, x + w), min(R, y + h)
    return imgs[..., y0:y1, x0:x1].reshape(imgs.shape[0], -1).ravel()


def sigma_from_roi(vals: np.ndarray, region: str):
    """ROI画素からノイズσを推定。background=Rayleigh, tissue=std。"""
    if region == "background":
        m2 = float(np.mean(np.clip(vals, 0, None) ** 2))
        return float(np.sqrt(m2 / 2.0))
    return float(np.std(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dicom_dir", help="DICOMシリーズ")
    ap.add_argument("--pattern", default="*", help="globパターン")
    ap.add_argument("--slice", default="mid",
                    help="スライス番号(0始まり) / 'mid' / 'all'（複数枚をまとめて計測）")
    ap.add_argument("--signal-roi", default=None, help="信号ROI 'x,y,w,h'")
    ap.add_argument("--noise-roi", default=None, help="ノイズROI 'x,y,w,h'")
    ap.add_argument("--noise-region", choices=["background", "tissue"],
                    default="background", help="ノイズROIの種別（既定 background）")
    args = ap.parse_args()

    s = load_series(args.dicom_dir, args.pattern)
    vol = s.volume                       # real-world値 (K,R,C)
    K = vol.shape[0]
    if args.slice == "all":
        imgs = vol
        sl_desc = f"all({K})"
    else:
        k = K // 2 if args.slice == "mid" else int(args.slice)
        imgs = vol[k:k + 1]
        sl_desc = str(k)
    print(f"[load] {K} slices {vol.shape[1]}x{vol.shape[2]}  計測スライス={sl_desc}")

    if not args.signal_roi and not args.noise_roi:
        sigma, sc, sl, masked = estimate_noise_sigma(vol)
        sig = signal_level(vol)
        print("ROI未指定 → 自動推定:")
        print(f"  corner σ={sc:.2f}  laplacian σ={sl:.2f}  "
              f"{'[背景マスク→laplacian]' if masked else '[corner]'}")
        print(f"  採用σ={sigma:.2f}  signal≈{sig:.1f}  target_snr={sig/(sigma+1e-9):.2f}")
        return

    sigma = sig = None
    if args.noise_roi:
        nv = roi_pixels(imgs, parse_roi(args.noise_roi))
        sigma = sigma_from_roi(nv, args.noise_region)
        print(f"[noise ROI] {args.noise_roi} ({args.noise_region})  "
              f"n={nv.size}  mean={nv.mean():.1f}  std={nv.std():.2f}  -> σ={sigma:.2f}")
    if args.signal_roi:
        sv = roi_pixels(imgs, parse_roi(args.signal_roi))
        sig = float(np.mean(sv))
        print(f"[signal ROI] {args.signal_roi}  n={sv.size}  "
              f"mean(signal)={sig:.1f}  std={sv.std():.2f}")

    if sigma is not None and sig is not None:
        snr = sig / (sigma + 1e-9)
        print(f"\n=== SNR = signal/σ = {sig:.1f} / {sigma:.2f} = {snr:.2f} ===")
        print(f"  lowfield_sim にそのまま: --target-snr {snr:.2f}")
        print(f"  （同一スケールで使う場合のみ: --noise-sigma {sigma:.2f}）")
    elif sigma is not None:
        print(f"\nσ={sigma:.2f}。信号ROIも指定するとSNRを算出します。")
        print(f"  同一スケールで使う場合: --noise-sigma {sigma:.2f}")


if __name__ == "__main__":
    main()
