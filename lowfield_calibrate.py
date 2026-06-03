#!/usr/bin/env python3
"""
lowfield_calibrate.py
=====================
実低磁場サンプルから、コントラスト別の「劣化プロファイル」を実測して JSON 化する。
unpaired（高磁場と別患者）でも使える、スケール不変量だけを保存する。

抽出する量（すべてスケール不変 ⇒ 別スキャナ/別患者の高磁場へ転用可）
------------------------------------------------------------------
- target_snr        : 代表信号レベル / ノイズσ（実低磁場の実測SNR）
- hf_fraction       : 高周波エネルギー比（実低磁場の実効解像度の指標）
- intensity_quantiles: 前景輝度の正規化分位（コントラスト＝組織輝度関係の記述子。
                       低磁場のT1短縮等の見た目をヒストグラムマッチングで移植するため）

使い方
------
    # 各コントラストごとに、その実低磁場フォルダから1つずつプロファイルを作る
    python lowfield_calibrate.py real_low_T1   --name T1    --out prof_T1.json
    python lowfield_calibrate.py real_low_T2   --name T2    --out prof_T2.json
    python lowfield_calibrate.py real_low_FLAIR --name FLAIR --out prof_FLAIR.json

    # 生成時に適用（高磁場の同コントラストへ）
    python lowfield_sim.py high_T1 out_T1 --profile prof_T1.json

注意
----
- ノイズσは画像コーナー(空気)の Rayleigh 統計から実測。多コイル/パラレルイメージング
  では noncentral-chi になり σ は近似だが、target_snr（信号/ノイズstd）は実測値として
  妥当。空間変動(g-factor)は本プロファイルでは平均化される。
- ヒストグラムマッチングは周辺分布の一致（unpairedで可能な範囲）。解剖学的構成比が
  高磁場側と大きく異なると歪むため、同部位・同コントラストのサンプルを用いること。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mri_slice_sim import load_series
from lowfield_sim import (estimate_noise_sigma, signal_level, fg_quantiles,
                          inplane_resolution_mm)


def recommend_knobs(res_low: float, res_high: float) -> dict:
    """解像度比 ρ=res_high/res_low から各解像度ノブの推奨値を出す。"""
    rho = min(1.0, res_high / (res_low + 1e-9))
    blur = float(np.sqrt(max(res_low ** 2 - res_high ** 2, 0.0)) / 2.3548)
    return {"rho": round(rho, 3), "downsample": round(rho, 3),
            "kspace_keep": round(rho, 3), "blur_mm": round(blur, 2)}


def calibrate_profile(low_dir: str, pattern: str, name: str,
                      nq: int, out_path: str, high_dir: str | None = None) -> dict:
    s = load_series(low_dir, pattern)
    vol = s.volume

    # ノイズ・SNR（背景マスク/PIに頑健: コーナーσが潰れていればLaplacianへ）
    sigma, sc, sl, masked = estimate_noise_sigma(vol)
    sig = signal_level(vol)
    target_snr = sig / (sigma + 1e-9)

    # 取得面内解像度（ゼロフィル対応: AcquisitionMatrix由来）
    recon_res, resolution_mm = inplane_resolution_mm(s.template)
    ps = [float(x) for x in s.template.PixelSpacing]

    # コントラスト（前景輝度の正規化分位）
    probs = np.linspace(0.0, 1.0, nq)
    q = fg_quantiles(vol, probs)
    q_norm = q / (sig + 1e-9)        # 代表レベルで正規化 → スケール不変

    prof = {
        "name": name,
        "source_low": os.path.abspath(low_dir),
        "n_slices": int(vol.shape[0]),
        "pixel_spacing": ps,
        "resolution_mm": resolution_mm,
        "recon_resolution_mm": float(recon_res),
        "sigma_measured": float(sigma),
        "signal_level": float(sig),
        "target_snr": float(target_snr),
        "intensity_quantiles": [float(x) for x in q_norm],
    }

    print(f"[calib] {name}: {vol.shape[0]} slices  ps={ps}mm")
    print(f"  resolution: recon(PixelSpacing)={recon_res:.2f}mm  "
          f"acquired(AcqMatrix)={resolution_mm:.2f}mm")
    print(f"  noise: corner σ={sc:.2f}  laplacian σ={sl:.2f}  "
          f"{'[背景マスク検出→laplacian採用]' if masked else '[corner採用]'}")
    print(f"  -> σ={sigma:.2f}  signal≈{sig:.1f}  target_snr={target_snr:.1f}")

    # 高磁場が与えられたら解像度ノブの推奨値を算出して表示・保存
    if high_dir:
        sh = load_series(high_dir, pattern)
        _, res_high = inplane_resolution_mm(sh.template)
        rec = recommend_knobs(resolution_mm, res_high)
        prof["recommended"] = {"res_high_mm": round(res_high, 3), **rec}
        print(f"  vs high-field acquired={res_high:.2f}mm -> ρ={rec['rho']}  "
              f"推奨: --downsample {rec['downsample']} | "
              f"--kspace-keep {rec['kspace_keep']} | --blur-mm {rec['blur_mm']}")

    print(f"  intensity quantiles (norm): "
          f"min={q_norm[0]:.2f} med={q_norm[nq//2]:.2f} max={q_norm[-1]:.2f}")
    with open(out_path, "w") as f:
        json.dump(prof, f, indent=2)
    print(f"[save] {out_path}")
    return prof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("low_dir", help="実低磁場DICOMシリーズ（単一コントラスト）")
    ap.add_argument("--name", default="lowfield", help="コントラスト名（T1/T2/FLAIR等）")
    ap.add_argument("--pattern", default="*", help="globパターン")
    ap.add_argument("--nq", type=int, default=64, help="輝度分位の点数")
    ap.add_argument("--out", default=None, help="出力JSON（既定: prof_<name>.json）")
    ap.add_argument("--high", default=None,
                    help="高磁場の同コントラストDICOM。解像度比ρと "
                         "--downsample/--kspace-keep/--blur-mm の推奨値を算出")
    args = ap.parse_args()

    out = args.out or f"prof_{args.name}.json"
    calibrate_profile(args.low_dir, args.pattern, args.name, args.nq, out, args.high)


if __name__ == "__main__":
    main()
