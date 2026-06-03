#!/usr/bin/env python3
"""
calibrate.py
============
同一患者・同一セッションで撮像した「実3D」と「実2D(正解)」のT2強調ペアから、
2Dシミュレーションのパラメータを実測較正・検証する。

目的
----
1. スライスプロファイル(SSP)推定 : 実2Dに最も一致する through-plane プロファイル
   P(z) を台形(FWHM, ramp)で当てはめ、ssp.npy に保存して本体シミュレータに焼き込む。
2. 精度の定量検証 : 較正後の擬似2D vs 実2D を NRMSE / Pearson r / SSIM で評価。

前提
----
- 同一セッション・体動なし → DICOM患者座標(IPP/IOP)だけで画素対応が取れる
  （レジストレーション不要）。各実2Dスライスの幾何で3Dを再標本化する。
- magnitude / 別コントラストでも、強度の線形変換 a·x+b を同時推定して吸収する
  （SSP推定が単純なスケール差で歪まないようにするため）。これは厳密なコントラスト
  変換ではなく、SSPを公平に測るための正規化。

forward model
-------------
    real2D(plane) ≈ a · Σ_t w(t; FWHM, ramp) · S3D(plane + t·n) + b
    （+ Rician noise）

各2Dスライス平面上で、法線 n 方向の密なオフセット t で3Dをトリリニア標本化した
平面群 S3D(t) を「一度だけ」計算し、プロファイル重み w(t) は線形結合で与える。
最適化は (FWHM, ramp) の2次元のみ非線形（a,b は各反復で閉形式）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates, uniform_filter, gaussian_filter
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import spearmanr

from mri_slice_sim import load_series, input_affine, Series


# --------------------------------------------------------------------------- #
# プロファイルモデル（台形）
# --------------------------------------------------------------------------- #
def trapezoid_weights(offsets: np.ndarray, fwhm: float, ramp: float) -> np.ndarray:
    """中心0の台形プロファイルを offsets で評価し、総和1に正規化して返す。"""
    half = fwhm / 2.0
    a = np.abs(offsets)
    w = np.ones_like(offsets, dtype=float)
    if ramp > 1e-6:
        edge = (a > half - ramp) & (a < half)
        w[edge] = (half - a[edge]) / ramp
    w[a >= half] = 0.0
    s = w.sum()
    return w / s if s > 0 else w


# --------------------------------------------------------------------------- #
# 3Dを各2Dスライス平面で標本化
# --------------------------------------------------------------------------- #
def plane_grid(ds2d: pydicom.Dataset):
    """実2Dスライスの面内世界座標 base(R,C,3) と法線 n を返す。"""
    iop = np.array(ds2d.ImageOrientationPatient, float)
    u, v = iop[0:3], iop[3:6]
    ipp = np.array(ds2d.ImagePositionPatient, float)
    ps = [float(x) for x in ds2d.PixelSpacing]   # [row, col]
    R, C = int(ds2d.Rows), int(ds2d.Columns)
    cc = np.arange(C) * ps[1]
    rr = np.arange(R) * ps[0]
    base = (ipp[None, None, :]
            + cc[None, :, None] * u[None, None, :]
            + rr[:, None, None] * v[None, None, :])
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    return base, n


def sample_planes(vol3d: np.ndarray, Ainv3d: np.ndarray,
                  base: np.ndarray, n: np.ndarray,
                  offsets: np.ndarray) -> np.ndarray:
    """3Dを base+t*n の各平面でトリリニア標本化。返り値 (len(offsets), R, C)。"""
    R, C, _ = base.shape
    out = np.empty((offsets.size, R, C), dtype=np.float32)
    for i, t in enumerate(offsets):
        world = base + t * n[None, None, :]
        homog = np.concatenate([world, np.ones((R, C, 1))], axis=-1)
        vox = homog @ Ainv3d.T                 # [col,row,slice]
        coords = np.stack([vox[..., 2], vox[..., 1], vox[..., 0]], axis=0)
        out[i] = map_coordinates(vol3d, coords, order=1, mode="constant", cval=0.0)
    return out


# --------------------------------------------------------------------------- #
# 較正本体
# --------------------------------------------------------------------------- #
def _fit_ab(x: np.ndarray, y: np.ndarray):
    """y ≈ a*x + b の最小二乗解 (a, b)。"""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b


def fit_profile(Pmat: np.ndarray, y: np.ndarray, offsets: np.ndarray,
                nominal: float):
    """
    Pmat: (T, N) 各オフセットの標本値（前景画素を連結）
    y   : (N,)   対応する実2D画素値
    戻り: dict(fwhm, ramp, a, b, sse)
    """
    def objective(params):
        fwhm, ramp = params
        w = trapezoid_weights(offsets, fwhm, ramp)
        sim = w @ Pmat                          # (N,)
        a, b = _fit_ab(sim, y)
        resid = a * sim + b - y
        return float(resid @ resid)

    # 粗いグリッドから初期値 → 局所最適化
    best = None
    for fwhm0 in np.linspace(0.4 * nominal, 2.0 * nominal, 9):
        for ramp0 in [0.0, 0.15 * nominal, 0.3 * nominal]:
            val = objective([fwhm0, ramp0])
            if best is None or val < best[0]:
                best = (val, fwhm0, ramp0)
    res = minimize(objective, x0=[best[1], best[2]], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-6, maxiter=500))
    fwhm, ramp = res.x
    ramp = max(0.0, min(ramp, fwhm / 2.0))
    w = trapezoid_weights(offsets, fwhm, ramp)
    sim = w @ Pmat
    a, b = _fit_ab(sim, y)
    return dict(fwhm=float(fwhm), ramp=float(ramp), a=float(a), b=float(b),
                sse=float(res.fun))


def fit_profile_inplane(s3d: Series, s2d: Series, Ainv: np.ndarray,
                        offsets: np.ndarray, nominal: float,
                        fg_percentile: float, n_slices: int = 3,
                        ds_factor: int = 2, init=None):
    """
    面内ガウシアンPSF σ を SSP と同時推定する。
    real2D ≈ a·[ Gauss(σ) ∘ Σ_t w(t)·S3D(plane+t·n) ] + b
    全画像が要るので少数スライスを ds_factor で間引いた解像度で当てはめる。
    返り値 dict(fwhm, ramp, sigma_mm, a, b, r0, r1)
      r0: σ=0(面内ボケ無し)の相関 / r1: 最適σでの相関
    """
    n2d = len(s2d.datasets)
    idx = np.unique(np.linspace(0, n2d - 1, min(n_slices, n2d)).round().astype(int))
    data = []
    for k in idx:
        ds = s2d.datasets[k]
        real = (ds.pixel_array.astype(np.float64)
                * float(getattr(ds, "RescaleSlope", 1) or 1)
                + float(getattr(ds, "RescaleIntercept", 0) or 0))
        base, n = plane_grid(ds)
        planes = sample_planes(s3d.volume, Ainv, base, n, offsets)  # (T,R,C)
        planes = planes[:, ::ds_factor, ::ds_factor]
        real = real[::ds_factor, ::ds_factor]
        mask = real > np.percentile(real, fg_percentile)
        ps_ds = float(ds.PixelSpacing[0]) * ds_factor
        data.append((planes, real, mask, ps_ds))

    def assemble(fwhm, ramp, sigma_px):
        w = trapezoid_weights(offsets, fwhm, ramp)
        xs, ys = [], []
        for planes, real, mask, _ in data:
            comb = np.tensordot(w, planes, axes=(0, 0))
            if sigma_px > 1e-3:
                comb = gaussian_filter(comb, sigma_px)
            xs.append(comb[mask]); ys.append(real[mask])
        return np.concatenate(xs), np.concatenate(ys)

    def objective(p):
        fwhm, ramp, sigma_px = p
        if fwhm <= 0 or ramp < 0 or sigma_px < 0:
            return 1e18
        x, y = assemble(fwhm, ramp, min(sigma_px, 6.0))
        a, b = _fit_ab(x, y)
        r = a * x + b - y
        return float(r @ r)

    f0, r0p = (init or (nominal, 0.0))[0], (init or (nominal, 0.0))[1]
    res = minimize(objective, x0=[f0, r0p, 0.8], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-6, maxiter=800))
    fwhm, ramp, sigma_px = res.x
    ramp = max(0.0, min(ramp, fwhm / 2.0)); sigma_px = max(0.0, sigma_px)
    ps_ds = data[0][3]

    x0, y0 = assemble(fwhm, ramp, 0.0)
    x1, y1 = assemble(fwhm, ramp, sigma_px)
    a0, b0 = _fit_ab(x0, y0); a1, b1 = _fit_ab(x1, y1)
    r0 = float(np.corrcoef(a0 * x0 + b0, y0)[0, 1])
    r1 = float(np.corrcoef(a1 * x1 + b1, y1)[0, 1])
    return dict(fwhm=float(fwhm), ramp=float(ramp),
                sigma_mm=float(sigma_px * ps_ds), a=float(a1), b=float(b1),
                r0=r0, r1=r1)


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #
def nrmse(sim: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    d = sim[mask] - ref[mask]
    rng = ref[mask].max() - ref[mask].min()
    return float(np.sqrt(np.mean(d ** 2)) / (rng + 1e-9))


def pearson(sim: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    x, y = sim[mask], ref[mask]
    return float(np.corrcoef(x, y)[0, 1])


def ssim(sim: np.ndarray, ref: np.ndarray, win: int = 7) -> float:
    """ローカル窓SSIM(平均)。前処理なしの簡易版。"""
    sim = sim.astype(np.float64); ref = ref.astype(np.float64)
    L = ref.max() - ref.min() + 1e-9
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_x = uniform_filter(sim, win); mu_y = uniform_filter(ref, win)
    sx = uniform_filter(sim * sim, win) - mu_x ** 2
    sy = uniform_filter(ref * ref, win) - mu_y ** 2
    sxy = uniform_filter(sim * ref, win) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2)
    return float(np.mean(num / den))


# --------------------------------------------------------------------------- #
# 診断 (--qa-dir)
# --------------------------------------------------------------------------- #
def estimate_shift(ref: np.ndarray, mov: np.ndarray):
    """位相相関で ref に対する mov の面内シフト(drow, dcol)[画素]を推定。"""
    F = np.fft.fft2(ref - ref.mean())
    G = np.fft.fft2(mov - mov.mean())
    R = F * np.conj(G)
    R /= np.abs(R) + 1e-9
    corr = np.fft.ifft2(R).real
    peak = np.array(np.unravel_index(np.argmax(corr), corr.shape), float)
    for i, dim in enumerate(corr.shape):
        if peak[i] > dim // 2:
            peak[i] -= dim
    return peak  # (drow, dcol)


def geometry_report(s3d: Series, s2d: Series) -> str:
    """FrameOfReference / 向き / 範囲の整合性を文字列で返す。"""
    lines = []
    f3 = getattr(s3d.template, "FrameOfReferenceUID", None)
    f2 = getattr(s2d.template, "FrameOfReferenceUID", None)
    lines.append(f"FrameOfReferenceUID 3D == 2D : {f3 == f2}")
    if f3 != f2:
        lines.append("  !! 異なる基準座標系 → IPP/IOPの直接対応は無効。レジストレーション必須。")
    iop3v = np.array(s3d.template.ImageOrientationPatient, float)
    iop2v = np.array(s2d.template.ImageOrientationPatient, float)
    lines.append(f"IOP 3D : {[round(float(x),3) for x in iop3v]}")
    lines.append(f"IOP 2D : {[round(float(x),3) for x in iop2v]}")
    # 2D法線 vs 3D法線の角度（奥行き方向の解像度混入を診断）
    n3 = s3d.normal
    n2 = np.cross(iop2v[0:3], iop2v[3:6]); n2 /= np.linalg.norm(n2)
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(n2, n3)), 0, 1)))
    lines.append(f"2D法線 vs 3D法線の角度 : {ang:.1f}度  "
                 "(0=同方向で最良/大きいほど3Dの粗い軸が奥行きに混入)")
    # 3Dの世界座標バウンディングbox
    A = input_affine(s3d)
    K, R, C = s3d.volume.shape
    corners = np.array([A @ [c, r, k, 1.0]
                        for k in (0, K - 1) for r in (0, R - 1)
                        for c in (0, C - 1)])[:, :3]
    lo, hi = corners.min(0), corners.max(0)
    lines.append(f"3D world bbox  X[{lo[0]:.1f},{hi[0]:.1f}] "
                 f"Y[{lo[1]:.1f},{hi[1]:.1f}] Z[{lo[2]:.1f},{hi[2]:.1f}] (LPS mm)")
    ipps = np.array([list(map(float, d.ImagePositionPatient)) for d in s2d.datasets])
    p_lo, p_hi = ipps.min(0), ipps.max(0)
    lines.append(f"2D IPP range   X[{p_lo[0]:.1f},{p_hi[0]:.1f}] "
                 f"Y[{p_lo[1]:.1f},{p_hi[1]:.1f}] Z[{p_lo[2]:.1f},{p_hi[2]:.1f}]")
    inside = np.all((ipps >= lo - 5) & (ipps <= hi + 5), axis=1).mean()
    lines.append(f"2D origin が3D範囲内の割合 : {100*inside:.0f}%  "
                 "(低い→位置ずれ/別座標系)")
    return "\n".join(lines)


def qa_dump(qa_dir: str, sim: np.ndarray, real: np.ndarray, mask: np.ndarray,
            ps: list, geom: str):
    """sim/real/diff の保存と数値診断。matplotlibがあればPNGも。"""
    os.makedirs(qa_dir, exist_ok=True)
    np.save(os.path.join(qa_dir, "real.npy"), real)
    np.save(os.path.join(qa_dir, "sim.npy"), sim)
    np.save(os.path.join(qa_dir, "diff.npy"), sim - real)
    with open(os.path.join(qa_dir, "geometry.txt"), "w") as f:
        f.write(geom + "\n")

    # 残差シフト（位置ずれ診断）
    drow, dcol = estimate_shift(real, sim)
    shift_mm = (drow * ps[0], dcol * ps[1])
    # 線形 vs 単調（コントラスト非線形性の診断）
    x, y = sim[mask], real[mask]
    r_lin = float(np.corrcoef(x, y)[0, 1])
    r_mono = float(spearmanr(x, y).statistic)

    report = [
        geom,
        "",
        f"residual in-plane shift : drow={drow:.1f}px dcol={dcol:.1f}px "
        f"=> ({shift_mm[0]:.2f}, {shift_mm[1]:.2f}) mm",
        "   |shift|>1px は位置ずれ/レジストレーション不足を示唆",
        f"Pearson r (linear)   : {r_lin:.4f}",
        f"Spearman r (monotone): {r_mono:.4f}",
        "   r_mono >> r_lin なら単調な非線形コントラスト差 → 強度マップの高度化で改善",
        "   両方低いなら 位置ずれ / 面内解像度差(PSF/Gibbs) が主因",
    ]
    txt = "\n".join(report)
    with open(os.path.join(qa_dir, "diagnostics.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n=== QA diagnostics ===")
    print(txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vmax = np.percentile(real[mask], 99)
        d = sim - real
        dmax = np.percentile(np.abs(d[mask]), 99)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].imshow(real, cmap="gray", vmin=0, vmax=vmax); ax[0, 0].set_title("real 2D")
        ax[0, 1].imshow(sim, cmap="gray", vmin=0, vmax=vmax); ax[0, 1].set_title("simulated 2D")
        ax[1, 0].imshow(d, cmap="bwr", vmin=-dmax, vmax=dmax); ax[1, 0].set_title("sim - real")
        ax[1, 1].hist2d(x, y, bins=128); ax[1, 1].set_xlabel("sim"); ax[1, 1].set_ylabel("real")
        ax[1, 1].set_title("joint histogram")
        for a in ax.ravel()[:3]:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(qa_dir, "compare.png"), dpi=110)
        plt.close(fig)
        print(f"[qa  ] images -> {qa_dir}/compare.png, *.npy, diagnostics.txt")
    except Exception as e:
        print(f"[qa  ] matplotlib無し: 配列(.npy)と診断テキストのみ保存 ({e})")


# --------------------------------------------------------------------------- #
def calibrate(dir3d: str, dir2d: str, pattern: str,
              out_ssp: str, max_fit_slices: int, pixel_budget: int,
              fg_percentile: float, qa_dir: str | None = None,
              fit_inplane: bool = False):
    s3d = load_series(dir3d, pattern)
    s2d = load_series(dir2d, pattern)
    Ainv = np.linalg.inv(input_affine(s3d))
    nominal = float(getattr(s2d.template, "SliceThickness", 5.0) or 5.0)
    print(f"[load] 3D: {s3d.volume.shape}  2D: {len(s2d.datasets)} slices, "
          f"nominal thickness={nominal}mm")

    # オフセット格子（公称厚の±1.5倍を 0.25mm 刻み）
    span = 1.5 * nominal
    step = min(0.25, nominal / 20)
    offsets = np.arange(-span, span + step / 2, step)

    # fit用スライスを中央から選ぶ
    n2d = len(s2d.datasets)
    idx = np.linspace(0, n2d - 1, min(max_fit_slices, n2d)).round().astype(int)
    idx = np.unique(idx)

    cols, ys = [], []
    rng = np.random.default_rng(0)
    for k in idx:
        ds = s2d.datasets[k]
        real = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        real = real * slope + inter
        base, n = plane_grid(ds)
        planes = sample_planes(s3d.volume, Ainv, base, n, offsets)  # (T,R,C)

        thr = np.percentile(real, fg_percentile)
        mask = real > thr
        flat = mask.ravel()
        fg = np.where(flat)[0]
        if fg.size == 0:
            continue
        # 画素予算に合わせて間引き
        per = max(1, pixel_budget // idx.size)
        if fg.size > per:
            fg = rng.choice(fg, per, replace=False)
        Pmat = planes.reshape(offsets.size, -1)[:, fg]    # (T, n)
        cols.append(Pmat)
        ys.append(real.ravel()[fg])

    Pmat = np.concatenate(cols, axis=1)
    y = np.concatenate(ys)
    print(f"[fit ] slices={idx.size} offsets={offsets.size} pixels={y.size}")

    fit = fit_profile(Pmat, y, offsets, nominal)
    # 矩形(公称厚)ベースラインとの比較
    w_rect = trapezoid_weights(offsets, nominal, 0.0)
    sim_r = w_rect @ Pmat; a_r, b_r = _fit_ab(sim_r, y)
    sse_rect = float(np.sum((a_r * sim_r + b_r - y) ** 2))

    print("\n=== Fitted slice profile (trapezoid) ===")
    print(f"  FWHM         : {fit['fwhm']:.2f} mm   (nominal {nominal:.2f})")
    print(f"  ramp(edge)   : {fit['ramp']:.2f} mm")
    print(f"  intensity a,b: {fit['a']:.4f}, {fit['b']:.2f}")
    print(f"  SSE  fitted  : {fit['sse']:.3e}")
    print(f"  SSE  rect    : {sse_rect:.3e}   (improvement "
          f"{100*(1-fit['sse']/sse_rect):.1f}%)")

    # 面内PSFを同時推定（オプション）
    if fit_inplane:
        ip = fit_profile_inplane(s3d, s2d, Ainv, offsets, nominal, fg_percentile,
                                 init=(fit['fwhm'], fit['ramp']))
        print("\n=== Joint fit with in-plane PSF ===")
        print(f"  FWHM         : {ip['fwhm']:.2f} mm   (nominal {nominal:.2f})")
        print(f"  ramp(edge)   : {ip['ramp']:.2f} mm")
        print(f"  in-plane σ   : {ip['sigma_mm']:.2f} mm  (面内ガウシアンPSF)")
        print(f"  r  σ=0 -> σ* : {ip['r0']:.4f} -> {ip['r1']:.4f}")
        fit.update(sigma_mm=ip['sigma_mm'], fwhm_joint=ip['fwhm'],
                   ramp_joint=ip['ramp'], r_inplane=ip['r1'])

    # SSP保存（offset, weight）
    w_fit = trapezoid_weights(offsets, fit['fwhm'], fit['ramp'])
    np.save(out_ssp, np.vstack([offsets, w_fit]))
    with open(os.path.splitext(out_ssp)[0] + ".json", "w") as f:
        json.dump(fit, f, indent=2)
    print(f"[save] SSP -> {out_ssp}  params -> {os.path.splitext(out_ssp)[0]}.json")

    # fit画素での指標（散布点）
    sim_fit = (w_fit @ Pmat) * fit['a'] + fit['b']
    m = np.ones_like(y, bool)
    print("\n=== Validation on fit pixels (fitted profile) ===")
    print(f"  NRMSE : {nrmse(sim_fit, y, m):.4f}")
    print(f"  r     : {pearson(sim_fit, y, m):.4f}")

    # 代表スライス1枚を全画素で再構成して画像指標(SSIM含む)
    kc = int(idx[len(idx) // 2])
    ds = s2d.datasets[kc]
    real = (ds.pixel_array.astype(np.float64)
            * float(getattr(ds, "RescaleSlope", 1) or 1)
            + float(getattr(ds, "RescaleIntercept", 0) or 0))
    base, n = plane_grid(ds)
    planes = sample_planes(s3d.volume, Ainv, base, n, offsets)
    sim_img = np.tensordot(w_fit, planes, axes=(0, 0)) * fit['a'] + fit['b']
    mask = real > np.percentile(real, fg_percentile)
    print(f"\n=== Validation on full slice #{kc} (fitted profile) ===")
    print(f"  NRMSE : {nrmse(sim_img, real, mask):.4f}")
    print(f"  r     : {pearson(sim_img, real, mask):.4f}")
    print(f"  SSIM  : {ssim(sim_img, real):.4f}")

    if qa_dir:
        ps = [float(x) for x in ds.PixelSpacing]
        qa_dump(qa_dir, sim_img, real, mask, ps, geometry_report(s3d, s2d))
    return fit


# --------------------------------------------------------------------------- #
# 自己検証: 既知の台形SSPで合成した「2D」を、推定器が復元できるか
# --------------------------------------------------------------------------- #
def self_test(dir3d: str, pattern: str):
    print("=== SELF TEST: recover a known trapezoid SSP ===")
    s3d = load_series(dir3d, pattern)
    Ainv = np.linalg.inv(input_affine(s3d))
    # 3Dの軸位中央付近に仮想2Dスライス平面を数枚作る
    K, R, C = s3d.volume.shape
    A = input_affine(s3d)
    u = np.array([1., 0, 0]); v = np.array([0, 1., 0]); n = np.array([0, 0, 1.])
    nominal = 5.0
    span = 1.5 * nominal; step = 0.25
    offsets = np.arange(-span, span + step / 2, step)
    true_fwhm, true_ramp, true_a, true_b = 5.0, 1.0, 1.3, 50.0
    w_true = trapezoid_weights(offsets, true_fwhm, true_ramp)

    ps_row = float(s3d.template.PixelSpacing[0])
    ps_col = float(s3d.template.PixelSpacing[1])
    cc = np.arange(C) * ps_col
    rr = np.arange(R) * ps_row
    cols, ys = [], []
    rng = np.random.default_rng(1)
    for kc in np.linspace(K * 0.3, K * 0.7, 5):
        ipp = (A @ np.array([0, 0, kc, 1.0]))[:3]
        base = (ipp[None, None, :]
                + cc[None, :, None] * u[None, None, :]
                + rr[:, None, None] * v[None, None, :])
        planes = sample_planes(s3d.volume, Ainv, base, n, offsets)
        synth = true_a * np.tensordot(w_true, planes, axes=(0, 0)) + true_b
        synth += rng.normal(0, 5.0, synth.shape)   # ノイズ
        real = synth
        thr = np.percentile(real, 40)
        fg = np.where((real > thr).ravel())[0]
        fg = rng.choice(fg, min(8000, fg.size), replace=False)
        cols.append(planes.reshape(offsets.size, -1)[:, fg])
        ys.append(real.ravel()[fg])

    Pmat = np.concatenate(cols, axis=1); y = np.concatenate(ys)
    fit = fit_profile(Pmat, y, offsets, nominal)
    print(f"  truth : FWHM={true_fwhm} ramp={true_ramp} a={true_a} b={true_b}")
    print(f"  fitted: FWHM={fit['fwhm']:.2f} ramp={fit['ramp']:.2f} "
          f"a={fit['a']:.3f} b={fit['b']:.1f}")
    ok = abs(fit['fwhm'] - true_fwhm) < 0.5 and abs(fit['ramp'] - true_ramp) < 0.6
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir3d", help="実3D DICOMフォルダ")
    ap.add_argument("dir2d", nargs="?", help="実2D(正解) DICOMフォルダ")
    ap.add_argument("--pattern", default="*", help="globパターン")
    ap.add_argument("--out-ssp", default="ssp.npy", help="推定SSPの保存先")
    ap.add_argument("--max-fit-slices", type=int, default=11, help="fitに使う2D枚数")
    ap.add_argument("--pixel-budget", type=int, default=60000, help="fit総画素数の上限")
    ap.add_argument("--fg-percentile", type=float, default=40.0,
                    help="前景マスクのしきい値パーセンタイル")
    ap.add_argument("--qa-dir", default=None,
                    help="診断出力先(sim/real/diff画像・残差シフト・線形/単調相関・幾何照合)")
    ap.add_argument("--fit-inplane", action="store_true",
                    help="面内ガウシアンPSF σ を SSP と同時推定（面内解像度差の較正）")
    ap.add_argument("--self-test", action="store_true",
                    help="既知SSPの復元テスト(dir2d不要)")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.dir3d, args.pattern)
        return
    if not args.dir2d:
        ap.error("dir2d が必要です（--self-test を除く）")
    calibrate(args.dir3d, args.dir2d, args.pattern, args.out_ssp,
              args.max_fit_slices, args.pixel_budget, args.fg_percentile,
              args.qa_dir, args.fit_inplane)


if __name__ == "__main__":
    main()
