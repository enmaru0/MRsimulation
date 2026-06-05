#!/usr/bin/env python3
"""
docs/make_figures.py
====================
ドキュメント用の説明図を **合成ファントム** から生成する（患者データ不使用＝コミット可）。
出力: docs/images/*.png

各図は recon_motion.py / mri_slice_sim.py の処理を可視化する:
  recon_pipeline.png  k空間 → コイル画像 → RSS（再構成）
  kspace_dc.png       DC中心 vs 端（--kspace-dc。四隅アーチファクト）
  undersampling.png   フル / R=5 / R=10（ゼロ詰めのボケ）
  slice_profile.png   3D薄スライス → 矩形プロファイル積分 → 2D厚スライス
  orientation.png     AX / COR / SAG リスライス
  profiles.png        rect vs gaussian プロファイル
  lowfield.png        高磁場 → 低磁場（ノイズ/解像度劣化）

使い方: python docs/make_figures.py
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as _fm  # noqa: E402

# 日本語ラベル用フォント（生成時に焼き込むので閲覧側はフォント不要）。
# 非Mac環境で再生成する場合は Noto Sans CJK JP 等に読み替える。
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in ("Hiragino Sans", "Noto Sans CJK JP", "Arial Unicode MS", "AppleGothic"):
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


# --------------------------- 合成ファントム ---------------------------
def ellipse(ny, nx, cy, cx, ry, rx, val, ang=0.0):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    yy -= cy; xx -= cx
    c, s = np.cos(ang), np.sin(ang)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    return np.where((xr / rx) ** 2 + (yr / ry) ** 2 <= 1, val, 0.0)


def brain2d(ny=160, nx=160):
    """簡易ブレイン風ファントム（同心楕円＋内部構造）。"""
    img = np.zeros((ny, nx))
    img += ellipse(ny, nx, ny / 2, nx / 2, ny * 0.42, nx * 0.34, 1.0)        # 頭蓋
    img -= ellipse(ny, nx, ny / 2, nx / 2, ny * 0.40, nx * 0.32, 0.55)       # 皮質と差
    img += ellipse(ny, nx, ny / 2, nx / 2, ny * 0.36, nx * 0.28, 0.55)       # 実質
    img += ellipse(ny, nx, ny * 0.46, nx * 0.40, ny * 0.10, nx * 0.05, 0.25, 0.3)  # 脳室
    img += ellipse(ny, nx, ny * 0.46, nx * 0.60, ny * 0.10, nx * 0.05, 0.25, -0.3)
    img += ellipse(ny, nx, ny * 0.60, nx * 0.50, ny * 0.06, nx * 0.10, -0.3, 0.0)   # 病変風
    return np.clip(img, 0, None)


def brain3d(nz=48, ny=140, nx=140):
    """z でゆっくり変化する 3D ブレイン風ボリューム（厚スライスの効果が見えるよう）。"""
    vol = np.zeros((nz, ny, nx))
    for z in range(nz):
        f = 0.6 + 0.4 * z / nz
        v = np.zeros((ny, nx))
        v += ellipse(ny, nx, ny / 2, nx / 2, ny * 0.42, nx * 0.34, 1.0)
        v -= ellipse(ny, nx, ny / 2, nx / 2, ny * 0.40, nx * 0.32, 0.55)
        v += ellipse(ny, nx, ny / 2, nx / 2, ny * 0.36, nx * 0.28, 0.55)
        # z に応じて移動・出現する小構造（through-plane の変化）
        cy = ny * (0.35 + 0.30 * z / nz)
        v += ellipse(ny, nx, cy, nx * 0.5, ny * 0.04, nx * 0.04, 0.5 * f)
        vol[z] = np.clip(v, 0, None)
    return vol


def coil_sens(ny, nx, n=4):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    s = []
    for i in range(n):
        cy = ny * (0.5 + 0.45 * np.cos(2 * np.pi * i / n))
        cx = nx * (0.5 + 0.45 * np.sin(2 * np.pi * i / n))
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        s.append(np.exp(-d2 / (2 * (0.6 * ny) ** 2)))
    return np.stack(s)


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)),
                                       axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)),
                                        axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def _panel(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# --------------------------- 図 ---------------------------
def fig_recon_pipeline():
    img = brain2d()
    sens = coil_sens(*img.shape, 4)
    coil_imgs = img[None] * sens
    ks = fft2c(coil_imgs)                      # (4,H,W) 中心DC k空間
    rss = np.sqrt((np.abs(ifft2c(ks)) ** 2).sum(0))
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
    _panel(ax[0], np.log(np.abs(ks[0]) + 1e-3), "1) k空間 (1ch, log|·|)", "magma")
    montage = np.concatenate([np.abs(ifft2c(ks[i])) for i in range(4)], axis=1)
    _panel(ax[1], montage, "2) コイル別 IFFT 画像 (4ch)")
    _panel(ax[2], rss, "3) コイル合成 RSS")
    _panel(ax[3], img, "(参照) 真画像")
    fig.suptitle("recon_motion: 中心化IFFT → コイル root-sum-of-squares", y=1.04, fontsize=12)
    save(fig, "recon_pipeline.png")


def fig_kspace_dc():
    img = brain2d()
    ks_corner = np.fft.fft2(img, norm="ortho")        # 端DC（標準FFT）
    rec_center = np.abs(ifft2c(ks_corner))            # 中心化IFFT → 四隅
    rec_corner = np.abs(np.fft.ifft2(ks_corner, norm="ortho"))  # 素のIFFT → 正
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    _panel(ax[0], rec_center, "--kspace-dc center (誤)\n→ 脳が四隅に分裂")
    _panel(ax[1], rec_corner, "--kspace-dc corner (正)\n→ 中央に再構成")
    fig.suptitle("端DC(Calgary-Campinas)を中心化IFFTすると四隅アーチファクト", y=1.04, fontsize=12)
    save(fig, "kspace_dc.png")


def fig_undersampling():
    img = brain2d()
    ks = fft2c(img)
    H = img.shape[0]

    def zerofill(R, acs=16):
        mask = np.zeros(H, bool)
        mask[::R] = True
        mask[H // 2 - acs // 2:H // 2 + acs // 2] = True
        k = ks.copy()
        k[~mask, :] = 0
        return np.abs(ifft2c(k)), mask.mean()

    r5, f5 = zerofill(5)
    r10, f10 = zerofill(10)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
    _panel(ax[0], img, "フルサンプリング (Train/Val)")
    _panel(ax[1], r5, f"Test-R=5 ゼロ詰め\n(ky {f5*100:.0f}%サンプル)")
    _panel(ax[2], r10, f"Test-R=10 ゼロ詰め\n(ky {f10*100:.0f}%サンプル)")
    fig.suptitle("アンダーサンプリング(R)が大きいほどゼロ詰め再構成はボケる", y=1.04, fontsize=12)
    save(fig, "undersampling.png")


def fig_slice_profile():
    vol = brain3d(nz=48)                 # 1mm 薄スライス
    # 矩形プロファイルで 5mm 厚（5枚平均）に
    c = 24
    thin = vol[c]
    thick = vol[c - 2:c + 3].mean(0)     # FWHM=5mm rect
    # 矢状断（YZ 断面）にプロファイル位置を重ねる
    sag = vol[:, :, vol.shape[2] // 2]   # (z, y)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(sag.T, cmap="gray", aspect="auto")
    ax[0].axhspan(0, sag.shape[1] - 1, xmin=(c - 2) / sag.shape[0], xmax=(c + 3) / sag.shape[0],
                  color="tab:red", alpha=0.3)
    ax[0].set_title("矢状断 + 5mm 厚スライス窓(赤)\n(縦=z, スライス方向)", fontsize=11)
    ax[0].axis("off")
    _panel(ax[1], thin, "薄スライス 1mm (1枚)")
    _panel(ax[2], thick, "厚スライス 5mm\n(矩形プロファイル積分)")
    fig.suptitle("mri_slice_sim: S(c)=∫P(z-c)M(z)dz（矩形プロファイルで隣接スライスを重み付き積分）",
                 y=1.05, fontsize=11)
    save(fig, "slice_profile.png")


def fig_orientation():
    vol = brain3d(nz=80, ny=120, nx=120)
    ax_img = vol[40]                       # axial (xy)
    cor_img = vol[:, 60, :]                # coronal (zx)
    sag_img = vol[:, :, 60]                # sagittal (zy)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    _panel(ax[0], ax_img, "axial（axis0 で積層）")
    _panel(ax[1], cor_img, "coronal（axis1 で積層）")
    _panel(ax[2], sag_img, "sagittal（axis2 で積層）")
    fig.suptitle("mri_slice_sim --orientation: 患者座標系で任意面へリスライスしてから厚スライス化",
                 y=1.04, fontsize=11)
    save(fig, "orientation.png")


def fig_profiles():
    z = np.linspace(-6, 6, 400)
    fwhm = 5.0
    rect = (np.abs(z) <= fwhm / 2).astype(float)
    sigma = fwhm / 2.3548
    gauss = np.exp(-z ** 2 / (2 * sigma ** 2))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(z, rect, label="rect (MRI 第一近似)", lw=2)
    ax[0].plot(z, gauss, label="gaussian (CT相当)", lw=2)
    ax[0].axvspan(-fwhm / 2, fwhm / 2, color="gray", alpha=0.12)
    ax[0].set_xlabel("スライス方向 z [mm]"); ax[0].set_ylabel("感度 P(z)")
    ax[0].set_title(f"スライスプロファイル (FWHM={fwhm}mm)", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # プロファイルで厚スライス化した違い
    vol = brain3d(nz=48)
    c = 24
    zc = np.arange(48) - c
    wr = (np.abs(zc) <= fwhm / 2).astype(float); wr /= wr.sum()
    wg = np.exp(-zc ** 2 / (2 * sigma ** 2)); wg /= wg.sum()
    tr = np.tensordot(wr, vol, axes=(0, 0))
    tg = np.tensordot(wg, vol, axes=(0, 0))
    _panel(ax[1], np.concatenate([tr, tg], axis=1), "厚スライス: rect | gaussian")
    fig.suptitle("--profile rect/gaussian の違い（矩形は裾を引かずシャープ）", y=1.04, fontsize=11)
    save(fig, "profiles.png")


def fig_lowfield():
    img = brain2d()
    from scipy.ndimage import gaussian_filter
    # 低磁場: 解像度低下(ボケ) + ノイズ増（Rician風）
    blur = gaussian_filter(img, 1.6)
    sig = blur + 0  # signal
    noise = RNG.normal(0, 0.06, img.shape) + 1j * RNG.normal(0, 0.06, img.shape)
    low = np.abs(blur + noise)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    _panel(ax[0], img, "高磁場 (3T) 入力")
    _panel(ax[1], low, "低磁場 (0.3-0.5T) 風\nノイズ↑ + 解像度↓")
    fig.suptitle("lowfield_sim: SNR低下(Rician) + 面内ボケ/k空間トランケーション", y=1.04, fontsize=12)
    save(fig, "lowfield.png")


def main():
    fig_recon_pipeline()
    fig_kspace_dc()
    fig_undersampling()
    fig_slice_profile()
    fig_orientation()
    fig_profiles()
    fig_lowfield()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
