#!/usr/bin/env python3
"""
inspect_h5.py
=============
fastMRI/ISMRMRD 形式 .h5 の **k空間の軸配置（レイアウト）** を調べる診断ツール。
recon_motion.py の出力形状がおかしい時（例: "256 slices 170x24"）に、各軸が
kx/ky/kz(partition)/coil のどれかを推定するために使う。

各 kspace 軸のサイズを ismrmrd_header の encodedSpace マトリクス(x,y,z) と
突き合わせて役割を推定し、推奨レイアウトと recon_motion のオプションを表示する。

使い方
------
    python inspect_h5.py <file.h5 か フォルダ>
"""
from __future__ import annotations

import glob
import os
import re
import sys

import h5py
import numpy as np


def _hdr_text(h):
    if "ismrmrd_header" not in h:
        return ""
    raw = h["ismrmrd_header"][()]
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def _space(hdr, name):
    """encodedSpace / reconSpace の matrixSize(x,y,z) と FOV(x,y,z) を取り出す。"""
    i = hdr.find(name)
    if i < 0:
        return None
    seg = hdr[i:i + 600]
    def g(tag, sub):
        m = re.search(rf"<[^>]*{tag}[^>]*>.*?<[^>]*{sub}[^>]*>([^<]+)<", seg, re.S)
        return float(m.group(1)) if m else None
    return {
        "mx": g("matrixSize", "x"), "my": g("matrixSize", "y"), "mz": g("matrixSize", "z"),
        "fx": g("fieldOfView_mm", "x"), "fy": g("fieldOfView_mm", "y"),
        "fz": g("fieldOfView_mm", "z"),
    }


def _limit_max(hdr, name):
    i = hdr.find(name)
    if i < 0:
        return None
    m = re.search(r"<[^>]*maximum[^>]*>([^<]+)<", hdr[i:i + 200])
    return int(m.group(1)) if m else None


def inspect(path: str) -> None:
    with h5py.File(path, "r") as h:
        print(f"=== {path} ===")
        print("keys:", list(h.keys()))
        ks = h["kspace"]
        shape = ks.shape
        ks_dtype = ks.dtype
        print(f"kspace: shape={shape}  ndim={ks.ndim}  dtype={ks_dtype}")
        for k, v in h.attrs.items():
            print(f"  attr {k} = {v}")
        hdr = _hdr_text(h)
        # サンプリング密度の診断（代表スライス1枚）→ アンダーサンプリング(R)の検出
        mid = tuple([shape[0] // 2] + [slice(None)] * (len(shape) - 1))
        try:
            sl = np.abs(np.asarray(ks[mid], dtype=np.float64))
        except Exception:  # noqa: BLE001
            sl = None

    enc = _space(hdr, "encodedSpace") or {}
    rec = _space(hdr, "reconSpace") or {}
    s1 = _limit_max(hdr, "kspace_encoding_step_1")
    s2 = _limit_max(hdr, "kspace_encoding_step_2")
    pos = re.search(r"<[^>]*patientPosition[^>]*>([^<]+)<", hdr)
    print("header:")
    print(f"  encodedSpace matrix = ({enc.get('mx')},{enc.get('my')},{enc.get('mz')})  "
          f"FOV = ({enc.get('fx')},{enc.get('fy')},{enc.get('fz')})")
    print(f"  reconSpace   matrix = ({rec.get('mx')},{rec.get('my')},{rec.get('mz')})  "
          f"FOV = ({rec.get('fx')},{rec.get('fy')},{rec.get('fz')})")
    print(f"  kspace_encoding_step_1 max = {s1}   step_2(partition) max = {s2}")
    print(f"  patientPosition = {pos.group(1) if pos else '?'}")
    is3d = bool((enc.get("mz") or 1) > 1 or (s2 or 0) > 0)
    print(f"  => 3D 取得か: {is3d}")

    # --- アンダーサンプリング(R)の検出 ---
    if sl is not None and sl.size:
        print("サンプリング密度（代表スライスの各軸、エネルギーのある面の割合）:")
        thr = sl.max() * 1e-6
        under = False
        for ax in range(sl.ndim):
            others = tuple(i for i in range(sl.ndim) if i != ax)
            e = sl.sum(axis=others)
            frac = float((e > thr).sum()) / len(e)
            note = ""
            if frac < 0.95:
                note = f"  ← 間引きあり！ 加速 R≈{1.0/max(frac,1e-9):.1f}（この軸が位相エンコード）"
                under = True
            print(f"  axis {ax} (size={len(e)}): サンプル率={frac*100:.0f}%{note}")
        if under:
            print("  => アンダーサンプリング・データ（Test-R=N）。ゼロ詰めIFFTは原理的にボケる。")
            print("     鮮明な像にはフルサンプリング(Train/Val)か、PI/CS/DL再構成が必要。")
        else:
            print("  => フルサンプリングのよう（ボケる場合は解像度/補間の問題を別途確認）。")

    # --- 各軸の役割推定（サイズ突き合わせ） ---
    ex, ey, ez = enc.get("mx"), enc.get("my"), enc.get("mz")
    # ISMRMRD: ky の取得行数は step_1 max+1 のことがある（部分フーリエ/オーバーサンプル）
    cand = {
        "kx (readout, =enc.x)": ex,
        "ky (phase, =enc.y)": ey,
        "kz (partition, =enc.z)": ez if (ez or 1) > 1 else None,
        "ky' (=step_1 max+1)": (s1 + 1) if s1 is not None else None,
        "kz' (=step_2 max+1)": (s2 + 1) if (s2 or 0) > 0 else None,
    }
    print("軸の推定（kspace 各軸サイズ → ヘッダの一致候補）:")
    for ax, n in enumerate(shape):
        hits = [name for name, val in cand.items() if val and abs(val - n) <= 1]
        tag = "  ".join(hits) if hits else "（一致なし → おそらく coil か oversampled readout）"
        print(f"  axis {ax}: size={n:>5}  -> {tag}")

    # --- 実数格納（実/虚インターリーブ）の検出 ---
    is_complex = np.issubdtype(np.dtype(ks_dtype), np.complexfloating)
    real_stored = not is_complex
    if real_stored:
        print("注意: dtype が実数 → 複素が **実/虚インターリーブ** で格納されている可能性大。")
        print(f"      最後の軸 size={shape[-1]} が偶数なら 2×コイル（実/虚ペア）。"
              f" → recon_motion に --real-imag-axis -1 を付けて複素化する。")

    # --- 推奨レイアウト ---
    print("推奨:")
    no_header = (enc.get("mx") is None)
    cc_like = (real_stored and len(shape) == 4 and shape[-1] % 2 == 0 and no_header)
    if cc_like:
        ncoil = shape[-1] // 2
        print("  → Calgary-Campinas 形式の可能性大（ヘッダ無し・実数・最後の軸=2×コイル）。")
        print(f"     軸構成の想定: (スライス={shape[0]}, ky={shape[1]}, kx={shape[2]}, "
              f"coil×2={shape[-1]}→{ncoil}coil)。第1軸は既に画像領域（CCチャレンジ規約）。")
        print("     コマンド例:")
        print("       python recon_motion.py --in-root <dir> --out-root cc_out \\")
        print("           --real-imag-axis -1 --transpose 0,3,1,2 --kspace-dc corner \\")
        print("           --pixel-spacing 1 --slice-spacing 1 --slice-thickness 1 --format png")
        print("       # 複素化(24->12coil) → (slice,coil,ky,kx) → 端DCの2D再構成")
        print("     ※ CC は DC が端[0,0]（標準FFT配置）。--kspace-dc corner 必須")
        print("        （無いと像が四隅に分裂）。ヘッダ無しなので voxel間隔は --*-spacing で指定")
        print("        （CCは概ね1mm等方。実値に合わせて調整）。向きは既定で補正なし。")
        print("        上下/スライスが逆なら --flip-y on / --reverse-slices on で調整。")
    else:
        print("  recon_motion は標準レイアウト (kz, coil, ky, kx) = 軸(0,1,2,3) を想定。")
        print("  上の推定で kx/ky が最後の2軸、partition(kz) が軸0 になっていない場合は、")
        print("  --transpose で並べ替える（面内 ky,kx を最後の2軸、coil を軸1 へ）。")
        print("  ※ coil 軸は通常ヘッダ matrix と一致しない軸（受信チャネル数）。")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    files = ([target] if target.endswith(".h5")
             else sorted(glob.glob(os.path.join(target, "**", "*.h5"), recursive=True)))
    if not files:
        print("no .h5 found")
        sys.exit(1)
    inspect(files[0])           # 先頭1ファイルを診断
    if len(files) > 1:
        print(f"\n（他 {len(files) - 1} ファイルあり。先頭のみ表示）")


if __name__ == "__main__":
    main()
