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
        print(f"kspace: shape={shape}  ndim={ks.ndim}  dtype={ks.dtype}")
        for k, v in h.attrs.items():
            print(f"  attr {k} = {v}")
        hdr = _hdr_text(h)

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

    # --- 推奨レイアウト ---
    print("推奨:")
    print("  recon_motion は標準レイアウト (kz, coil, ky, kx) = 軸(0,1,2,3) を想定。")
    print("  上の推定で kx/ky が最後の2軸、partition(kz) が軸0 になっていない場合は、")
    print("  実データをその順へ並べ替える必要がある（--part-axis でパーティション軸を指定可能。")
    print("  面内 ky,kx が最後の2軸でない場合は要相談：軸順を教えてください）。")
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
