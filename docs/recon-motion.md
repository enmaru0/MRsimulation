[← MRsimulation トップ](../README.md)

# k-space からの再構成 (`recon_motion.py`)

fastMRI 形式のマルチコイル k-space（`.h5`）を画像に再構成し、**PNG / DICOM / binary(.raw)** で
出力するツール。`motion/`・`multicoil_test/`・`gre_data/` など、同形式のフォルダすべてに使える。

## 入力データ形式（fastMRI brain multicoil）

各 `.h5` ファイルは以下を持つ:

| キー | 内容 |
|---|---|
| `kspace` | `(slices, coils, H, W)` complex64 の生 k-space |
| `reconstruction_rss` | `(slices, H, W)` float32 の正解再構成（検証用） |
| `ismrmrd_header` | 撮像ジオメトリ/シーケンス情報（FOV・厚・間隔・TR/TE/TI・FA・磁場強度・各種UID） |
| attrs | `acquisition`（AXT1/AXT2/AXFLAIR/AXGRE 等）, `max`, `patient_id` |

## 再構成アルゴリズム

各コイルを **中心化 2D 逆FFT**（`ifftshift → ifft2(norm="ortho") → fftshift`）し、
コイル方向に **root-sum-of-squares (RSS)** で合成する:

```
img_c = ifft2c(kspace_c)                       # コイルごとの複素画像
rss   = sqrt( Σ_c |img_c|² )                   # コイル合成 magnitude
```

これは fastMRI 標準手法で、付属の `reconstruction_rss` を**厳密に再現**する
（検証で相対誤差 ~1e-7）。モーションアーチファクト（位相エンコード方向のゴースト/ブレ）は
そのまま画像に現れるので、モーション補正アルゴリズムの評価データにもなる。

> 補足: numpy の既定 `ifft2`（norm=`backward`, 1/N 正規化）だと振幅が √(H·W)=256 倍ずれる。
> fastMRI と一致させるには **`norm="ortho"` が必須**。

## 出力形式（`--format`）

| 値 | 出力 |
|---|---|
| `png`（既定） | `<out_root>/<basename>/sl00.png …` 8bit グレースケール。ボリューム最大値（`attrs['max']`）で正規化 |
| `dicom` | `<out_root>/<basename>/sl00.dcm …` MR Image Storage。ヘッダのジオメトリ＋シーケンス情報を埋め込む |
| `binary` | `<out_root>/<basename>.raw` + `.hdr` + `.tag`（3Dボリュームを2byte生バイナリ） |
| `both` | png + dicom |
| `all` | png + dicom + binary |

png / dicom は**スライス画像**なので各 h5 ごとにサブフォルダ `<out_root>/<basename>/slNN.*` に出す。
binary は**1ボリューム3点セット** `<out_root>/<basename>.raw/.hdr/.tag` を出す。
`motion/` のように `inter-scan_motion/…` のサブフォルダがある入力では、その階層を保ったまま出力する。

### DICOM に埋め込む情報

`ismrmrd_header` から実値を取り出して格納する（1 h5 = 1 シリーズ）:

- **ジオメトリ**: `PixelSpacing`（FOV/マトリクス＝0.9375mm 等）, `SliceThickness`, `SpacingBetweenSlices`,
  `ImageOrientationPatient`（軸位 HFS = `[1,0,0,0,1,0]`）, `ImagePositionPatient`（FOV中心基準でスライス積層）
- **シーケンス**: `MagneticFieldStrength`(0.3T), `RepetitionTime`, `EchoTime`, `InversionTime`, `FlipAngle`,
  `Manufacturer`/`ManufacturerModelName`, `ProtocolName`, `SequenceName`
- **UID**: ヘッダの `seriesInstanceUID`/`studyUID`/`frameOfReferenceUID` を使用（妥当でなければ自動生成）、
  `SOPInstanceUID` はスライスごとに採番
- **画素**: uint16（max→約65000 にスケール）。`RescaleSlope` を設定するので
  `pixel_array * RescaleSlope` で**元の magnitude を復元可能**（量子化誤差 ~1e-5）。
  派生画像であることを `ImageType = [DERIVED, SECONDARY]` で明示

### binary（`.raw` / `.hdr` / `.tag`）

3Dボリュームを **2byte 直接バイナリ**で保存する。1 h5 = 同名3ファイル。

- **`.raw`** — `uint16` リトルエンディアン。並びは **x が最速 → y → z**（C順, shape `(z,y,x)`）。
  画素値は magnitude を max→約65000 にスケールした整数。
  `magnitude = stored * rescale_slope`（`rescale_slope` は `.tag` に記載）。
- **`.hdr`** — 1行テキスト、半角スペース区切り（末尾にもスペース1個）:

  ```
  Xサイズ Yサイズ Zサイズ 2 X物理スペーシング(mm) Y物理スペーシング(mm) Z物理スペーシング(mm)
  ```

  例: `256 256 18 2 0.9375 0.9375 6 `（`2` は1ボクセル2byteの意）。
- **`.tag`** — 各種メタ情報を `key: value` で記録（acquisition, patient_id, dims_xyz,
  voxel_spacing_mm_xyz, slice_thickness_mm, spacing_between_slices_mm, data_type, byte_order,
  voxel_order, intensity_max, rescale_slope, field_strength_T, TR/TE/TI/flip, protocol, sequence,
  manufacturer, model, institution, 各種UID 等）。

読み込み例（Python）:

```python
import numpy as np
nx, ny, nz, bpp, dx, dy, dz = open("vol.hdr").read().split()
vol = np.fromfile("vol.raw", dtype="<u2").reshape(int(nz), int(ny), int(nx))  # (z,y,x)
```

## 使い方

```bash
# PNG（既定）— 全ファイル
python recon_motion.py --in-root motion         --out-root motion_png
python recon_motion.py --in-root multicoil_test --out-root multicoil_test_png
python recon_motion.py --in-root gre_data       --out-root gre_data_png

# DICOM 出力
python recon_motion.py --in-root gre_data --out-root gre_data_dicom --format dicom

# binary(.raw/.hdr/.tag) 出力
python recon_motion.py --in-root gre_data --out-root gre_data_raw --format binary

# 全形式、先頭2ファイルだけ動作確認
python recon_motion.py --in-root motion --out-root out_check --format all --limit 2
```

| オプション | 既定 | 説明 |
|---|---|---|
| `--in-root` | `motion` | 入力 `.h5` ルート（再帰的に探索） |
| `--out-root` | `motion_png` | 出力ルート |
| `--format` | `png` | `png` / `dicom` / `binary` / `both`(=png+dicom) / `all` |
| `--limit` | `0` | 先頭 N ファイルのみ（0=全部、動作確認用） |

## 注意

- k-space（`.h5`）と再構成画像（PNG/DICOM/raw）は **患者データ**のため git に push しない
  （`.gitignore` で `motion*/`・`multicoil_test*/`・`gre_data*/`・`*.h5` を除外済み）。
- DICOM/binary の絶対位置（IPP/原点）は元の患者座標が不明なため **FOV 中心を原点**として再構成している。
  同一ボリューム内のスライス間隔・厚み・面内スケールは header の実値で正しい。
- 必要パッケージ: `h5py`, `Pillow`（PNG）, `pydicom`（DICOM）。binary は標準ライブラリのみ。

---

### 関連ページ

- [高磁場→低磁場シミュレーション (`lowfield_sim.py`)](lowfield.md) — 再構成画像を学習データへ
- [3D薄スライス → 2D厚スライス (`mri_slice_sim.py`)](slice-simulation.md)
- [← トップへ戻る](../README.md)
