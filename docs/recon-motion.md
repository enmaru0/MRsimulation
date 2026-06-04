[← MRsimulation トップ](../README.md)

# k-space からの再構成 (`recon_motion.py`)

fastMRI 形式の k-space（`.h5`）を画像に再構成し、**PNG / DICOM / binary(.raw)** で
出力するツール。**マルチコイル(brain)** と **単コイル(knee)** の両形式に対応し、
`motion/`・`multicoil_test/`・`gre_data/`・`singlecoil_test/` などに使える。

## 入力データ形式

| データ | `kspace` | 再構成 | 備考 |
|---|---|---|---|
| **マルチコイル(brain)** | `(slices, coils, H, W)` complex64 | コイル RSS、出力=encoded サイズ | `reconstruction_rss`(正解) 同梱 |
| **単コイル(knee)** | `(slices, H, W)` complex64 | `|IFFT|`、reconSpace へ中央クロップ | `mask` でアンダーサンプリング(ゼロ詰め再構成) |

両形式とも `ismrmrd_header`（FOV・厚・間隔・TR/TE/TI・FA・磁場強度・UID）と attrs
（`acquisition`, `patient_id`, brain は `max`）を持つ。

## 再構成アルゴリズム

**中心化 2D 逆FFT**（`ifftshift → ifft2(norm="ortho") → fftshift`）したのち、
マルチコイルは **root-sum-of-squares (RSS)**、単コイルは絶対値で magnitude を作る:

```
img_c = ifft2c(kspace_c)                       # コイルごとの複素画像
mag   = sqrt( Σ_c |img_c|² )  (multicoil)      # コイル合成 magnitude
mag   = |ifft2c(kspace)|       (single coil)
```

最後に `ismrmrd_header` の **reconSpace マトリクスへ中央クロップ**する（例 単コイル膝
640×368 → 320×320。マルチコイル brain は encoded=recon=256 でクロップ無し）。

マルチコイルは fastMRI 標準手法で、付属の `reconstruction_rss` を**厳密に再現**する
（検証で相対誤差 ~1e-7）。モーションアーチファクト（位相エンコード方向のゴースト/ブレ）は
そのまま画像に現れるので、モーション補正アルゴリズムの評価データにもなる。
単コイル膝はアンダーサンプリング(acceleration)済みなので、ゼロ詰め再構成にエイリアスが残る。

> 補足: numpy の既定 `ifft2`（norm=`backward`, 1/N 正規化）だと振幅が √(H·W) 倍ずれる。
> fastMRI と一致させるには **`norm="ortho"` が必須**。

## 低磁場シミュレーション（再構成前・k空間ドメイン）

再構成前に k空間で劣化を与えて「低磁場風」の画像を再構成できる。**出力画像の画素数
（解像度グリッド）は変えず**、実効解像度／SNR だけを落とす。

| オプション | 処理 | 効果 |
|---|---|---|
| `--acq-matrix N`（または `R,C`） | k空間中央を N×N にクロップ→元グリッドへゼロ詰め | **取得マトリクスを N に低下**＝解像度ダウン。出力画素数は不変 |
| `--lowfield-snr S` | 複素ガウシアンノイズを付加（RSS後 Rician） | 目標 SNR≈S までノイズを増やす |
| `--seed` | ノイズ乱数シード | データ拡張 |

`--acq-matrix 192` は「取得は192×192だが再構成は元グリッド」のゼロフィル再構成に相当する
（FOV・PixelSpacing は不変、高周波が無くなりボケる）。実低磁場が低マトリクス取得＋
ゼロフィル再構成である状況を、最も素直に k空間で再現する。

```bash
# 取得192×192に落として再構成（出力の画素数は不変）
python recon_motion.py --in-root singlecoil_test --out-root singlecoil_lf192 --acq-matrix 192

# さらに低磁場ノイズも付加（目標SNR=8）
python recon_motion.py --in-root singlecoil_test --out-root singlecoil_lf \
    --acq-matrix 192 --lowfield-snr 8 --seed 0
```

> `--lowfield-snr` のノイズは**単コイルで厳密**（magnitude Rician）、マルチコイル RSS では
> コイル結合の都合で **1/√C の一次近似**（おおよそ目標SNRに合うが厳密でない）。
> 厳密にノイズ/コントラストを実低磁場へ合わせるなら、再構成画像(DICOM)を
> [`lowfield_sim.py`](lowfield.md) に通す方法もある。

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
- **向き補正（既定ON）**: 行(y)を反転（上下を合わせる）し、スライス積層方向（InstanceNumber /
  `ImagePositionPatient` の z 向き）を反転する。ビューアで上下・スライス順が逆になる問題を解消。
  `--dicom-no-flip-y` / `--dicom-no-reverse-slices` で各々無効化できる。

### binary（`.raw` / `.hdr` / `.tag`）

3Dボリュームを **2byte 直接バイナリ**で保存する。1 h5 = 同名3ファイル。

- **`.raw`** — **`int16`（符号付き short）** リトルエンディアン。並びは **x が最速 → y → z**
  （C順, shape `(z,y,x)`）。画素値は magnitude を **max→32000** にスケールした整数
  （`.hdr` の「2byte」を符号付き short として読むビューアで**オーバーフローしない**よう、
  int16範囲 ±32767 内に収める）。スケールは各ボリュームの実最大値基準なので、
  低磁場処理(acq-matrix/ノイズ)で値が増えても飽和しない。
  `magnitude = stored * rescale_slope`（`rescale_slope` は `.tag` に記載）。
- **向き** — 多くの `.raw` ビューアは行原点が下(bottom-up)なので、**既定で y(行) を反転**して
  上下が正しく表示されるようにしている（`--raw-no-flip-y` で無効化可）。
- **`.hdr`** — 1行テキスト、半角スペース区切り（末尾スペースなし）:

  ```
  Xサイズ Yサイズ Zサイズ 2 X物理スペーシング(mm) Y物理スペーシング(mm) Z物理スペーシング(mm)
  ```

  例: `256 256 18 2 0.9375 0.9375 6`（`2` は1ボクセル2byteの意）。
- **`.tag`** — 各種メタ情報を `key: value` で記録（acquisition, patient_id, dims_xyz,
  voxel_spacing_mm_xyz, slice_thickness_mm, spacing_between_slices_mm, data_type(int16), byte_order,
  voxel_order(y反転の有無), intensity_max, stored_max, rescale_slope, field_strength_T,
  TR/TE/TI/flip, protocol, sequence, manufacturer, model, institution, 各種UID 等）。

読み込み例（Python）:

```python
import numpy as np
nx, ny, nz, bpp, dx, dy, dz = open("vol.hdr").read().split()
vol = np.fromfile("vol.raw", dtype="<i2").reshape(int(nz), int(ny), int(nx))  # int16, (z,y,x)
```

## 使い方

```bash
# PNG（既定）— 全ファイル
python recon_motion.py --in-root motion          --out-root motion_png
python recon_motion.py --in-root multicoil_test  --out-root multicoil_test_png
python recon_motion.py --in-root gre_data        --out-root gre_data_png
python recon_motion.py --in-root singlecoil_test --out-root singlecoil_test_png   # 単コイル膝

# DICOM 出力
python recon_motion.py --in-root gre_data --out-root gre_data_dicom --format dicom

# binary(.raw/.hdr/.tag) 出力
python recon_motion.py --in-root gre_data --out-root gre_data_raw --format binary

# 低磁場: 取得192×192へ落として再構成（出力画素数は不変）
python recon_motion.py --in-root singlecoil_test --out-root singlecoil_lf192 --acq-matrix 192

# 全形式、先頭2ファイルだけ動作確認
python recon_motion.py --in-root motion --out-root out_check --format all --limit 2
```

| オプション | 既定 | 説明 |
|---|---|---|
| `--in-root` | `motion` | 入力 `.h5` ルート（再帰的に探索） |
| `--out-root` | `motion_png` | 出力ルート |
| `--format` | `png` | `png` / `dicom` / `binary` / `both`(=png+dicom) / `all` |
| `--limit` | `0` | 先頭 N ファイルのみ（0=全部、動作確認用） |
| `--acq-matrix` | — | 低磁場: 取得マトリクスを N か R,C へ（k空間中央クロップ＋ゼロ詰め、出力画素数不変） |
| `--lowfield-snr` | — | 低磁場: 目標SNRまでノイズ付加（単コイル厳密/マルチコイル近似） |
| `--seed` | `0` | `--lowfield-snr` のノイズ乱数シード |
| `--raw-no-flip-y` | off | binary(.raw)の行(y)反転をしない（既定は反転して上下を合わせる） |
| `--dicom-no-flip-y` | off | DICOMの行(y)反転をしない（既定は反転して上下を合わせる） |
| `--dicom-no-reverse-slices` | off | DICOMのスライス方向反転をしない（既定は反転して積層方向を合わせる） |

## 注意

- k-space（`.h5`）と再構成画像（PNG/DICOM/raw）は **患者データ**のため git に push しない
  （`.gitignore` で `motion*/`・`multicoil_test*/`・`gre_data*/`・`singlecoil_test*/`・
  `*.h5`・`*.raw`・`*.hdr`・`*.tag` を除外済み）。
- DICOM/binary の絶対位置（IPP/原点）は元の患者座標が不明なため **FOV 中心を原点**として再構成している。
  同一ボリューム内のスライス間隔・厚み・面内スケールは header の実値で正しい。
- 必要パッケージ: `h5py`, `Pillow`（PNG）, `pydicom`（DICOM）。binary は標準ライブラリのみ。

---

### 関連ページ

- [高磁場→低磁場シミュレーション (`lowfield_sim.py`)](lowfield.md) — 再構成画像を学習データへ
- [3D薄スライス → 2D厚スライス (`mri_slice_sim.py`)](slice-simulation.md)
- [← トップへ戻る](../README.md)
