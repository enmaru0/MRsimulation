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
| **3D取得** | `(partitions, coils, H, W)` complex64 | **3D IFFT**（kz も復元）→RSS→reconSpace へクロップ | `encodedSpace.z`>1 / `kspace_encoding_step_2`>0 で自動判定 |

各形式とも `ismrmrd_header`（FOV・厚・間隔・TR/TE/TI・FA・磁場強度・UID、3D は encoded/recon の z）と
attrs（`acquisition`, `patient_id`, brain は `max`）を持つ。

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

### 真の 3D 取得（パーティション方向も k空間エンコード）

2Dマルチスライス（各スライス独立）と違い、**真の 3D 取得**ではスライス（パーティション=kz）方向も
k空間エンコードされる。この場合は**面内だけでなく kz 方向にも IFFT**が必要（＝3D IFFT）。
形状だけでは 2D と区別できないため、`ismrmrd_header` の **`encodedSpace.matrixSize.z`>1** または
**`kspace_encoding_step_2.maximum`>0** で自動判定する（`--recon-3d auto`、既定）。

```
imgs = ifftc(ifft2c(kspace), axis=partition)       # kz も含めた 3D IFFT
mag  = sqrt( Σ_c |imgs_c|² )                        # コイル RSS
mag  = crop(mag, reconSpace.z)                      # パーティションのオーバーサンプル除去
```

- 既定の k空間レイアウトは **`(kz, coils, ky, kx)`**（軸0=パーティション、最後の2軸=面内）。
- **軸配置の自動判定**: 3D時は `encodedSpace` の (x,y,z) と各軸サイズを突き合わせて、
  `kx=enc.x` `ky=enc.y` `kz=enc.z` 残り=coil を推定し、標準順へ自動で並べ替える。
- 自動判定が外れる/出力形状がおかしい時（例 `256 slices 170x24`）は、まず **`inspect_h5.py`**
  で各軸の役割を確認し、**`--transpose`** で軸順を明示する。

```bash
# 1) 軸配置を確認（各 kspace 軸が kx/ky/kz/coil のどれか）
python inspect_h5.py <3D_h5_dir>

# 2) 例: 実データが (ky, kx, kz, coil) なら 標準(kz,coil,ky,kx) へ並べ替え
python recon_motion.py --in-root <dir> --out-root out --transpose 2,3,0,1
```

- `--recon-3d on`/`off` で強制/無効化、`--part-axis` でパーティション軸を指定。
- 合成3Dファントムのラウンドトリップで検証済み（標準/非標準レイアウトとも相対誤差 ~1e-7）。
  2D 取得は自動で従来どおり面内のみ IFFT（回帰確認済み）。

### 実/虚インターリーブ格納（Calgary-Campinas など）

`dtype` が **実数(float)** の k空間は、複素が **実/虚インターリーブ**（`[r0,i0,r1,i1,...]`）で
格納されていることが多い。`--real-imag-axis N` でその軸を複素化する（軸サイズ `2C → C` コイル）。

**Calgary-Campinas（CC-359）3D脳・12ch** は典型例: `kspace shape=(256, 218, 170, 24)` float32・
ヘッダ無し。`24 = 12コイル×2(実/虚)`、軸構成 `(スライス, ky, kx, coil×2)`、第1軸は既に画像領域。

```bash
# inspect で形式を確認（CC形式なら推奨コマンドも表示）
python inspect_h5.py Calgary-campinas/test

# 複素化(24→12coil) → (slice,coil,ky,kx) へ並べ替え → 2D再構成
python recon_motion.py --in-root Calgary-campinas/test --out-root cc_out \
    --real-imag-axis -1 --transpose 0,3,1,2 --format png
```

`--real-imag-axis` は `--transpose` より先に適用される（複素化後の軸番号で transpose を指定）。
合成CCデータのラウンドトリップで検証済み（相対誤差 ~1e-7）。
もし第1軸が画像領域でなく k空間（真の3D）なら `--recon-3d on --part-axis 0` を追加して比較する。

> このリポジトリの既存データ（brain/gre/knee）はすべて 2D マルチスライス取得。3D データは
> このPCには無いが、上記の自動判定＋3D IFFT で再構成できるよう実装・検証してある。

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

## 2D 厚スライスのシミュレーション（`--slab`）

薄スライス積層の3Dボリュームから、**擬似 2D 厚スライス**を再構成する。`--slab N` は
**連続 N 枚を矩形(rect)スライスプロファイルで合成**（等重み平均、総和1で正規化）して
1枚の厚スライスにする。3D薄スライス→2D厚スライスの簡易版で、
[`mri_slice_sim.py`](slice-simulation.md) の方針（magnitude・同コントラスト・矩形プロファイル）に準拠。

| オプション | 既定 | 説明 |
|---|---|---|
| `--slab N` | `1`（無し） | N枚を合成して厚2Dスライス化（厚み≈N×元厚） |
| `--slab-step M` | `=N` | スラブの送り。`M<N` で重なりスラブ |

```bash
# 38枚の薄スライス → 5枚ずつ rect 合成して厚2Dスライス（≈7枚）に
python recon_motion.py --in-root knee_multicoil_test --out-root knee_2dsim --slab 5

# 重なりスラブ（N=5, step=2）でスライス数を保ちつつ厚みだけ増やす
python recon_motion.py --in-root knee_multicoil_test --out-root knee_2dsim_ov --slab 5 --slab-step 2
```

出力 DICOM/binary の `SliceThickness`/`SpacingBetweenSlices` は合成枚数ぶん厚く更新される。
スライス方向に平均化されるため、through-plane の構造はなだらかになり厚スライス2D像に近づく。

> 注: これは**再構成後の magnitude を空間領域で合成**する簡易法。真の 2D 励起（厚いRFスラブ）や
> 部分容積での位相打ち消しは扱わない（[制約](slice-simulation.md#制約と注意) は mri_slice_sim と同様）。
> なお本データ(`knee_multicoil_test`)は 2D マルチスライス取得＋8倍アンダーサンプリングのため、
> 元の薄スライスにエイリアスが残る。

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
- **画素**: 格納整数 = `round(magnitude / RescaleSlope)`（控えめなスケール、~1000-3000台）。
  `RescaleSlope` をボリュームごとに設定するので `pixel_array * RescaleSlope` で
  **元の magnitude を正確に復元可能**。`WindowCenter/Width` も rescale 後（magnitude）単位で設定。
  振幅がデータセットで桁違い（ブレイン ~1e2、単コイル膝 ~1e-4）でも自動で適正スケールに収める。
  派生画像であることを `ImageType = [DERIVED, SECONDARY]` で明示
- **向き**: ボリュームは出力前に [向き補正](#向きpatientposition由来) 済み。`PatientPosition`
  タグにはヘッダ値（HFS/FFS 等）を格納する。

### binary（`.raw` / `.hdr` / `.tag`）

3Dボリュームを **2byte 直接バイナリ**で保存する。1 h5 = 同名3ファイル。

- **`.raw`** — **`int16`（符号付き short）** リトルエンディアン。並びは **x が最速 → y → z**
  （C順, shape `(z,y,x)`）。画素値 = `round(magnitude / rescale_slope)`（控えめなスケール、
  ~1000-3000台。従来の32000は過大だった）。`rescale_slope` は10のべき乗で `.tag` に記載し、
  `magnitude = stored * rescale_slope` で**実信号を正確に復元**できる。値は int16範囲(±32767)内に
  収まるので、2byteを符号付きshortで読むビューアでも**オーバーフローしない**。
- **向き** — 出力前に [向き補正](#向きpatientposition由来) 済み（raw/DICOM/PNG で一貫）。
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
magnitude = vol * rescale_slope            # rescale_slope は .tag / summary.csv に記載
```

## 向き（patientPosition由来）

スライス方向や上下が**入力データによって反転**するのは、データセットで `patientPosition` が
異なるため（例: ブレイン系は **HFS**=Head First Supine、単コイル膝は **FFS**=Feet First Supine。
HFS と FFS では S-I（スライス）と L-R が逆になる）。

> **重要**: この簡易 fastMRI 形式（`.h5`）には**スライスごとの
> `ImageOrientationPatient` / `ImagePositionPatient` は含まれていない**（フル ISMRMRD の
> acquisition ヘッダにあるが、ここには非エクスポート）。参照できる向き情報は
> `ismrmrd_header` の **`patientPosition`** と FOV/マトリクスのみ。そこで patientPosition から
> スライス積層方向を決め、**raw / DICOM / PNG すべてで同じ向き**に揃える。

自動（`auto`）の既定:

| 補正 | auto の挙動 |
|---|---|
| `--flip-y`（行=上下） | **ON**（撮像面に依らず supine なら anterior が上に揃う） |
| `--reverse-slices`（スライス積層方向） | **patientPosition 由来**: Head-First→ON / Feet-First→OFF |
| `--flip-x`（列=左右） | **OFF** |

各フラグは `auto` / `on` / `off` で明示上書きできる（例: 特定ビューアの慣習に合わせて
`--reverse-slices off`）。適用した向きは `.tag` の `orientation` 行と `summary.csv`
（`patient_position` / `flip_x` / `flip_y` / `reverse_slices` 列）に記録される。

```bash
# patientPosition に従って自動で向きを揃える（既定）
python recon_motion.py --in-root singlecoil_test --out-root out --format all
# スライス方向を強制的に反転しない
python recon_motion.py --in-root gre_data --out-root out --reverse-slices off
```

## summary.csv（症例ごとのまとめ）

変換のたび、出力ルート直下に **`summary.csv`** を生成する（**1症例＝1行**）。
各 h5 のタグ・ジオメトリ・シーケンス情報と、今回の変換設定をまとめる。主な列:

| 列 | 内容 |
|---|---|
| `file` / `basename` | 入力 h5 の相対パス / ベース名 |
| `acquisition` / `patient_id` / `patient_position` | コントラスト種別 / 患者ID(ハッシュ) / HFS・FFS等 |
| `n_slices` / `nx,ny,nz` | スライス数 / 出力次元 |
| `dx_mm,dy_mm,dz_mm` / `slice_thickness_mm` / `spacing_mm` | 物理スペーシング・厚み |
| `field_strength_T` / `TR_ms,TE_ms,TI_ms,flip_deg` | 磁場強度・シーケンスパラメータ |
| `protocol` / `sequence` / `manufacturer` / `model` / `institution` | 撮像プロトコル・装置 |
| `intensity_max` / `rescale_slope` / `stored_max` | 信号最大・rescale係数・格納整数の最大 |
| `formats` / `acq_matrix` / `lowfield_snr` | 出力形式 / 低磁場の取得マトリクス・SNR |
| `flip_x` / `flip_y` / `reverse_slices` | 適用した向き補正 |
| `series_uid` / `study_uid` | UID |

`magnitude = stored_value * rescale_slope` で実信号を復元できる（列 `rescale_slope`）。

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
| `--flip-y` | `auto` | 行(上下)反転 auto(=ON)/on/off。raw/DICOM/PNG 共通 |
| `--flip-x` | `auto` | 列(左右)反転 auto(=OFF)/on/off |
| `--reverse-slices` | `auto` | スライス方向反転 auto(=patientPosition由来)/on/off |
| `--slab` | `1` | N枚を rect 合成して厚2Dスライス化（2Dシミュレーション） |
| `--slab-step` | `=N` | スラブの送り（`<N` で重なり） |
| `--recon-3d` | `auto` | 3D取得の再構成 auto(=ヘッダ判定)/on(強制3D)/off(面内のみ) |
| `--part-axis` | `0` | 3D時のパーティション(kz)軸（`(kz,coil,ky,kx)`前提） |

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
