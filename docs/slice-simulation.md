[← MRsimulation トップ](../README.md)

# 3D薄スライス → 2D厚スライス (`mri_slice_sim.py`)

1mmスライス厚などの3D薄スライスMRIから、擬似的な **5mm厚 / 6mm間隔** などの2D厚スライス画像を、
MRIの原理（スライス感度プロファイル積分）に基づいて合成する。

> 背景（なぜCTの「ガウシアン平滑化＋ダウンサンプル」では不正確か）は
> [トップの解説](../README.md#背景-なぜctの方法そのままではダメか)を参照。

---

## 本ツールが対象とするケース

| 項目 | 設定 | 意味 |
|---|---|---|
| 入力データ | **magnitude のみ** | $M(z)$ ≈ 入力画素値 とみなす。位相打ち消しは再現しない |
| 出力コントラスト | **入力と同一** (TR/TE据え置き) | 信号方程式の再計算は不要 |
| スライスプロファイル | **矩形** (FWHM = スライス厚) | MRIはガウシアンでなく矩形が第一近似 |

この条件では処理は **「スライス方向に矩形プロファイルで重み付き加算 → 指定間隔で再配置」** に帰着する。
CTのガウシアン版との違いは次の2点だけ。

1. カーネル形状が **ガウシアン → 矩形**
2. 強度スケールを保つため重みを **総和1に正規化**

### アルゴリズム

1. DICOMシリーズを読み込み、`ImageOrientationPatient` からスライス法線を算出してソート
2. 各入力スライスの「受け持ち区間」を隣接中点(Voronoi)で定義（軸を隙間・重複なくタイル化）
3. 出力スライス中心の矩形窓 $[c-\text{FWHM}/2,\ c+\text{FWHM}/2]$ と各区間の **重なり長 = 重み**
4. 重みを総和1に正規化し、線形領域(HU/信号値)で加算
5. ジオメトリ(`ImagePositionPatient` / `SliceThickness` / `SpacingBetweenSlices`)と
   新UID・`ImageType=DERIVED` を付けて書き出し

矩形プロファイルに対してこの重なり積分は厳密で、端の部分被覆も自動的に扱える。
任意プロファイル関数を数値積分する実装なので、将来 RF由来プロファイル等に差し替え可能。

---

## 使い方

```bash
.venv/bin/python mri_slice_sim.py <入力dir> <出力dir> [オプション]
```

### 例: 5mm厚 / 6mm間隔の2Dを生成

```bash
.venv/bin/python mri_slice_sim.py SampleData out_sim \
    --thickness 5 --spacing 6 --profile rect --pattern "*.DCM"
```

出力例:

```
[load] 271 slices, in-plane 512x512, slice spacing ~2.000 mm, normal=[0. 0. 1.]
[sim ] profile=rect FWHM=5.0mm support=5.0mm -> 91 slices @ 6.0mm spacing
[save] 91 files -> out_sim
```

### オプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--thickness` | `5.0` | 出力スライス厚 [mm]（= プロファイルFWHM） |
| `--spacing` | `6.0` | 出力スライス間隔（中心間）[mm] |
| `--profile` | `rect` | スライス感度プロファイル: `rect`(MRI) / `gaussian`(CT比較用) |
| `--support` | 自動 | プロファイルの積分範囲 [mm]（rect=thickness、それ以外=3×thickness） |
| `--start` | 入力先頭 | 先頭出力スライス中心の法線方向位置 [mm] |
| `--pattern` | `*` | 入力ファイルの glob パターン |
| `--desc` | `Simulated 2D thick-slice` | 出力の `SeriesDescription` |
| `--orientation` | `native` | 出力面: `native`(入力スライス方向) / `axial` / `coronal` / `sagittal` |
| `--in-plane-spacing` | 入力最小 | リスライス時の面内画素間隔 [mm] |
| `--recon-step` | 入力最小 | リスライス時の法線方向の細刻み [mm] |
| `--ssp-file` | — | `calibrate.py` の実測SSP(.npy)。`--profile`/`--thickness` より優先 |
| `--in-plane-blur` | `0` | 面内ガウシアンPSFのσ[mm]（`calibrate --fit-inplane` の `sigma_mm`） |
| `--format` | `dicom` | 出力形式 `dicom`/`png`/`binary`/`both`(=dicom+png)/`all` |
| `--raw-no-flip-y` | off | binary(.raw)の行(y)反転をしない（既定は反転） |

---

## 出力形式（DICOM / PNG / binary）

`--format` で出力形式を選べる（[`recon_motion.py`](recon-motion.md) と同規約）。

| 値 | 出力 |
|---|---|
| `dicom`（既定） | 2D厚スライスDICOM（`*.DCM`、ジオメトリ・新UID付き） |
| `png` | 8bit グレースケール `sl00.png …`（ボリューム最大値で正規化） |
| `binary` | `<out_dir名>.raw`(int16 LE) + `.hdr`(`X Y Z 2 dx dy dz`) + `.tag` |
| `both` | dicom + png |
| `all` | dicom + png + binary |

binary の `.hdr` の z は**出力スライス間隔**、`.tag` に `slice_thickness_mm` と
`rescale_slope`（`magnitude = stored * rescale_slope`）を記録。すべて `out_dir` 直下に出る。

```bash
# 5mm厚/6mm間隔・軸位を PNG と binary でも出力
.venv/bin/python mri_slice_sim.py cc_dicom/<basename> out_ax \
    --thickness 5 --spacing 6 --orientation axial --pattern "*.dcm" --format all
```

---

## 出力面の選択（AX / COR / SAG）

`--orientation` で出力2Dの面を選べる。`native` は入力のスライス方向に沿って集約する
（リスライスなし、最速）。`axial` / `coronal` / `sagittal` を指定すると、患者座標系(LPS)で
3Dボリュームを **その面へリスライス（トリリニア再標本化）してから** プロファイル積分する。

```bash
# 軸位で取得した3Dデータから冠状断の5mm厚2Dを生成
.venv/bin/python mri_slice_sim.py SampleData out_cor \
    --thickness 5 --spacing 6 --orientation coronal --pattern "*.DCM"
```

処理の流れ:

1. 入力DICOMのジオメトリ(`IOP`/`PixelSpacing`/スライス位置)からボクセル→患者座標(LPS)の
   アフィンを構築
2. 出力面の正規直交基底 (列 u, 行 v, 法線 n) を設定（下表）
3. 法線方向に細かい刻み(`--recon-step`)で 3D を再標本化し、`build_weights` で
   プロファイル積分 → 指定厚・間隔の2Dスライスに集約
4. 出力面の `IOP`/`IPP`/`PixelSpacing` を付けて書き出し

| 面 | 列方向 u | 行方向 v | 法線 n (スライス積み上げ) | DICOM `IOP` |
|---|---|---|---|---|
| axial | L→R `[1,0,0]` | A→P `[0,1,0]` | I→S `[0,0,1]` | `[1,0,0, 0,1,0]` |
| coronal | L→R `[1,0,0]` | S→I `[0,0,-1]` | A→P `[0,1,0]` | `[1,0,0, 0,0,-1]` |
| sagittal | A→P `[0,1,0]` | S→I `[0,0,-1]` | R→L `[1,0,0]` | `[0,1,0, 0,0,-1]` |

> 注: リスライスの面内解像度は元データの解像度に律速される。等方性に近い3D入力では
> どの面でもほぼ同等だが、異方性入力(例: 薄い面内・厚いスライス)では、元のスライス方向を
> 面内に含む再フォーマット面の分解能が粗くなる。

---

## プロファイルの違い（rect と gaussian）

2mm間隔サンプル・出力中心10mm・FWHM=5mm のときの重み:

| プロファイル | 寄与サンプル位置 [mm] | 重み |
|---|---|---|
| `rect`（MRI） | 8, 10, 12 | 0.30 / 0.40 / 0.30 |
| `gaussian`（CT相当） | 4, 6, 8, 10, 12, 14, 16 | 0.009 / 0.070 / 0.240 / 0.363 / 0.240 / 0.070 / 0.009 |

矩形は裾を引かずシャープ、ガウシアンは広い範囲に裾を引く。これがMRI/CTの差を生む。

---

## 制約と注意

- **magnitude かつ同一コントラスト** のときに物理的に妥当。
- 位相情報が無いため **部分容積での位相打ち消しは再現されない**（境界の信号低下を過小評価しうる）。
- 面内解像度は変更しない（through-plane のみ）。面内を変える場合は別途
  **k空間トランケーション（Gibbsリンギング）** が必要。
- **2Dマルチスライス**の through-plane は本質的に空間領域の操作（RFスライスプロファイル）であり、
  スライス方向にk空間サンプリングは存在しない。よって z方向のFFTは2D多断面では非物理的。
- 入力DICOMが患者データの場合、リポジトリにコミットしない（`.gitignore` で `SampleData_*/` や
  `*.DCM` を除外済み）。

---

### 関連ページ

- [実ペアデータによる較正・検証 (`calibrate.py`)](calibration.md) — スライスプロファイルの実測
- [高磁場→低磁場シミュレーション (`lowfield_sim.py`)](lowfield.md)
- [← トップへ戻る](../README.md)
