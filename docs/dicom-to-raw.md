[← MRsimulation トップ](../README.md)

# DICOM → raw 一括変換 (`dicom_to_raw.py`)

DICOM が保存されたフォルダ階層を**再帰的に探索**し、**最下層（葉）フォルダごと**に含まれる
DICOM を 1 つの 3D 画像として読み込んで `.raw` / `.hdr` で出力する。あわせて、出力した各画像の
**DICOM タグを 1 行ずつ** `summary_dicom_tag.csv` にまとめる。

## 動作

1. `os.walk` で**サブフォルダを持たないディレクトリ（最下層）**を列挙。
2. 各最下層フォルダの DICOM を読み、**`SeriesInstanceUID` ごと**にグループ化
   （1フォルダに複数シリーズがあれば各々を別ボリュームに）。
3. スライス順は `ImageOrientationPatient × ImagePositionPatient`（法線方向の位置）で決定。
   無ければ `SliceLocation` → `InstanceNumber` でフォールバック。
4. 各シリーズを 3D スタックして `.raw` / `.hdr` を出力、タグを CSV に1行追記。

## 出力

| ファイル | 内容 |
|---|---|
| `<out>/患者ID_シリーズ_撮影日_時刻.raw` | 3D ボリューム（2byte バイナリ、x最速→y→z） |
| `<out>/患者ID_シリーズ_撮影日_時刻.hdr` | `X Y Z 2 dx dy dz`（mm。[recon_motion](recon-motion.md) と同形式） |
| `<out>/summary_dicom_tag.csv` | 全出力画像の DICOM タグ（**1画像=1行**） |

- ファイル名 = `PatientID_SeriesNumber_StudyDate_StudyTime`（無効文字は `_`、同名衝突は連番）。
- **画素は DICOM の格納値をそのまま** 2byte で保存（`PixelRepresentation` により int16/uint16）。
  実値は `magnitude/HU = 格納値 × RescaleSlope + RescaleIntercept`（係数は CSV に記録）。
- raw ビューア向けに**既定で行(y)反転**（`--no-flip-y` で無効化）。

### summary_dicom_tag.csv の主な列

患者/検査（`PatientID`, `PatientName`, `StudyDate`, `StudyTime`, `StudyInstanceUID`,
`AccessionNumber`）、シリーズ（`SeriesNumber`, `SeriesDescription`, `SeriesInstanceUID`,
`Modality`, `BodyPartExamined`）、装置（`Manufacturer`, `ManufacturerModelName`,
`MagneticFieldStrength`, `InstitutionName`）、シーケンス（`ProtocolName`, `SequenceName`,
`MRAcquisitionType`, `RepetitionTime`, `EchoTime`, `InversionTime`, `FlipAngle`,
`EchoTrainLength`, `PixelBandwidth`, `AcquisitionMatrix`）、ジオメトリ（`SliceThickness`,
`SpacingBetweenSlices`, `ImageOrientationPatient`, `PatientPosition`）、画素
（`RescaleSlope`, `RescaleIntercept`, `BitsStored`, `PixelRepresentation`,
`WindowCenter`, `WindowWidth`）、および派生情報（`output_file`, `source_folder`,
`n_slices`, `rows`, `columns`, `pixel_spacing_*_mm`, `slice_spacing_mm`,
`image_position_first`, `raw_dtype`, `y_flipped`）。

## 使い方

```bash
# <DICOMルート> 配下を再帰探索し、最下層ごとに 3D raw を raw_out/ へ
python dicom_to_raw.py <DICOMルート> --out-root raw_out

# 行(y)反転をしない（ビューアが top-down の場合）
python dicom_to_raw.py <DICOMルート> --out-root raw_out --no-flip-y
```

| オプション | 既定 | 説明 |
|---|---|---|
| `input` | — | DICOM を含むルートフォルダ（再帰探索） |
| `--out-root` | `raw_out` | `.raw/.hdr/.csv` の出力先 |
| `--no-flip-y` | off | raw の行(y)反転をしない（既定は反転） |

## 注意

- `.raw` / `.hdr` / `summary_dicom_tag.csv` は**患者データ**を含むため git に push しない
  （`.gitignore` で `*.raw`・`*.hdr`・`summary_dicom_tag.csv`・`raw_out/` を除外済み）。
- 1スライスのみ等で間隔が取れない場合は `SpacingBetweenSlices` → `SliceThickness` → 1mm の順で補完。
- `RescaleSlope/Intercept` は CSV に保存。実値が必要なら読み込み側で適用する
  （`値 = raw × RescaleSlope + RescaleIntercept`）。

---

### 関連ページ

- [k-spaceからの再構成 (`recon_motion.py`)](recon-motion.md) — 同じ `.raw/.hdr` 規約
- [← トップへ戻る](../README.md)
