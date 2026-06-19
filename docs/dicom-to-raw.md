[← MRsimulation トップ](../README.md)

# DICOM → raw 一括変換 (`dicom_to_raw.py`)

DICOM が保存されたフォルダ階層を**再帰的に探索**し、**最下層（葉）フォルダごと**に含まれる
DICOM を 1 つの 3D 画像として読み込んで `.raw` / `.hdr` で出力する。あわせて、出力した各画像の
**DICOM タグを 1 行ずつ** `summary_dicom_tag.csv` にまとめる。

## 動作

1. `os.walk` で**サブフォルダを持たないディレクトリ（最下層）**を列挙。
2. 各最下層フォルダの DICOM を読み、**`SeriesInstanceUID` ごと**にグループ化
   （1フォルダに複数シリーズがあれば各々を別ボリュームに）。
3. さらに**シリーズ内を「ボリュームを区別するタグ」でサブグループ化**
   （拡散 b 値・拡散傾斜方向・エコー番号・時相）→ [シリーズ内の自動分割](#シリーズ内の自動分割)。
4. スライス順は `ImageOrientationPatient × ImagePositionPatient`（法線方向の位置）で決定。
   無ければ `SliceLocation` → `InstanceNumber` でフォールバック。
5. 各ボリュームを 3D スタックして `.raw` / `.hdr` を出力、タグを CSV に1行追記。

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

## シリーズ内の自動分割

同じ `SeriesInstanceUID`（同じシリーズ番号）でも、**別の画像が混在**することがある
（例: 拡散強調の **b 値違い** `(0018,9087) DiffusionBValue` が1シリーズにまとまっている）。
これを**別々の `.raw` に分離**するため、シリーズ内を「1ボリューム内では一定・ボリューム間で
変わるタグ」で**サブグループ化**する。既定の分割タグ:

| タグ | (group,element) | ファイル名ラベル |
|---|---|---|
| `DiffusionBValue` | (0018,9087) | `b` 例 `_b1000` |
| `DiffusionGradientOrientation` | (0018,9089) | `dir` 例 `_dir3`（方向ごとの通し番号） |
| `EchoNumbers` | (0018,0086) | `e` 例 `_e2`（マルチエコー） |
| `TemporalPositionIdentifier` | (0020,0100) | `t` 例 `_t5`（ダイナミック） |

- 分割が起きた時だけファイル名に**サフィックス**が付く（例
  `PatientID_5_20230101_130000_b1000.raw`）。通常シリーズ（タグが一定/無い）は分割されず
  従来どおり 1 ボリューム。
- CSV に `split_label` / `diffusion_b_value` / `volume_in_series` 列を記録。
- **ベンダー私的タグ**で b 値を持つ機種（例 Siemens `(0019,100c)`）等は `--split-tags` で追加:
  ```bash
  python dicom_to_raw.py <root> --split-tags 0019,100c        # 私的b値タグを追加
  python dicom_to_raw.py <root> --split-tags EchoTime         # TEでも分けたい場合
  ```
- 分割したくない（1シリーズ=1ボリューム）場合は `--no-split`。

## 向きの制御

| オプション | 効果 |
|---|---|
| `--reverse-z` | 出力スライス(z=軸0)方向を反転（積層の向きが逆の時に使う） |
| `--absolute-zyx` | **患者絶対座標 LPS の ZYX 順**へ並べ替え（軸0=Z=S-I, 軸1=Y=A-P, 軸2=X=L-R、index増=+S/+P/+L）。撮像面(AX/COR/SAG)に依らず**常に一定の向き**で出力 |

- `--absolute-zyx` は `ImageOrientationPatient`/`ImagePositionPatient` から各配列軸の患者方向を求め、
  最近接の解剖軸へ **permute＋flip のみ**（リサンプルなし）で揃える。`.hdr` の dx/dy/dz も
  その軸順に合わせて並べ替える。AX/COR/SAG の同一検査がすべて同じ ZYX 軸で並ぶ。
- `--absolute-zyx` 指定時は向きが座標で確定するため **y反転は適用しない**。
- `--reverse-z` は両モードで効く（絶対座標時は Z=S-I を反転して I→S にしたい場合等）。
- IOP/IPP が無いデータでは絶対ZYX再配置はできず、取得順で出力（警告を表示）。
- 適用した向きは CSV の `absolute_zyx` / `reverse_z` / `y_flipped` / `axis_order` 列に記録。

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

# スライス方向(z)を反転
python dicom_to_raw.py <DICOMルート> --out-root raw_out --reverse-z

# 撮像面に依らず常に患者絶対座標 ZYX(S-I, A-P, L-R) で出力
python dicom_to_raw.py <DICOMルート> --out-root raw_out --absolute-zyx
```

| オプション | 既定 | 説明 |
|---|---|---|
| `input` | — | DICOM を含むルートフォルダ（再帰探索） |
| `--out-root` | `raw_out` | `.raw/.hdr/.csv` の出力先 |
| `--no-flip-y` | off | raw の行(y)反転をしない（既定は反転。`--absolute-zyx` 時は無効） |
| `--reverse-z` | off | 出力スライス(z)方向を反転 |
| `--absolute-zyx` | off | 患者絶対座標 LPS の ZYX 順へ並べ替え（撮像面に依らず一定の向き） |
| `--split-tags` | — | シリーズ内分割タグを追加（キーワード or `gggg,eeee`、カンマ区切り） |
| `--no-split` | off | シリーズ内のサブグループ分割をしない（1シリーズ=1ボリューム） |

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
