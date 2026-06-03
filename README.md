# MRsimulation

3DのMRI(または任意)DICOMシリーズから、擬似的な **2D厚スライスDICOM** を合成するシミュレーションツール。

例: 1mmスライス厚の3Dデータ → **5mm厚 / 6mm間隔**の2D画像を、MRIの原理に基づいて生成する。

---

## なぜCTの方法そのままではダメか

CTでは画素値(HU)が線形減衰係数に比例し、スライス厚方向の信号は **線形加算（積分）** なので、「スライス方向にガウシアン平滑化 → ダウンサンプリング」でよく一致する。

MRIには2つの非線形性があり、そのままでは正確にならない。

1. **信号は複素横磁化の積分** — 厚いスライスの信号は magnitude の単純和ではなく、複素横磁化
   $M_{xy}(z)\,e^{i\phi(z)}$ の積分の絶対値。位相が場所で変わると **部分容積で信号が打ち消し合う**。
2. **画素値と組織パラメータが非線形** — 信号は PD, T1, T2/T2\* と シーケンス(TR, TE, フリップ角) の関数。
   出力のコントラストを変えるなら、フィルタ処理では作れず信号方程式での再計算が要る。

厚いスライスのMR信号は、連続体の横磁化 $M(z)$ をスライス感度プロファイル $P(z)$ で重み付き積分したもの:

$$S(c) = \int P(z - c)\,M(z)\,dz$$

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

## セットアップ

Python 3.9+ を推奨（numpy 2系・scipy のため）。

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install pydicom numpy scipy
```

> 注: numpy 2系では `np.trapz` が `np.trapezoid` に改名されている（本コードは対応済み）。

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

## 実ペアデータによる較正・検証 (`calibrate.py`)

同一患者・同一セッションで撮像した **実3D と 実2D(正解)** のペアがあれば、
スライスプロファイルを「仮定」ではなく **実測** し、シミュレーション精度を定量検証できる。

```
実3D ──[シミュレーション]──> 擬似2D
                                  │ 比較・最適化
実2D(正解) ──────────────────────┘
```

### 前提
- **同一セッション・体動なし** を想定。DICOM患者座標(IPP/IOP)だけで画素対応が取れるため
  レジストレーション不要。各実2Dスライスの幾何で3Dを再標本化する。
- magnitude / 別コントラスト(2D TSE と 3D SPACE 等)でも、強度の線形変換 `a·x+b` を
  同時推定して吸収する（SSP推定を単純なスケール差で歪ませないための正規化）。

### フォワードモデル

```
real2D(plane) ≈ a · Σ_t w(t; FWHM, ramp) · S3D(plane + t·n) + b
```

各2Dスライス平面で法線方向 `n` の密なオフセット `t` で3Dを一度だけトリリニア標本化し、
プロファイル `w` は **台形(FWHM, ramp)** で当てはめる。非線形最適化は (FWHM, ramp) の
2次元のみで、`a,b` は各反復で閉形式に解く。

### 使い方

```bash
# 1) SSPを推定（実測較正） → ssp.npy / ssp.json を出力
.venv/bin/python calibrate.py <実3D dir> <実2D dir> --pattern "*.dcm" --out-ssp ssp.npy

# 2) 推定SSPを焼き込んで擬似2Dを生成
.venv/bin/python mri_slice_sim.py <実3D dir> out_calib \
    --spacing 6 --thickness 5 --ssp-file ssp.npy --pattern "*.dcm"
```

`calibrate.py` の出力（推定FWHM/ramp、強度a,b、矩形ベースラインに対するSSE改善率、
fit画素でのNRMSE/Pearson r）で精度を確認できる。

### 推定器の自己検証

実データが無くても、既知の台形SSPで合成した「2D」を推定器が復元できるか検証できる:

```bash
.venv/bin/python calibrate.py <任意の3D dir> --pattern "*.dcm" --self-test
# truth: FWHM=5.0 ramp=1.0 a=1.3 b=50.0 / fitted: FWHM≈4.8 ramp≈0.8 ... PASS
```

### `calibrate.py` の主なオプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--out-ssp` | `ssp.npy` | 推定SSPの保存先（`(offset, weight)` 配列） |
| `--max-fit-slices` | `11` | fitに使う2D枚数（中央から等間隔抽出） |
| `--pixel-budget` | `60000` | fitに使う総画素数の上限 |
| `--fg-percentile` | `40` | 前景マスクのしきい値パーセンタイル |
| `--self-test` | — | 既知SSPの復元テスト（`dir2d` 不要） |

> 注: `--ssp-file` 使用時は `--thickness` も指定すると、出力DICOMの `SliceThickness`
> タグが公称厚に一致する（SSP自体は実測形状を使う）。

---

## 今後の拡張余地

| 目的 | 追加処理 |
|---|---|
| 位相打ち消しまで厳密化 | 複素データ(real/imag)での複素加算 |
| 現実的スライスプロファイル | 実RFパルス(windowed-sinc)を FFT/Bloch で導出 |
| 別コントラスト出力 | 定量マップ(PD/T1/T2) ＋ 信号方程式で再計算 |
| 面内解像度変更 | k空間トランケーション（Gibbsリンギング / sinc PSF） |
| 3Dパーティション撮像の模擬 | z方向FFT → パーティション切り出し → IFFT |
| 実機SNR合わせ | Rician ノイズ付加 |

---

## ライセンス

未指定。
