# MRsimulation

MRI DICOM から物理的に整合したシミュレーション画像を合成するツール群。

- **`mri_slice_sim.py`** — 3D薄スライス → 2D厚スライス（スライスプロファイル積分）
- **`calibrate.py`** — 実2D/3Dペアからスライスプロファイル等を実測較正・診断
- **`lowfield_sim.py`** — 高磁場(1.5T/3T) → 低磁場(0.3-0.5T)風の劣化（ノイズ/解像度）

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
| `--ssp-file` | — | `calibrate.py` の実測SSP(.npy)。`--profile`/`--thickness` より優先 |
| `--in-plane-blur` | `0` | 面内ガウシアンPSFのσ[mm]（`calibrate --fit-inplane` の `sigma_mm`） |

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
| `--qa-dir` | — | 診断出力先（sim/real/diff画像・残差シフト・線形/単調相関・幾何照合） |
| `--fit-inplane` | — | 面内ガウシアンPSF σ を SSP と同時推定（面内解像度差の較正） |
| `--edge-weight` / `--no-edge-weight` | ON | 勾配でエッジ重み付けfit。OFFだと平坦部に引っ張られFWHMが過大に出る |
| `--self-test` | — | 既知SSPの復元テスト（`dir2d` 不要） |

> SSP推定が公称厚より大幅に大きく出る場合、平坦部の過平滑化（ノイズ平均でSSEが下がる
> 方向への過適合）が原因。`--edge-weight`（既定ON）は勾配でエッジを重視し、真のFWHMへ
> 収束させる。推定SSPが過大なまま生成すると **シミュレーション2Dがボケる**（特に斜め断面では
> 厚さ×sin(角度) の面内滲みが増幅される）ので注意。

### 面内解像度差の較正 (`--fit-inplane`)

3Dが高解像（例 1024²）で実2Dが低解像の場合、ジオメトリ標本化だけの擬似2Dは
シャープすぎてエッジが合わず、SSIM/相関が低くなる。`--fit-inplane` はフォワードモデルに
面内ガウシアンPSFを加えて σ を SSP と同時推定する:

```
real2D ≈ a · [ Gauss_inplane(σ) ∘ Σ_t w(t)·S3D(plane + t·n) ] + b
```

```bash
.venv/bin/python calibrate.py <実3D> <実2D> --fit-inplane --out-ssp ssp.npy
# => in-plane σ [mm] と、σ=0→σ* での相関改善を表示
```

推定された `in-plane σ` を本体に渡して、面内ボケ込みの擬似2Dを生成する:

```bash
.venv/bin/python mri_slice_sim.py <実3D> out_calib \
    --spacing 6 --thickness 3 --ssp-file ssp.npy --in-plane-blur <σ_mm>
```

> 補足: SSP推定が公称厚より極端に大きく出る（例 nominal 3mm に対し FWHM≈9mm）場合、
> 面内解像度差を厚み方向で代理しているサイン。`--fit-inplane` を併用すると SSP は
> 公称付近へ戻り、信頼できる値になる。

### 一致が低いときの診断 (`--qa-dir`)

`--self-test` が PASS なのに実ペアで `r` / SSIM が低い場合、ツールではなく
**モデルと実データの食い違い**が原因。`--qa-dir qa/` を付けると診断材料を出力する:

```bash
.venv/bin/python calibrate.py <実3D> <実2D> --qa-dir qa/
# qa/compare.png (real / sim / 差分 / 結合ヒストグラム) ※matplotlib必要
# qa/geometry.txt, qa/diagnostics.txt, qa/{real,sim,diff}.npy
```

読み方:

| 兆候 | 解釈 | 対処 |
|---|---|---|
| `FrameOfReferenceUID 3D==2D : False` | 別座標系。IPP/IOPの直接対応が無効 | 剛体レジストレーションが必須 |
| 残差(全体)シフト `|shift|>1px` | 位置ずれ/体動 | レジストレーションで補正 |
| `r after bias-field` ≫ Pearson かつ bias幅が広い | コイル感度/正規化のバイアス場 | 滑らかな乗法バイアス場で補正 |
| `local warp p90` が大（>1–2mm） | 局所幾何歪み(B0/帯域差) | 非剛体(deformable)レジストレーション |
| Spearman ≫ Pearson | 単調な非線形コントラスト差 | 強度マップを線形→単調へ高度化 |
| どれでも上がらない（Spearmanも低い） | 真のコントラスト差(別シーケンス/脂肪抑制等) | 厚み較正の限界。組織別モデル等が必要 |

> `compare.png` の出力には matplotlib が必要（`pip install matplotlib`）。無い場合は
> `.npy` 配列と診断テキストのみ保存される。

### 重要：検証は「同一シーケンス」の2Dで行う

較正・検証は **実3Dと実2Dが同一シーケンス（同じコントラスト・同等の面内解像度）** である
ことを前提とする。例えば **3D T2 VISTA を 2D T2 TSE と比べる**と、一致が低くても
それはスライス厚シミュレーションの誤差ではなく、次の**シーケンス差**が主因:

- **面内解像度差** — 3D TSE/VISTA は長いエコートレインのT2ブラーで実効解像度が低く、
  高マトリクス(例 1024²)でもゼロフィル補間で見かけ上高精細なだけ。本ツールはボケを
  「足す」ことしかできず、よりシャープな2Dは原理的に復元できない（`hf_ratio>1` で検出）。
- **コントラスト差** — VISTA と TSE で組織の相対輝度が異なる（`Spearman` も低いまま）。

→ **同コントラストの厚スライス2D**（例: 3D VISTA → 厚 VISTA 2D）を作るのが目的なら、
スライス厚シミュレーション自体は正しく機能する。SSP実測・検証は **同一シーケンスの2D**
（例 2D VISTA）を参照に行うこと。別シーケンスの2Dを基準にした SSP 推定値は無効。

> 注: `--ssp-file` 使用時は `--thickness` も指定すると、出力DICOMの `SliceThickness`
> タグが公称厚に一致する（SSP自体は実測形状を使う）。

---

## 高磁場→低磁場シミュレーション (`lowfield_sim.py`)

1.5T/3Tの高画質MRIから、物理的に整合した **低磁場(0.3-0.5T)風の劣化画像** を合成する。
教師あり学習の **(入力=低磁場風 / 正解=高磁場)** ペア生成用。各スライスを独立処理し、
ジオメトリ（IPP/IOP/PixelSpacing）はそのまま保持する。

### モデル化する劣化

| 劣化 | 物理 | モデル |
|---|---|---|
| **SNR低下(主役)** | SNR ∝ B0^p (既定 p=1)。3T→0.5Tで約6倍ノイズ | **Rician**（複素ガウシアン→magnitude）。低SNRでノイズフロア/信号バイアスを再現 |
| **解像度低下/ボケ** | 大ボクセル・低マトリクス・再構成フィルタ | 面内ガウシアンPSF / k空間トランケーション(Gibbs) / ダウンサンプル |
| **T1短縮(任意・近似)** | T1 ∝ B0^α で低磁場ほど短縮 | 経験的コントラスト圧縮（既定OFF。厳密には定量マップが必要） |

### ノイズ量の決め方（「実画像と同等」にする鍵）

既存ノイズ `σ_high` を **画像コーナー(空気)の Rayleigh 統計** から実測し、磁場比でスケール:

```
σ_low = σ_high · (B0_high / B0_low)^p ,   σ_add = sqrt(σ_low² − σ_high²)
```

実際の低磁場画像があれば `--ref-low` でその背景から `σ_low` を直接実測して合わせられる
（最も実機に近い）。`--target-snr` / `--noise-sigma` で直接指定も可能。

### 使い方

```bash
# 3T -> 0.5T: 磁場比でノイズ、面内0.8mmボケ、k空間70%(Gibbs)
.venv/bin/python lowfield_sim.py highfield_dir lowfield_out --pattern "*.dcm" \
    --field-high 3.0 --field-low 0.5 --blur-mm 0.8 --kspace-keep 0.7

# 実低磁場画像の背景ノイズに合わせる
.venv/bin/python lowfield_sim.py highfield_dir lowfield_out \
    --ref-low real_lowfield_dir --in-plane-res 1.2

# 学習用に複数のノイズレベル/シードでデータ拡張
for s in 0 1 2; do
  .venv/bin/python lowfield_sim.py highfield_dir out_seed$s --seed $s --field-low 0.4
done
```

### 主なオプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--field-high` / `--field-low` | `3.0` / `0.5` | SNRスケール用の磁場強度[T] |
| `--snr-exponent` | `1.0` | SNR ∝ B0^p の p（体雑音支配で≈1、コイル雑音支配で最大≈1.75） |
| `--target-snr` | — | 目標SNRを直接指定（σ_high推定を使わない） |
| `--noise-sigma` | — | 追加前の目標σを実値で直接指定 |
| `--ref-low` | — | 実低磁場シリーズの背景からσを実測して合わせる |
| `--blur-mm` | `0` | 面内ガウシアンPSF σ[mm] |
| `--in-plane-res` | — | 目標面内解像度[mm]（等価ボケに換算） |
| `--kspace-keep` | `1.0` | k空間中央の保持割合(0-1]。<1でGibbs/解像度低下 |
| `--downsample` | `1.0` | 取得解像度の縮小率(0-1]。<1で縮小→ノイズ→拡大 |
| `--t1-strength` | `0` | 低磁場T1短縮の近似強度[0-1]（近似なので注意） |
| `--profile` | — | `lowfield_calibrate.py` のコントラスト別プロファイル(.json)。実低磁場に合わせて上書き |
| `--seed` | `0` | 乱数シード（データ拡張用） |

### 実低磁場サンプルからの較正 (`lowfield_calibrate.py`)

**unpaired（高磁場と別患者）** の実低磁場サンプルから、**コントラスト別の劣化プロファイル**
を実測して JSON 化する。保存するのは**スケール不変量**だけなので別スキャナ/別患者の
高磁場へ転用できる。

| 実測量 | 内容 | 生成時の使われ方 |
|---|---|---|
| `target_snr` | 代表信号/ノイズσ（コーナー空気のRayleigh） | Ricianノイズ量を合わせる |
| `intensity_quantiles` | 前景輝度の正規化分位（コントラスト記述子） | **ヒストグラムマッチング**で低磁場の見た目（T1短縮等）を移植 |
| `resolution_mm` | 取得面内解像度(PixelSpacing) | 高磁場との差を等価ボケσに換算 |

```bash
# 各コントラストごとにプロファイルを作成（unpairedでOK）
python lowfield_calibrate.py real_low_T1    --name T1    --out prof_T1.json
python lowfield_calibrate.py real_low_T2    --name T2    --out prof_T2.json
python lowfield_calibrate.py real_low_FLAIR --name FLAIR --out prof_FLAIR.json

# 高磁場の同コントラストへ適用（ノイズ/解像度/コントラストを実低磁場に合わせる）
python lowfield_sim.py high_T1    out_T1    --profile prof_T1.json --pattern "*.dcm"
python lowfield_sim.py high_FLAIR out_FLAIR --profile prof_FLAIR.json --pattern "*.dcm"
```

`--profile` 指定時は `target_snr`・`resolution_mm`・`intensity_quantiles` が
`--field-*`/`--blur-mm`/`--t1-strength` より優先される（`--blur-mm` はさらに上乗せ可）。

### 解像度ノブ（`--downsample` / `--kspace-keep` / `--blur-mm`）の決め方

この3つは **同じ「解像度比 ρ = res_high / res_low」を別表現したもの**。低磁場ほど粗いので
0<ρ<1（例: 高磁場1mm・低磁場2mm → ρ=0.5）。

| ノブ | ρからの換算 | アーチファクト | いつ使う |
|---|---|---|---|
| `--blur-mm` (σ) | σ = √(res_low²−res_high²)/2.355 | 滑らかなボケ | 再構成フィルタ/T2ブラー的劣化、微調整 |
| `--kspace-keep` | keep = ρ | **Gibbsリンギング** | 実低磁場にリンギングが見える（低マトリクス取得） |
| `--downsample` | factor = ρ | 部分容積＋**相関ノイズ** | 大ボクセル取得を最も物理的に再現（学習データ推奨） |

ρ は **取得マトリクスから真の解像度** `res = FOV / AcquisitionMatrix` で求めるのが本筋
（再構成PixelSpacingはゼロフィル補間で見かけ細かく、真の解像度を反映しないことがある）。
`lowfield_calibrate.py --high <高磁場の同コントラスト>` を付けると、両者の取得解像度から
**ρ と各ノブの推奨値を自動算出**して表示・JSONに保存する:

```bash
python lowfield_calibrate.py real_low_T1 --name T1 --high high_T1 --out prof_T1.json
#   -> resolution: recon=0.50mm acquired=1.20mm
#      vs high-field acquired=0.50mm -> ρ=0.42  推奨: --downsample 0.42 | --kspace-keep 0.42 | --blur-mm 1.09
```

学習データなら `--downsample`(=ρ) を主に、リンギングが見えるなら `--kspace-keep`(=ρ) を併用、
`--blur-mm` は実低磁場の見た目に合わせる微調整に使う。`--profile` 適用時は profile の
`resolution_mm`（取得解像度）から `--blur-mm` 相当が自動で入る。

### 注意・限界

- **コントラスト変換は周辺分布のヒストグラムマッチング**（unpairedで可能な範囲）。
  同部位・同コントラストのサンプルを使うこと。per-pixelの厳密変換にはペアが必要。
- **解像度は取得PixelSpacingの差から換算**（スペクトルからの自動推定は強ノイズで
  不安定なため不採用）。低磁場がゼロフィル補間で見かけ高精細な場合は実解像度を
  反映しないので、その時は `--blur-mm` で明示的に補う。
- **単一コイル magnitude を仮定**（Rician）。パラレルイメージング/多コイルは
  noncentral-chi + 空間変動 g-factor になるため本ツールは近似（`target_snr` は実測値
  として妥当だが、g-factorの空間変動は平均化される）。
- **T1コントラスト変化は粗い近似**。T2強調など磁場ロバストなコントラストでは省略可。
  厳密化には定量マップ(T1/T2)＋信号方程式が必要。
- 低磁場の **厚いスライス** も同時に作る場合は `mri_slice_sim.py` と組み合わせる。
- 磁場比パスは入力に測定可能な背景ノイズがある前提。ほぼ無ノイズの入力では
  `--target-snr` / `--noise-sigma` を使う。

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
