[← MRsimulation トップ](../README.md)

# 実ペアデータによる較正・検証 (`calibrate.py`)

同一患者・同一セッションで撮像した **実3D と 実2D(正解)** のペアがあれば、
スライスプロファイルを「仮定」ではなく **実測** し、シミュレーション精度を定量検証できる。

```
実3D ──[シミュレーション]──> 擬似2D
                                  │ 比較・最適化
実2D(正解) ──────────────────────┘
```

## 前提
- **同一セッション・体動なし** を想定。DICOM患者座標(IPP/IOP)だけで画素対応が取れるため
  レジストレーション不要。各実2Dスライスの幾何で3Dを再標本化する。
- magnitude / 別コントラスト(2D TSE と 3D SPACE 等)でも、強度の線形変換 `a·x+b` を
  同時推定して吸収する（SSP推定を単純なスケール差で歪ませないための正規化）。

## フォワードモデル

```
real2D(plane) ≈ a · Σ_t w(t; FWHM, ramp) · S3D(plane + t·n) + b
```

各2Dスライス平面で法線方向 `n` の密なオフセット `t` で3Dを一度だけトリリニア標本化し、
プロファイル `w` は **台形(FWHM, ramp)** で当てはめる。非線形最適化は (FWHM, ramp) の
2次元のみで、`a,b` は各反復で閉形式に解く。

## 使い方

```bash
# 1) SSPを推定（実測較正） → ssp.npy / ssp.json を出力
.venv/bin/python calibrate.py <実3D dir> <実2D dir> --pattern "*.dcm" --out-ssp ssp.npy

# 2) 推定SSPを焼き込んで擬似2Dを生成
.venv/bin/python mri_slice_sim.py <実3D dir> out_calib \
    --spacing 6 --thickness 5 --ssp-file ssp.npy --pattern "*.dcm"
```

`calibrate.py` の出力（推定FWHM/ramp、強度a,b、矩形ベースラインに対するSSE改善率、
fit画素でのNRMSE/Pearson r）で精度を確認できる。

## 推定器の自己検証

実データが無くても、既知の台形SSPで合成した「2D」を推定器が復元できるか検証できる:

```bash
.venv/bin/python calibrate.py <任意の3D dir> --pattern "*.dcm" --self-test
# truth: FWHM=5.0 ramp=1.0 a=1.3 b=50.0 / fitted: FWHM≈4.8 ramp≈0.8 ... PASS
```

## `calibrate.py` の主なオプション

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

## 面内解像度差の較正 (`--fit-inplane`)

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

## 一致が低いときの診断 (`--qa-dir`)

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

## 重要：検証は「同一シーケンス」の2Dで行う

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

### 関連ページ

- [3D薄スライス → 2D厚スライス (`mri_slice_sim.py`)](slice-simulation.md) — 較正したSSPの適用先
- [高磁場→低磁場シミュレーション (`lowfield_sim.py`)](lowfield.md)
- [← トップへ戻る](../README.md)
