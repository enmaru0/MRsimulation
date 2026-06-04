# MRsimulation

MRI DICOM / k-space から物理的に整合したシミュレーション画像を合成するツール群。

- **`mri_slice_sim.py`** — 3D薄スライス → 2D厚スライス（スライスプロファイル積分）
- **`calibrate.py`** — 実2D/3Dペアからスライスプロファイル等を実測較正・診断
- **`lowfield_sim.py`** — 高磁場(1.5T/3T) → 低磁場(0.3-0.5T)風の劣化（ノイズ/解像度）
- **`recon_motion.py`** — fastMRI 形式マルチコイル k-space(`.h5`) → 画像再構成（PNG / DICOM / binary 出力）

例: 1mmスライス厚の3Dデータ → **5mm厚 / 6mm間隔**の2D画像を、MRIの原理に基づいて生成する。

---

## ドキュメント

詳細は用途ごとにページを分けてある。

| ページ | 対象ツール | 内容 |
|---|---|---|
| [3D薄スライス → 2D厚スライス](docs/slice-simulation.md) | `mri_slice_sim.py` | スライスプロファイル積分、出力面(AX/COR/SAG)、プロファイル選択 |
| [実ペアデータによる較正・検証](docs/calibration.md) | `calibrate.py` | SSP実測、フォワードモデル、面内較正、QA診断 |
| [高磁場→低磁場シミュレーション](docs/lowfield.md) | `lowfield_sim.py` ほか | ノイズ/解像度劣化、低磁場プロファイル較正、手動ノイズ計測 |
| [k-spaceからの再構成](docs/recon-motion.md) | `recon_motion.py` | コイルRSS再構成、PNG/DICOM/binary 出力 |

---

## セットアップ

Python 3.9+ を推奨（numpy 2系・scipy のため）。

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install pydicom numpy scipy
# k-space 再構成(recon_motion.py)を使う場合は追加で:
.venv/bin/pip install h5py Pillow
```

> 注: numpy 2系では `np.trapz` が `np.trapezoid` に改名されている（本コードは対応済み）。

---

## 背景: なぜCTの方法そのままではダメか

CTでは画素値(HU)が線形減衰係数に比例し、スライス厚方向の信号は **線形加算（積分）** なので、「スライス方向にガウシアン平滑化 → ダウンサンプリング」でよく一致する。

MRIには2つの非線形性があり、そのままでは正確にならない。

1. **信号は複素横磁化の積分** — 厚いスライスの信号は magnitude の単純和ではなく、複素横磁化
   $M_{xy}(z)\,e^{i\phi(z)}$ の積分の絶対値。位相が場所で変わると **部分容積で信号が打ち消し合う**。
2. **画素値と組織パラメータが非線形** — 信号は PD, T1, T2/T2\* と シーケンス(TR, TE, フリップ角) の関数。
   出力のコントラストを変えるなら、フィルタ処理では作れず信号方程式での再計算が要る。

厚いスライスのMR信号は、連続体の横磁化 $M(z)$ をスライス感度プロファイル $P(z)$ で重み付き積分したもの:

$$S(c) = \int P(z - c)\,M(z)\,dz$$

本ツール群はこの原理に基づき、用途別に劣化/再構成を行う。各ツールの詳細は
[ドキュメント](#ドキュメント)の各ページを参照。

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
