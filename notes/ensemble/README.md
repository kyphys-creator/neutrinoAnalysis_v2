# Bias / dispersion study (Monte-Carlo ensemble)

📖 [English](README.en.md)

Bob Cousins が求めた point estimate のバイアスと分散を、Graciela の指示した手続きで計算したもの。

**手続き** — best fit が予測するデータ $N_i$ を Gaussian $\mathcal{N}(N_i,\sqrt{N_i})$ で揺らし、
各擬似データを再フィットして $\delta\hat\Phi_j$ の平均と分散を取る。
真値は実データへの best fit ($b_j=\langle\delta\hat\Phi_j\rangle-\delta\Phi_j^{\rm Best\,fit}$)。

**背景の扱い** — `results/` の図は**背景も揺らした版**（コードの `bkg_penalty=True`）。
$B_i=h_i+b_i$ を擬似データとは独立に $\mathcal{N}(B_i,\sqrt{B_i})$ から引き、フィットではその値を差し引く。
背景を揺らすべきという点は共著者間で合意済み。背景を公称値に固定した初期版は
`extra/bkg_fixed_figures/` に比較用として残してある。

> **No Bkg では両者は完全に一致する。** `background_scenario='none'` では `Bkg_vector` が全 bin でちょうど 0 なので
> 揺らす対象がなく、結果はビット単位で同一（バグではない）。差が出るのは Exp Bkg の 2 設定のみで、
> 分散が約 12–13% 増える。

共通設定: $\mathcal{E}=3$ kg·yr, $N_{\rm int}=80$, $M=500$, vertex selection ON, seed 20260821。

## フォルダ構成

```
scripts/     再実行用スクリプト（run_bias_variance.py, make_all_figures.py, make_histograms.py）
results/     本番4設定の図（共著者に送るもの）
extra/       背景固定版の図、ヒストグラム、chi2
data/        アンサンブルの生出力 (JSON) と補助データ
archive/     試行錯誤で作った旧版の図・ログ（参照用、使わない）
```

## results/ — 本番の4設定

`1eV_ExpBkg` / `1eV_NoBkg` / `5eV_ExpBkg` / `5eV_NoBkg` の各フォルダに、同じ4種類の図（pdf+png）:

| ファイル | 内容 |
|---|---|
| `bias_{tag}_N80_vertexon` | バイアス $b_j$（物理単位）+ $\pm\sigma_j$ の帯と I 型エラーバー |
| `relbias_reldisp_{tag}_N80` | 相対バイアス $b_j/\delta\Phi_j^{\rm BF}$ と相対分散 $\sigma_j/\delta\Phi_j^{\rm BF}$ を同一図に |
| `relbias2_reldisp2_{tag}_N80` | 上の 2 乗（対数軸） |
| `bias2_over_var_{tag}_N80` | $b_j^2/\sigma_j^2$（対数軸、Tikhonov 参考線 = 1） |
| `hist_deltaPhi_{tag}_N80` | Bob 要求: point estimate $\delta\hat\Phi_j$ の分布（5 エネルギー） |
| `hist_residual_{tag}_N80` | Bob 要求: $\delta\hat\Phi_j-\delta\Phi_j^{\rm true}$ の分布（真値は入力フラックス） |

相対量の図では **best fit = 0 の区間を除外**している（1eV で 6 区間、5eV で 7 区間、いずれも最高エネルギー側）。

MC 誤差: $\mathrm{Err}(b_j)=\sigma_j/\sqrt{M}$（図のエラーバー）、$\mathrm{Err}(\sigma_j^2)=\sigma_j^2\sqrt{2/(M-1)}=6.3\%$。
分散は $M-1$ で割っている。

### 主な数値（$b_j^2/\sigma_j^2$）

背景変動版（`results/`、括弧内は背景固定版 `extra/bkg_fixed_figures/`）:

| 設定 | $j$=0 | $j$=20 | $j$=45 | $j$=70 |
|---|---|---|---|---|
| 1eV ExpBkg | 0.455 (0.415) | 0.008 (0.008) | 0.005 (0.016) | 0.450 (0.582) |
| 1eV NoBkg  | 0.389 (同一) | 0.015 (同一) | 0.026 (同一) | 0.689 (同一) |
| 5eV ExpBkg | 0.380 (0.385) | 0.027 (0.053) | 0.043 (0.053) | 0.885 (1.098) |
| 5eV NoBkg  | 0.373 (同一) | 0.096 (同一) | 0.169 (同一) | 1.007 (同一) |

中央エネルギーではバイアス小、両端で $\sim0.4$–$1.0$（Tikhonov 相当）。
背景のレベル（No Bkg / Exp Bkg）による差も、背景の不確かさを入れるかどうかによる差も小さい。

## extra/

- `histograms/` — ヒストグラムの初期版（1eV ExpBkg・背景固定）。現行版は `results/<tag>/` にある。
- `bkg_fixed_figures/` — 背景を公称値に固定した版の図（4 設定 × 4 図）。`results/` と対で比較用。
- `chi2_1eV_b_T3_N80_M500.csv` — 500 擬似実験の $\chi^2$（背景固定・変動の両方、同一 seed なので replica 単位で比較可）。
  **値は $10^4$ で割ってある。**

## data/

- `N80/` — 本番run の JSON（`_bkgvar` 付きが背景変動版 = `results/` の元データ）。`samples`（全 500×80 の生フィット結果）、`mean`/`sd`/`bias`/`mse`/`corr`/`chi2` などを含む。
- `N160/` — $N_{\rm int}=160$。M=500（N80↔N160 比較用）と **M=10000**（Bob の「$10^4$ 個で」への回答、
  4 設定、seed 同一なので最初の 500 toys は M=500 run と一致）。実測 ~50–65 分/設定（0.3–0.4 s/fit）。
- `N180/` — $N_{\rm int}=180$。初期の M=500 探索run に加え、**M=10000**（4 設定、seed 20260821、
  N160 との安定性確認用。中央値は N160 と数%以内で一致、最端の相対量のみ増える）。
- `true_deltaPhi_1eV_N80.npy` — 入力フラックスから計算した真の $\delta\Phi_j$（区間平均）。
- `relative_1eV_N80.npz` — 1eV ExpBkg の相対量まとめ。

## 再実行

```bash
cd notes/ensemble
PY=~/miniforge3/envs/python313/bin/python
DYLD_LIBRARY_PATH=~/miniforge3/envs/python313/lib $PY scripts/run_bias_variance.py <thr> <bkg> <T> <M> <seed> <modes> <N_int> [bkgvar]
```

- `<thr>`: `1eV` | `5eV`
- `<bkg>`: `b` (exponential) | `none` (no background) | `flat` | `a` | `c`
- `<modes>`: `vertex_on` | `vertex_off` | `vertex_on,vertex_off`
- 8 番目の引数に `1` を渡すと背景も揺らす

例（本番4設定）:
```bash
$PY scripts/run_bias_variance.py 1eV b    3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 1eV none 3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 5eV b    3 500 20260821 vertex_on 80 1
$PY scripts/run_bias_variance.py 5eV none 3 500 20260821 vertex_on 80 1
$PY scripts/make_all_figures.py
$PY scripts/make_histograms.py
```

1 設定あたり約 70 秒。JSON は `data/N80/` に、図は `results/<tag>/` に出る。
背景固定版を作るには 8 番目の引数を省く（`make_all_figures.py` の読み込みファイル名も要変更）。

## 注意

- **頂点選択 LP は約 7 割の toy で失敗し、内部解にフォールバックしている**（実測: 1eV ExpBkg 100 toys で
  N80 76%・N160 67%）。原因は QP 内部解が単調性を許容誤差程度に破ったまま等式制約として LP に渡るため。
  `run_bias_variance.py` の `warnings.filterwarnings('ignore')` で従来は完全に不可視だった。
  この挙動は最初から全 run に共通なので、M=500/M=10⁴ 間・N80/N160 間の比較は整合している。
  つまりアンサンブルの「vertex ON」は実質「LP が成功した ~3 割だけ頂点、残りは内部解」の混合。
  実データのベストフィット（論文の階段解）では LP は成功している。
- 頂点選択 LP はまれに HiGHS 単体法内で退化サイクルに入り**ハングする**（5eV N180 で初出、
  N160 M=10⁴ では 4 run 中 2 run で発生）。対策として `_select_vertex` の全ソルバー呼び出しに
  **10 秒の時間制限**を追加（1eV/5eV 両方の `neutrino_analysis_band.py`）。タイムアウトした toy は
  上記と同じ内部解フォールバックになり、ログに `[vertex]` 行が出る。
- vertex selection の ON/OFF は低エネルギー端のバイアスに効く（1eV ExpBkg の $j$=0 で +104% vs +166%）。
  $j\gtrsim12$ では一致する。
