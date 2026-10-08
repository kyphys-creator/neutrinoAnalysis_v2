# neutrinoAnalysis

📖 [English](README.en.md)

ニュートリノフラックスの最適化パイプライン。χ² 最小化で観測レート（`Ratebin`）と
応答行列（`CRmat`）からフラックスを再構成し、モンテカルロ（Neyman 構成）で
各フラックスパラメータの信頼区間を求める。

2 つの交換可能なソルバーバックエンド（`scipy` / `osqp`）を持つ。

## ディレクトリ構成

```
neutrino_analysis_fast.py   # 本体。scipy / osqp の 2 バックエンド
neutrino_analysis_band.py   # fast + 信頼バンドの根探索・保存・比較プロット
montecarlo.ipynb            # 最適化・スキャンの使用例
confidence_band.ipynb       # 信頼バンド探索・保存・オーバーレイの使用例
confidence_band_flat/_Bkg/_noBkg.ipynb  # シナリオ別のバンド計算
comparison.ipynb            # 複数シナリオのバンド比較
CRmat/originalUnit/         # 応答行列 CRmat<intervals>_originalUnit.csv
Ratebin/                    # Ratebin2 / Ratebin7 (観測レート)
Danny’s files/              # 理論フラックス曲線 (fig1-solid / fig1-dashed ほか)
scenario_bkg_<x>/           # プロットの保存先 (実行時に自動生成)
scenario_bkg_<x>/bands/     # 信頼バンド JSON の保存先 (実行時に自動生成)
```

実行時のカレントディレクトリにこれらのデータフォルダが必要。

## 依存パッケージ

```bash
pip install numpy scipy matplotlib pandas numba
pip install cvxpy osqp        # solver='osqp' を使う場合に必須
pip install joblib            # scipy バックエンドのモンテカルロ並列化 (任意)
```

`cvxpy` は CLARABEL / SCS / HiGHS も同梱する（バックエンドのフォールバックと頂点選択に使用）。
Python 3.12 で動作確認。

## 基本的な使い方

```python
from neutrino_analysis_fast import NeutrinoAnalysis

# scipy バックエンド (安定・既定)
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='scipy')

res = a.optimize(a.data_vector)      # χ² 最小化
print(res.fun / a.c)                 # χ²/c
a.plot_flux_comparison(save=True)    # 最適フラックス vs 理論曲線
```

### ソルバーの切り替え

```python
a.set_solver('osqp')   # 高速バックエンド (cvxpy + OSQP/CLARABEL/SCS)
```

- `scipy` — `trust-constr` を解析的ヤコビアン・定数ヘッシアンで解く。安定。
- `osqp` — χ² を二次計画として解く。フリー解は高速。固定パラメータ解は内部で
  CLARABEL に切り替わる。劣決定（パラメータ数 > ビン数）の場合は HiGHS シンプレックスで
  区分定数（階段状）の頂点解を選ぶ（`_OSQPBackend.vertex_select`、既定 True）。

両バックエンドが返す χ² は同一定義。劣決定の場合フラックス *形状* は一意でないため、
バックエンドにより異なる等価最適解になりうる（詳細は下記「注意点」）。

## Δχ² スキャン（手動グリッド）

```python
a.optimize(a.data_vector)            # 先にベストフィットを求める
scan = a.scan_fixed_parameter(
    fixed_index=39, scan_range=0.95, num_points=21,
    num_pseudo_data=100, seed=42, n_jobs=1,
)
import pandas as pd
pd.DataFrame(scan)                   # 各固定値の included / cutoff / Δχ²
```

`scan` は辞書のリスト。`pd.DataFrame(scan)` でそのまま表になる。

## 信頼バンドの根探索（`neutrino_analysis_band.py`）

グリッドの上下限を手で詰める代わりに、ブラケット＋幾何二分法でバンドの端を直接探す。

```python
from neutrino_analysis_band import NeutrinoAnalysis, load_band

a = NeutrinoAnalysis(background_scenario='flat', intervals='360',
                     GeV=0.32e16, solver='osqp')
a.optimize(a.data_vector)

# 1 インデックスで 1σ/90%/2σ を同時に求める
band = a.find_confidence_band(
    fixed_index=0,
    levels=(0.678, 0.90, 0.954),
    num_pseudo_data=20,   # ブラケット段階 (粗い)
    n_pseudo_edge=200,    # 端の二分法 (大きいほど安定・遅い)
    step=1.5, rel_tol=0.03, seed=42, verbose=True,
)
```

最も広い 2σ で外側をブラケットし、その区間に全レベルの端が入れ子で収まる。
端付近のみ `n_pseudo_edge` を増やして cutoff のモンテカルロノイズを抑える。
`seed` 固定で各点を再現可能にし、二分法のジッターを防ぐ。

### 複数インデックスを保存して重ねる

```python
for idx in [0, 5, 10, 20, 40]:
    a.find_and_save_band(idx, outdir='scenario_bkg_flat/bands',
                         num_pseudo_data=50, n_pseudo_edge=500,
                         step=1.5, rel_tol=0.03, seed=42)

# 保存したバンドを optimize 結果の上に重ねる
a.plot_flux_with_bands('scenario_bkg_flat/bands/band_*.json',
                       levels=(0.678, 0.90, 0.954), ylim=(0, 3e13))
```

- `save_band` / `load_band` … バンドを 1 インデックス 1 ファイルの JSON で保存・読込
- `find_and_save_band` … 1 インデックスを探索して即保存
- `plot_flux_with_bands` … 保存済みバンドを各インデックスの誤差棒として散布図に重ねる
  （バンド中心は保存時の `self.result.x[index]` なので最適フラックスと一致）

### 複数シナリオのバンドを重ねて比較

```python
a_flat = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                          GeV=0.32e16, solver='osqp'); a_flat.optimize(a_flat.data_vector)
a_a    = NeutrinoAnalysis(background_scenario='a',    intervals='180',
                          GeV=0.32e16, solver='osqp'); a_a.optimize(a_a.data_vector)

a_flat.plot_band_comparison(
    {'flat': 'scenario_bkg_flat/bands/band_*.json',
     'a':    'scenario_bkg_a/bands/band_*.json'},
    level=0.954,
    optimized={'flat': a_flat, 'a': a_a},   # 各シナリオの optimize 結果も重ねる
    ylim=(0, 3e13), save=True,
)
```

- 各シナリオを別色の誤差棒で重ねる。バンドは物理単位なので GeV 単位の違いに依らず比較可能。
- `optimized` に `{ラベル: NeutrinoAnalysis または flux 配列}` を渡すと、最適フラックス散布図を
  バンドと同色で重ねる（ラベルを `groups` と一致させると同色になる）。

## バンド探索パラメータの指針

| 引数 | 役割 | 推奨 |
|---|---|---|
| `num_pseudo_data` | ブラケット段階の擬似データ数 | 要求パーセンタイルを表現できる数。2σ なら **≥50** |
| `n_pseudo_edge` | 端の二分法での擬似データ数（cutoff 精度） | **≥500**（seed 依存ノイズが収束） |
| `step` | ブラケットの拡大率（到達範囲 `v0·step^max_bracket`） | **1.5**。小さすぎ（例 1.05）+ 既定 `max_bracket=25` だと遠い上端に届かず端が `inf` になる |
| `rel_tol` | 端の相対許容幅（端の分解能） | 0.03。**これ以下の桁の差は意味を持たない** |
| `seed` | 乱数固定（二分法の再現性） | 固定する |

注意:
- 端の値は cutoff のモンテカルロ揺らぎを持つ。小さい `n_pseudo_edge` では seed を変えるだけで
  2σ 幅が数 % ブレる（90% は中心寄りで安定、**2σ は極端な分位点なので特にブレやすい**）。
- シナリオ間の差が `rel_tol`（端の分解能）より小さい場合、その大小に物理的意味はない。
  差を主張するなら、**seed アンサンブルで端 ± 誤差を出し、差が誤差を超えることを示す**。
- 上端が弱くしか制約されない（プロファイルが浅い）場合は、精度を上げるより**片側極限として報告**する方が筋が良い。

## 背景シナリオ

`background_scenario` は `'a' / 'b' / 'b2' / 'c' / 'flat' / 'none'` から選ぶ。
`set_background()` で実行中に切り替え可能（OSQP のキャッシュは自動でクリアされる）。

## 注意点

- **劣決定性**: 例えば `intervals='180'` はフラックス 180 パラメータに対し観測は 29 ビン
  （rank 29）。χ² 最小解は 151 次元の面となり、フラックスは一意に定まらない。
  scipy / osqp は等価最適解の中の異なる点を返しうる。**Δχ² プロファイル（信頼区間）は
  χ² のみに依存し、この非一意性に左右されない**ため、物理的結論は安定。
- **OSQP のスケーリング**: 自然単位定数 `c` が ~1e70 と巨大なため、内部で列スケーリング
  してから解く。これがないと OSQP/CLARABEL は誤った点を `optimal` として返す。
- **バンド探索の前提**: バンドが連結（上下に交差点 1 つずつ）であることを仮定する。

## バックエンドの違い（まとめ）

| | scipy | osqp |
|---|---|---|
| ソルバー | trust-constr | OSQP→CLARABEL→SCS（固定解は CLARABEL 優先） |
| フリー解 | 安定 | 高速 |
| 固定パラメータ解 | 安定 | CLARABEL（OSQP は大規模で未収束のため） |
| モンテカルロ並列 (`n_jobs`) | 有効 | 無効（逐次） |
| フラックス形状 | アルゴリズム依存の頂点 | HiGHS で区分定数の頂点を明示選択 |
