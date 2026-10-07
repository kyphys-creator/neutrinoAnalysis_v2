# neutrinoAnalysis

📖 **[English](#english) · [日本語](#日本語)**

---

<a id="english"></a>
## English

Neutrino-flux optimization pipeline. The flux is reconstructed by χ² minimization
from the observed rates (`Ratebin`) and the response matrix (`CRmat`), and
Monte-Carlo (Neyman construction) confidence intervals are obtained for each
flux parameter.

Two interchangeable solver backends (`scipy` / `osqp`) are provided.

### Directory layout

```
README.md
1eV/                              # working directory for the 1 eV threshold
  neutrino_analysis_band.py       # main module (per-threshold copy; differs)
  CRmat/originalUnit/             # response matrices CRmat<intervals>_originalUnit.csv
  Ratebin/                        # Ratebin2 / Ratebin7
  Danny’s files/                  # theoretical flux curves
  confidence_band_<scen>_T<T>.ipynb
  comparison*.ipynb
  T3/  T20/  T300/                # per-T output roots (auto-created)
    scenario_bkg_<scen>/
      flux_*.pdf
      bands/band_bkg<scen>_idx<NNN>.json

5eV/                              # 5 eV threshold, same layout
  neutrino_analysis_fast.py       # 5eV also keeps the fast module alongside band
  neutrino_analysis_band.py
  ...

figures/                          # plot_inputs.py and the input figures it makes
paper/                            # mainv2.tex (Overleaf copy) and a compiled draft
references/                       # external papers and reference printouts
method1/                          # Method 1 (Appendix C): fit up to 7 MeV without truncating R_ij
  data/  results/                 # grids & response matrices / best fits & figures
notes/                            # analysis notes (ensemble/, ...)
correspondence/                   # email printouts: local only, git-ignored
```

**Important:** run notebooks with the cwd set to `1eV/` or `5eV/`. The code reads
`CRmat/originalUnit/` etc. by relative path from there. Outputs are written
under `T<T>/scenario_bkg_<scen>/`, so threshold × T × background-scenario
combinations never overwrite each other.

### Dependencies

```bash
pip install numpy scipy matplotlib pandas numba
pip install cvxpy osqp        # required when using solver='osqp'
pip install highspy           # optional; fastest vertex (staircase) solver
pip install joblib            # optional; enables scipy-backend MC parallelism
```

`cvxpy` brings CLARABEL / SCS, used for solver fallback. The piecewise-constant
(staircase) flux needs a *simplex* LP solver; the code tries HiGHS → GLPK →
SciPy-linprog in order. SciPy ships with cvxpy, so the staircase appears even
without `highspy` — but installing `highspy` makes vertex selection faster.
Tested on Python 3.12.

### Basic usage

Run with cwd `1eV/` or `5eV/` (the data paths anchor here).

```python
# At 5eV both fast and band can be imported; at 1eV only band.
from neutrino_analysis_band import NeutrinoAnalysis

# T is a constructor argument (default 3). Output goes to T<T>/scenario_bkg_<scen>/.
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='scipy', T=3)

res = a.optimize(a.data_vector)      # χ² minimization
print(res.fun / a.c)                 # χ² / c
a.plot_flux_comparison(save=True)    # optimized flux vs. theoretical models
print(a.scenario_dir, a.bands_dir)   # check the output paths
```

#### Switching solvers

```python
a.set_solver('osqp')   # fast backend (cvxpy + OSQP/CLARABEL/SCS)
```

- `scipy` — `trust-constr` with analytic Jacobian and constant Hessian. Stable.
- `osqp` — formulates χ² as a QP. The free solve is fast; fixed-parameter
  solves auto-route to CLARABEL. For underdetermined problems (params > bins),
  a HiGHS-simplex vertex selection returns a piecewise-constant (staircase)
  flux (`_OSQPBackend.vertex_select`, default `True`).

Both backends compute the same χ². For underdetermined problems the flux
*shape* is not unique, so backends may return different equally-optimal
solutions (see Caveats below).

### Δχ² scan (manual grid)

```python
a.optimize(a.data_vector)            # compute the best fit first
scan = a.scan_fixed_parameter(
    fixed_index=39, scan_range=0.95, num_points=21,
    num_pseudo_data=100, seed=42, n_jobs=1,
)
import pandas as pd
pd.DataFrame(scan)                   # included / cutoff / Δχ² per fixed value
```

`scan` is a list of dicts; `pd.DataFrame(scan)` turns it into a table directly.

### Confidence band by root finding (`neutrino_analysis_band.py`)

Instead of tuning grid endpoints by hand, locate the band edges directly with
bracketing + geometric bisection.

```python
from neutrino_analysis_band import NeutrinoAnalysis, load_band

a = NeutrinoAnalysis(background_scenario='flat', intervals='360',
                     GeV=0.32e16, solver='osqp', T=3)
a.optimize(a.data_vector)

# locate the 1σ / 90% / 2σ edges for one index in a single call
band = a.find_confidence_band(
    fixed_index=0,
    levels=(0.678, 0.90, 0.954),
    num_pseudo_data=50,   # bracketing stage (coarse)
    n_pseudo_edge=500,    # bisection at the edges (larger → more stable, slower)
    step=1.5, rel_tol=0.03, seed=42, verbose=True,
)
```

The outer reach is bracketed with the widest level (2σ); all narrower-level
edges nest inside that interval. Use larger `n_pseudo_edge` near the edges to
suppress Monte-Carlo noise in the cutoff. Fix `seed` so each evaluation is
reproducible (this stops the bisection from jittering).

#### Save per index and overlay on the optimize result

```python
for idx in [0, 5, 10, 20, 40]:
    a.find_and_save_band(idx,
                         num_pseudo_data=50, n_pseudo_edge=500,
                         step=1.5, rel_tol=0.03, seed=42)
# default outdir = a.bands_dir = 'T3/scenario_bkg_flat/bands'

# overlay the saved bands on the optimized flux
a.plot_flux_with_bands(f'{a.bands_dir}/band_*.json',
                       levels=(0.678, 0.90, 0.954), ylim=(0, 3e13))
```

- `save_band` / `load_band` — JSON, one file per index
- `find_and_save_band` — search one index and save immediately
- `plot_flux_with_bands` — overlay per-index asymmetric error bars on the
  scatter (centres = saved best fits, so they line up with the optimize result)

#### Multi-scenario comparison

```python
a_flat = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                          GeV=0.32e16, solver='osqp', T=3); a_flat.optimize(a_flat.data_vector)
a_a    = NeutrinoAnalysis(background_scenario='a',    intervals='180',
                          GeV=0.32e16, solver='osqp', T=3); a_a.optimize(a_a.data_vector)

a_flat.plot_band_comparison(
    {'flat': f'{a_flat.bands_dir}/band_*.json',
     'a':    f'{a_a.bands_dir}/band_*.json'},
    level=0.954,
    optimized={'flat': a_flat, 'a': a_a},   # overlay each scenario's optimized flux
    ylim=(0, 3e13), save=True,
)
```

- Each scenario uses its own colour. Bands are in physical units
  (cm⁻² s⁻¹) and are comparable regardless of the GeV unit choice.
- `optimized` takes `{label: NeutrinoAnalysis or flux array}` and scatters each
  scenario's optimized flux in the matching colour (the labels should match
  those in `groups`).

### Band-search parameter tuning

| Argument | Role | Recommended |
|---|---|---|
| `num_pseudo_data` | pseudo-data sample size during bracketing | enough to resolve the requested percentile; **≥50 for 2σ** |
| `n_pseudo_edge` | pseudo-data sample size during edge bisection (cutoff precision) | **≥500** (seed-to-seed noise converges) |
| `step` | bracketing expansion factor (reach = `v0 · step^max_bracket`) | 1.5. Too small (e.g. 1.05) with default `max_bracket=25` cannot reach a far upper edge and returns `inf` |
| `rel_tol` | relative tolerance of the edge (its resolution) | 0.03. **Sub-`rel_tol` differences carry no physical meaning** |
| `seed` | RNG seed (bisection reproducibility) | fix it |

Notes:
- Edge values carry MC noise from the cutoff. Small `n_pseudo_edge` makes the
  2σ width fluctuate by a few percent between seeds (90% is more central and
  stable; **2σ is an extreme percentile and therefore noisier**).
- If a scenario-to-scenario difference is smaller than `rel_tol`, its sign is
  not physical. To claim a difference, report **edge ± uncertainty from a
  seed ensemble** and show that the difference exceeds the uncertainty.
- When the upper edge is weakly constrained (shallow profile), report a
  **one-sided limit** rather than chasing precision.

### Backgrounds

`background_scenario` ∈ `{'a', 'b', 'b2', 'c', 'flat', 'none'}`. `set_background()`
switches it on the fly (the OSQP cache is cleared automatically).

### Background variation option (experimental)

By default the background B_i = h_i + b_i is fixed at its nominal value in the
χ². With `bkg_penalty=True` (constructor flag, or `a.set_bkg_penalty(True)`),
each pseudo-experiment instead draws a "measured" background by fluctuating
each B_i independently with a Gaussian of Poisson width,

```
B_varied_i ~ Normal(B_i, sqrt(B_i))     (event counts, clipped at 0)
```

drawn independently of the pseudo-data event count O_i, and the fit subtracts
that B_varied in place of the nominal background:

```
χ² = T · Σ (O − B_varied − M·x)² / O
```

The background is held fixed within each fit (it is *not* a free nuisance
parameter); only its variation across pseudo-experiments propagates the
background uncertainty into the Δχ² distribution, and hence into the band
width. Both backends support it; default off reproduces the previous behaviour
bit-for-bit.

```python
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='osqp', T=3, bkg_penalty=True)
a.optimize(a.data_vector)
a.find_and_save_band(10, num_pseudo_data=50, n_pseudo_edge=500, seed=42)
# saved as band_bkgflat_idx010_bkgpen.json — never overwrites standard bands
a.plot_bkg_sampling_check()   # sanity check of the Gaussian sampling
print(a.bkg_penalty_stats)    # edge-case counters (negative-B clips, …)
```

Edge cases: a `B_varied < 0` draw is clipped to 0; bins with `B_i <`
`bkg_zero_threshold` (default 1e−3 events) carry no background and are left at
`B_varied = 0`. All occurrences are counted in `bkg_penalty_stats`. Note
`'none'` has `B_i = 0` everywhere, so the variation is inert there.

### Caveats

- **Underdeterminacy.** With `intervals='180'` there are 180 flux parameters
  versus 29 observed bins (rank 29). The χ² minimizer is a 151-dimensional
  face, so the flux itself is not uniquely determined. `scipy` and `osqp` may
  return different points on that face. **The Δχ² profile (and the confidence
  interval) depends only on χ² and is unaffected**; physical conclusions are
  stable.
- **Scaling (QP and the vertex LP).** The natural-unit constant `c` is ~1e70,
  so both the OSQP QP and the vertex-selection LP column-scale internally
  (`x = D·z`). Without it, OSQP/CLARABEL can return a wrong point still labelled
  `optimal`, and the LP can satisfy `M_s·x = μ` only loosely — making the
  returned flux's χ² much larger than the QP's reported value (most visible at
  large `GeV`).
- **GeV choice affects OSQP precision.** Although the physical flux is
  GeV-invariant, OSQP's *relative* accuracy degrades as `GeV` grows. At a given
  threshold, `GeV` ≈ 0.3–0.6e16 reaches χ²/c ~1e-13, whereas `GeV=1e16` only
  ~1e-8 and `2e16` ~1e-7. For narrow bands (large `T`) this matters: prefer the
  smaller `GeV` that still works.
- **Band search assumption.** The band is taken to be connected (one crossing
  on each side).

### Troubleshooting

- **No staircase (smooth ramp instead).** Vertex selection needs a simplex LP
  solver. If none is found the flux falls back to the smooth interior solution
  and a one-time `RuntimeWarning` is emitted. `pip install highspy`, or rely on
  the bundled SciPy solver (check `cvxpy.installed_solvers()` contains
  `'SCIPY'`).
- **Bands look jagged / non-monotonic, especially at large `T`.** Usually a
  stale cache of bands computed before a scaling fix, or too-large `GeV`.
  Regenerate the bands with the current code and a smaller `GeV` (see above).
- **Edge returns `inf`.** `step` too small for a far upper edge; use `step=1.5`
  (default reach `v0·step^25`) or raise `max_bracket`.

### Backend differences

| | scipy | osqp |
|---|---|---|
| Solver | trust-constr | OSQP → CLARABEL → SCS (CLARABEL first for fixed) |
| Free solve | stable | fast |
| Fixed-parameter solve | stable | CLARABEL (OSQP does not converge at large `n`) |
| MC parallelism (`n_jobs`) | works | serial |
| Flux shape | algorithm-dependent vertex | explicit piecewise-constant vertex via simplex LP (HiGHS → GLPK → SciPy) |

---

<a id="日本語"></a>
## 日本語

ニュートリノフラックスの最適化パイプライン。χ² 最小化で観測レート（`Ratebin`）と
応答行列（`CRmat`）からフラックスを再構成し、モンテカルロ（Neyman 構成）で
各フラックスパラメータの信頼区間を求める。

2 つの交換可能なソルバーバックエンド（`scipy` / `osqp`）を持つ。

### ディレクトリ構成

```
README.md
1eV/                              # 検出器しきい値 1 eV の作業ディレクトリ
  neutrino_analysis_band.py       # 本体（しきい値別に独立コピー、差分あり）
  CRmat/originalUnit/             # 応答行列 CRmat<intervals>_originalUnit.csv
  Ratebin/                        # Ratebin2 / Ratebin7
  Danny’s files/                  # 理論フラックス曲線
  confidence_band_<scen>_T<T>.ipynb
  comparison*.ipynb
  T3/  T20/  T300/                # T 別の出力ルート (実行時に自動生成)
    scenario_bkg_<scen>/
      flux_*.pdf
      bands/band_bkg<scen>_idx<NNN>.json

5eV/                              # しきい値 5 eV、同じ構成
  neutrino_analysis_fast.py       # 5eV 側のみ fast 版も同居
  neutrino_analysis_band.py
  ...（1eV と同じレイアウト）

figures/                          # plot_inputs.py と、それが作る入力の図
paper/                            # mainv2.tex (Overleaf の写し) とコンパイル済み原稿
references/                       # 外部論文・参考資料の印刷
method1/                          # Method 1 (Appendix C): R_ij を切らずに 7 MeV までフィット
  data/  results/                 # グリッド・応答行列 / best fit・図
notes/                            # 解析ノート (ensemble/ など)
correspondence/                   # メールの印刷 (手元のみ、git 管理外)
```

**重要:** notebook の実行は **`1eV/` または `5eV/` をカレントディレクトリに**して行う。
コードはそこから `CRmat/originalUnit/` などを相対パスで読み込む。
出力は `T<T>/scenario_bkg_<scen>/` 配下に自動で振り分けられるため、しきい値・T・背景シナリオ
の組み合わせが互いに上書きすることはない。

### 依存パッケージ

```bash
pip install numpy scipy matplotlib pandas numba
pip install cvxpy osqp        # solver='osqp' を使う場合に必須
pip install highspy           # 任意: 階段状フラックスを最速で求める頂点ソルバー
pip install joblib            # 任意: scipy バックエンドのモンテカルロ並列化
```

`cvxpy` は CLARABEL / SCS を同梱（フォールバックに使用）。区分定数（階段状）の
フラックスには *シンプレックス* 系 LP ソルバーが必要で、コードは HiGHS → GLPK →
SciPy-linprog の順に試す。SciPy は cvxpy に同梱されるため `highspy` が無くても
階段状になるが、`highspy` を入れると頂点選択が速くなる。Python 3.12 で動作確認。

### 基本的な使い方

cwd を `1eV/` または `5eV/` にして実行する（データの相対パスがそこに紐づく）。

```python
# 5eV では fast / band どちらも import 可能。1eV では band のみ。
from neutrino_analysis_band import NeutrinoAnalysis

# T を引数で渡す（既定 T=3）。出力先は自動で T<T>/scenario_bkg_<scen>/...
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='scipy', T=3)

res = a.optimize(a.data_vector)      # χ² 最小化
print(res.fun / a.c)                 # χ²/c
a.plot_flux_comparison(save=True)    # 最適フラックス vs 理論曲線
print(a.scenario_dir, a.bands_dir)   # 出力先の確認
```

#### ソルバーの切り替え

```python
a.set_solver('osqp')   # 高速バックエンド (cvxpy + OSQP/CLARABEL/SCS)
```

- `scipy` — `trust-constr` を解析的ヤコビアン・定数ヘッシアンで解く。安定。
- `osqp` — χ² を二次計画として解く。フリー解は高速。固定パラメータ解は内部で
  CLARABEL に切り替わる。劣決定（パラメータ数 > ビン数）の場合は HiGHS シンプレックスで
  区分定数（階段状）の頂点解を選ぶ（`_OSQPBackend.vertex_select`、既定 True）。

両バックエンドが返す χ² は同一定義。劣決定の場合フラックス *形状* は一意でないため、
バックエンドにより異なる等価最適解になりうる（詳細は下記「注意点」）。

### Δχ² スキャン（手動グリッド）

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

### 信頼バンドの根探索（`neutrino_analysis_band.py`）

グリッドの上下限を手で詰める代わりに、ブラケット＋幾何二分法でバンドの端を直接探す。

```python
from neutrino_analysis_band import NeutrinoAnalysis, load_band

a = NeutrinoAnalysis(background_scenario='flat', intervals='360',
                     GeV=0.32e16, solver='osqp', T=3)
a.optimize(a.data_vector)

# 1 インデックスで 1σ/90%/2σ を同時に求める
band = a.find_confidence_band(
    fixed_index=0,
    levels=(0.678, 0.90, 0.954),
    num_pseudo_data=50,   # ブラケット段階 (粗い)
    n_pseudo_edge=500,    # 端の二分法 (大きいほど安定・遅い)
    step=1.5, rel_tol=0.03, seed=42, verbose=True,
)
```

最も広い 2σ で外側をブラケットし、その区間に全レベルの端が入れ子で収まる。
端付近のみ `n_pseudo_edge` を増やして cutoff のモンテカルロノイズを抑える。
`seed` 固定で各点を再現可能にし、二分法のジッターを防ぐ。

#### 複数インデックスを保存して重ねる

```python
for idx in [0, 5, 10, 20, 40]:
    a.find_and_save_band(idx,
                         num_pseudo_data=50, n_pseudo_edge=500,
                         step=1.5, rel_tol=0.03, seed=42)
# 既定の保存先 = a.bands_dir = 'T3/scenario_bkg_flat/bands'

# 保存したバンドを optimize 結果の上に重ねる
a.plot_flux_with_bands(f'{a.bands_dir}/band_*.json',
                       levels=(0.678, 0.90, 0.954), ylim=(0, 3e13))
```

- `save_band` / `load_band` … バンドを 1 インデックス 1 ファイルの JSON で保存・読込
- `find_and_save_band` … 1 インデックスを探索して即保存
- `plot_flux_with_bands` … 保存済みバンドを各インデックスの誤差棒として散布図に重ねる
  （バンド中心は保存時の `self.result.x[index]` なので最適フラックスと一致）

#### 複数シナリオのバンドを重ねて比較

```python
a_flat = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                          GeV=0.32e16, solver='osqp', T=3); a_flat.optimize(a_flat.data_vector)
a_a    = NeutrinoAnalysis(background_scenario='a',    intervals='180',
                          GeV=0.32e16, solver='osqp', T=3); a_a.optimize(a_a.data_vector)

a_flat.plot_band_comparison(
    {'flat': f'{a_flat.bands_dir}/band_*.json',
     'a':    f'{a_a.bands_dir}/band_*.json'},
    level=0.954,
    optimized={'flat': a_flat, 'a': a_a},   # 各シナリオの optimize 結果も重ねる
    ylim=(0, 3e13), save=True,
)
```

- 各シナリオを別色の誤差棒で重ねる。バンドは物理単位なので GeV 単位の違いに依らず比較可能。
- `optimized` に `{ラベル: NeutrinoAnalysis または flux 配列}` を渡すと、最適フラックス散布図を
  バンドと同色で重ねる（ラベルを `groups` と一致させると同色になる）。

### バンド探索パラメータの指針

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

### 背景シナリオ

`background_scenario` は `'a' / 'b' / 'b2' / 'c' / 'flat' / 'none'` から選ぶ。
`set_background()` で実行中に切り替え可能（OSQP のキャッシュは自動でクリアされる）。

### 背景変動オプション（試験実装）

既定では背景 B_i = h_i + b_i は χ² 内でノミナル値に固定。`bkg_penalty=True`
（コンストラクタ引数、または `a.set_bkg_penalty(True)`）にすると、疑似実験ごとに
各 B_i を**全イベント数 O_i とは独立に** Poisson 幅の Gaussian で揺らした
「測定された」背景

```
B_varied_i ~ Normal(B_i, sqrt(B_i))     （イベント数、0 でクリップ）
```

を生成し、フィットはノミナル背景の代わりにこの B_varied を引く：

```
χ² = T · Σ (O − B_varied − M·x)² / O
```

背景は各フィット内では固定（自由なニューサンスパラメータ**ではない**）。疑似実験
間での B_varied のばらつきだけが Δχ² 分布、ひいてはバンド幅に背景不定性を伝える。
両バックエンド対応。既定 off では従来とビット単位で同一の挙動。

```python
a = NeutrinoAnalysis(background_scenario='flat', intervals='180',
                     GeV=0.32e16, solver='osqp', T=3, bkg_penalty=True)
a.optimize(a.data_vector)
a.find_and_save_band(10, num_pseudo_data=50, n_pseudo_edge=500, seed=42)
# 保存名は band_bkgflat_idx010_bkgpen.json — 通常バンドを上書きしない
a.plot_bkg_sampling_check()   # Gaussian サンプリングのサニティチェック
print(a.bkg_penalty_stats)    # エッジケースのカウンタ（負 B クリップ等）
```

エッジケース: `B_varied < 0` の引きは 0 にクリップ、`B_i <`
`bkg_zero_threshold`（既定 1e−3 イベント）のビンは背景なしとして `B_varied = 0`
に固定。全て `bkg_penalty_stats` でカウントされる。なお `'none'` は全ビン
`B_i = 0` のため変動は不活性。

### 注意点

- **劣決定性**: 例えば `intervals='180'` はフラックス 180 パラメータに対し観測は 29 ビン
  （rank 29）。χ² 最小解は 151 次元の面となり、フラックスは一意に定まらない。
  scipy / osqp は等価最適解の中の異なる点を返しうる。**Δχ² プロファイル（信頼区間）は
  χ² のみに依存し、この非一意性に左右されない**ため、物理的結論は安定。
- **スケーリング（QP と頂点 LP）**: 自然単位定数 `c` が ~1e70 と巨大なため、OSQP の QP も
  頂点選択 LP も内部で列スケーリング（`x = D·z`）してから解く。これがないと OSQP/CLARABEL は
  誤った点を `optimal` として返し、LP は `M_s·x = μ` を緩くしか満たさず、返ってくる
  フラックスの χ² が QP の報告値より桁違いに大きくなる（`GeV` が大きいほど顕著）。
- **GeV の選択が OSQP 精度に効く**: 物理フラックスは GeV 不変だが、OSQP の *相対* 精度は
  `GeV` が大きいほど悪化する。同じしきい値で `GeV` ≈ 0.3–0.6e16 なら χ²/c ~1e-13、
  `GeV=1e16` で ~1e-8、`2e16` で ~1e-7。狭いバンド（大きい `T`）では効くので、
  動く範囲で小さい `GeV` を選ぶとよい。
- **バンド探索の前提**: バンドが連結（上下に交差点 1 つずつ）であることを仮定する。

### トラブルシュート

- **階段状にならない（なめらかなランプになる）**: 頂点選択にはシンプレックス系 LP ソルバーが
  必要。見つからないとフラックスはなめらかな内点解にフォールバックし、一度だけ
  `RuntimeWarning` を出す。`pip install highspy`、または同梱の SciPy ソルバーに頼る
  （`cvxpy.installed_solvers()` に `'SCIPY'` があるか確認）。
- **バンドがガタガタ／非単調、特に大きい `T` で**: たいていスケーリング修正前に作った
  古いバンドのキャッシュか、`GeV` が大きすぎるのが原因。現行コードと小さめの `GeV` で
  バンドを作り直す（上記参照）。
- **端が `inf` になる**: 遠い上端に対して `step` が小さすぎる。`step=1.5`（到達範囲は
  既定で `v0·step^25`）にするか `max_bracket` を増やす。

### バックエンドの違い（まとめ）

| | scipy | osqp |
|---|---|---|
| ソルバー | trust-constr | OSQP→CLARABEL→SCS（固定解は CLARABEL 優先） |
| フリー解 | 安定 | 高速 |
| 固定パラメータ解 | 安定 | CLARABEL（OSQP は大規模で未収束のため） |
| モンテカルロ並列 (`n_jobs`) | 有効 | 無効（逐次） |
| フラックス形状 | アルゴリズム依存の頂点 | シンプレックス LP（HiGHS→GLPK→SciPy）で区分定数の頂点を明示選択 |
