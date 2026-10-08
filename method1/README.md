# Method 1 (Appendix E): un-truncated fit up to 7 MeV

「やったこと」単位のサブフォルダ構成。**各サブフォルダは自分の `data/` と `results/`
だけで完結する**(独立性のため共有入力はフォルダごとに複製してある)。リポジトリ本体
(解析クラス `1eV/`・`5eV/`, `Mathematica/*/output/curlyR_table.csv`, 公式 CRmat,
physrev スタイル) への参照だけは共通。

> **命名規約 (2026-10-07)**: Method 1 と呼べるのは `uniform/` (method1u) と
> `equalized/` (method1eq) のみ — 純 curlyR 行列で, グリッド規則が 2 MeV を一切
> 参照しないもの。`spliced_crosscheck/` (旧タグ method1: 2 MeV 以下に公式列を継いだ
> 混成行列) は公式行列とのクロスチェック用で, Method 1 ではない。

| フォルダ | 内容 |
|---|---|
| `mockdata/` | **Method 1 のモックデータ (分解能あり)**: `make_mockdata_res.py <thr>` が curlyR_i (±1 eV 箱型分解能入り) を理論フラックス (With NC) に掛けて `data/Ratebin7res_<thr>_originalUnit.csv` を作り, `uniform/data`・`equalized/data` に複製。元の Ratebin7 は分解能なしで curlyR_i と整合しない (1–3 eV ビン −3%) ため, Method 1 はこちらを使う (2026-10-07 決定)。E_ν < E_min の寄与は本文の E_ν^min の定義どおり含めない (省いた量: 1 eV の 1–3 eV ビン 5.0%, 5 eV の 5–7 eV ビン 1.2%) |
| `kernels/` | 公式レシピの box カーネル節点 (0.01 MeV 刻み) を Mathematica で再生成。`wolframscript -file driver_kernels.wls <1eV|5eV>` → `data/curlyR_knots_*.csv`。他フォルダの knots はここからの複製 |
| `uniform/` | **method1u**: 等間隔グリッド (n=674/746) の純 curlyR 行列。`build_uniform_Rij.py` → `fit_uniform_lp.py <thr>` (データは既定で `Ratebin7res`, `--data=ratebin7` で元データ; LP フィット + 尾重み頂点 + Δχ²=0 マージ → d−1 段) → `make_paper_figure.py <thr>` |
| `equalized/` | **method1eq**: どの R_ij も 2 eV ビン行の最大要素を超えないよう細分する適応グリッド (n=920/1070)。使い方は uniform と同じ (タグ method1eq)。結果は method1u と同一の階段 |
| `fine_above_2MeV/` | **method1f** (試行, 2026-10-07): 2 MeV 以下 = Method 2 と同じ一様 180 区間, 2 MeV 以上 = 一定の細かい幅 (1 eV 2.09 keV = Δ/4.85, 5 eV 1.68 keV = Δ/5.27; N_int = 2577/3162)。幅は最後のビン (81–120 eV) の 2 MeV 以上での R_ij のピークが最初のビンのピークと同じ高さになる最小の本数で決める (`build_fine_above2_Rij.py`)。ただし最後のビンの curlyR_i は 2.005 MeV でピークになり 2 MeV 直下 (幅 Δ) でもほぼピークなので, 1.99 MeV の要素が最初のピークの 4.8 倍 (1 eV) / 5.2 倍 (5 eV) で残る。データは `Ratebin7res`。best fit は equalized と同じ (χ²=0, d−1 段) |
| `resolution/` | box (±1 eV 分解能入り) カーネル vs 分解能なしデータの不整合の研究: `resolution_cut_check.py` (1–3 eV ビンの −3% 閉包ずれの原因 = E_ν^min での切断), `first_step_check.py` / `plot_first_step.py` (最初の段の診断), `*_nores` 行列とそのフィット。**元データ (Ratebin7, 分解能なし) に対する研究。box 版は x(E_min) が nores 版より ~10% 高い。決着: データ側を分解能ありで作り直した (`mockdata/`); box 行列 + Ratebin7res で最初の段は理論曲線をまたぐ** |
| `spliced_crosscheck/` | 旧「method1」: 2 MeV 以下 = 公式 CRmat180 列 + 以上 = 等積分適応グリッド。cvxpy 版 (`build_and_fit.py` ほか) と純 scipy 版 (`fit_and_merge_scipy.py`), d−1 マージ (`exact_merge.py`), 縮退バンド, 段マージ実験, ノートブック。nores 版と ~2% で一致することが公式行列とのクロスチェックになっている |
| `stability/` | 安定性・頑健性の数値実験: `grid_dependence_test.py` (2 MeV 超の区間数 46–184 で best fit 不変), `stability_experiments.py` (S_A 適応 / S_B 一様 / S_C h_i 減算 / S_D Method 2 の切り分け; 実行結果は print のみ) |

共通の物理結論: Method 1 は適切なグリッドで完全に安定 (χ²=0, 頂点/初期値によらず同一解,
d−1 = 30/28 段で FDS 上限に飽和)。2 MeV 超は縮退しており (E_R^max(2 MeV) ≈ 120 eV =
反跳窓の上端), 総量のみ決まる — これが Method 2 で 2 MeV で切ることの定量的正当化。
