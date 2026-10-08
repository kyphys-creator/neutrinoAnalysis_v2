(* Method 1 用: 公式 06 と同一の節点 (0.01 MeV 刻み) で box カーネルを計算し CSV 出力.
   既存ファイルには一切書き込まない (出力は method1/data/ のみ).
   usage: wolframscript -file driver_kernels.wls <1eV|5eV> *)
thr = $ScriptCommandLine[[2]];
emin = If[thr === "1eV", 0.18, 0.41];
repo = "/Users/koichiromacstudio/Documents/Claude/neutrinoAnalysis_v2";
src = FileNameJoin[{repo, "Mathematica", thr, "src"}];
Get[FileNameJoin[{src, #}]] & /@ {"01_setup.wl", "02_functions_math.wl",
   "03_functions_response.wl", "04_response_defs.wl"};
evList = Range[emin, 7.0, 0.01];
pairs = Partition[datERbin, 2, 1];
toPhys = MeV*(10^3*grams)/cm^2;   (* 05b と同じ cm^2/(MeV kg) 規約 *)
t0 = AbsoluteTime[];
tab = Table[
   Table[CurlRbox[ev*MeV, pr[[1]]*eV, pr[[2]]*eV, 1*eV]*toPhys, {pr, pairs}],
   {ev, evList}];
Print[thr, ": ", Length[evList], " x ", Length[pairs], " knots in ",
      Round[AbsoluteTime[] - t0], " s"];
header = Prepend[
   Table[ToString[Round[pr[[1]]]] <> "-" <> ToString[Round[pr[[2]]]] <> " eV", {pr, pairs}],
   "Ev_MeV"];
rows = MapThread[Prepend, {tab, evList}];
Export[FileNameJoin[{repo, "method1", "kernels", "data", "curlyR_knots_" <> thr <> ".csv"}],
   Prepend[rows, header]];
Print["saved curlyR_knots_", thr, ".csv"];
