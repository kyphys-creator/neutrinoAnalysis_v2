repo = "/Users/koichiromacstudio/Documents/Claude/neutrinoAnalysis_v2";
src = FileNameJoin[{repo, "Mathematica", "1eV", "src"}];
Get[FileNameJoin[{src, #}]] & /@ {"01_setup.wl", "02_functions_math.wl",
   "03_functions_response.wl", "04_response_defs.wl"};
Do[
  t = AbsoluteTime[]; v = CurlRbox[ev*MeV, p[[1]]*eV, p[[2]]*eV, 1*eV];
  Print["CurlRbox[", ev, " MeV, ", p[[1]], "-", p[[2]], " eV] = ",
        ScientificForm[v, 4], "   ", Round[AbsoluteTime[] - t, 0.01], " s"],
  {ev, {0.5, 2.5, 6.0}}, {p, {{1, 3}, {45, 47}}}];
