"""Run every tab handler of the app once and dump its outputs (Plotly JSON, tables, texts) for static rendering
(tools/compose_screens.py), used where no headless browser can run.

  BIMA_WEIGHTS_DIR=<staging> BIMA_DEVICE=cuda python space/tools/render_tabs.py --out <dir> [--example AFIB]
"""
import argparse, json, os, sys, time
SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SPACE)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--example", default="AFIB")
ap.add_argument("--config", default="I+II+V1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
import pandas as pd  # noqa: E402
import app as A  # noqa: E402
import engine as E  # noqa: E402

EX = next(c for c in A.example_choices() if c.startswith(a.example + " -"))
SRC = "Bundled example"
dump = dict(example=EX, config=a.config, runtime=A.runtime_md(), banner="Research demonstration — not for clinical use.",
            presets=[(n, c) for n, c, _ in A.PRESETS], tabs={})


def fig(name, f):
    if f is None:
        return None
    p = os.path.join(a.out, name + ".json"); open(p, "w").write(f.to_json()); return name


def tab(name, f):
    if f is None:
        return None
    p = os.path.join(a.out, name + ".csv"); f.to_csv(p, index=False); return name


t0 = time.perf_counter()
o = A.do_reconstruct(SRC, EX, None, 500, a.config, E.METHOD_BIMA, True, True)
st = o[-1]; st["ui"] = (SRC, EX, a.config)
dump["tabs"]["1"] = dict(controls=f"Model: BiMa-SR | Show linear-filter overlay: on | Run classifier: on",
                         items=[("md", o[3]), ("fig", fig("t1_plot", o[0])), ("md", o[2]), ("table", tab("t1_metrics", o[1])),
                                ("md", "[Download reconstruction (CSV, mV)]  [Download reconstruction (NPY, mV, 12 x N)]  "
                                       f"-> {os.path.basename(o[4])}, {os.path.basename(o[5])}")],
                         wall=time.perf_counter() - t0)
f2, md2, upd = A.class_view(st, True)
cls = upd["value"]
t0 = time.perf_counter()
e = A.do_explain(st, cls, 0, 5, SRC, EX, None, 500, a.config, E.METHOD_BIMA, True)
pn = e[5]["choices"]; sel = e[5]["value"]
dump["tabs"]["2"] = dict(controls=f"Include the linear-filter reconstruction: on | Class to explain: {cls} | attribution window "
                                  f"0.00-5.00 s | Predicted lead: {sel}",
                         items=[("fig", fig("t2_bars", f2)), ("md", md2), ("md", e[4]), ("fig", fig("t2_ig", e[0])),
                                ("fig", fig("t2_importance", e[1])), ("fig", fig("t2_profile", e[2])),
                                ("fig", fig("t2_matrix", e[3]))], wall=time.perf_counter() - t0)
t0 = time.perf_counter()
c = A.do_compare(SRC, EX, None, 500, a.config)
dump["tabs"]["3"] = dict(controls="[Run all input sets (BiMa-SR)]",
                         items=[("md", c[3]), ("fig", fig("t3_small_multiples", c[0])), ("fig", fig("t3_pcc", c[1])),
                                ("table", tab("t3_table", c[2]))], wall=time.perf_counter() - t0)
t0 = time.perf_counter()
u = A.do_uncertainty(SRC, EX, None, 500, 8)
dump["tabs"]["4"] = dict(controls="Noise draws N = 8 | [Sample]",
                         items=[("md", u[2]), ("fig", fig("t4_plot", u[0])), ("table", tab("t4_table", u[1]))],
                         wall=time.perf_counter() - t0)
T1, T2 = A.results_tables()
dump["tabs"]["5"] = dict(controls="", items=[("img", os.path.join(SPACE, "assets", "Graphical_Abstract.png")),
                                             ("md", "### Table 1 — Reconstruction fidelity (pooled median PCC, 95% record-bootstrap CI; generated from the bundled results CSV)"),
                                             ("table", tab("t5_table1", T1)),
                                             ("md", "### Table 2 — Diagnostic readability (record-level macro AUROC of the 12-lead classifier)"),
                                             ("table", tab("t5_table2", T2)),
                                             ("img", os.path.join(SPACE, "assets", "Figure1.png")), ("md", A.ABOUT)], wall=0)
t0 = time.perf_counter()
o = A.do_reconstruct(SRC, EX, None, 500, "II", E.METHOD_D6, False, True)
dump["tabs"]["1b"] = dict(controls="Preset: Monitor (II) | Model: DiBiMa-SR (DDIM-6) | Show linear-filter overlay: off",
                          items=[("md", o[3]), ("fig", fig("t1b_plot", o[0])), ("md", o[2]), ("table", tab("t1b_metrics", o[1]))],
                          wall=time.perf_counter() - t0, config="II")
json.dump(dump, open(os.path.join(a.out, "dump.json"), "w"), indent=1)
print({k: round(v["wall"], 2) for k, v in dump["tabs"].items()})
