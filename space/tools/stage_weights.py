"""Collect the deployable artefacts of FINISHED runs into a local staging directory (outside the repository), which
deploy.py later uploads to the private model repo. Run directories are only read.

Staged layout (same paths in the private HF model repo):
  MANIFEST.json            what is available (models, FIR baselines, judge, amplitude prior) + provenance
  MANIFEST.sha256          sha256 of every staged file (sha256sum -c compatible)
  sr/<run>.pth             best.pth state_dict (BiMa-SR: bima_sr_<tag>; DiBiMa-SR: dibima_sr_<tag>), byte copy
  sr/<run>.config.json     run config.json (training recipe, n_params, in_idx)
  fir/fir_<tag>.npz        multi-input FIR baseline fitted on PTB-XL training windows (W [k*(2L+1), 12], in_idx, L)
  judge/judge12.pt         12-lead classifier checkpoint (model state_dict + info), byte copy of best.pt
  amplitude_prior.json     per configuration and lead: median over PTB-XL training windows of std_lead / s
                           (s = common input scale), used by the app to display predicted leads in mV

A run is "finished" when best.pth, config.json, summary.csv and done_<cohort>[_<sampler>].json for every test cohort
exist (baselines: fir_weights.npz and done_<cohort>_fir.json). Unfinished runs are listed and skipped.
Usage: python space/tools/stage_weights.py [--out /media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging]
"""
import argparse, datetime, hashlib, json, os, shutil, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from bima_ecgsr import config as CFG  # noqa: E402

DEFAULT_OUT = os.environ.get("BIMA_WEIGHTS_STAGING", "/media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging")


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def finished(run_dir, samplers):
    need = ["best.pth", "config.json", "summary.csv"] + [f"done_{c}{'_' + s if s else ''}.json" for c in CFG.COHORTS for s in samplers]
    missing = [f for f in need if not os.path.exists(os.path.join(run_dir, f))]
    return not missing, missing


def amplitude_prior(in_idx, data_dir):
    """median_w(std_c / s_w), s_w = sqrt(mean_{j in inputs} std_j^2), over PTB-XL training windows (ecg 10565 excluded)."""
    import pandas as pd
    S = np.load(os.path.join(data_dir, "STATS_ptbxl_train.npy")).astype(np.float64)
    meta = pd.read_csv(os.path.join(data_dir, "meta_ptbxl_train.csv"), usecols=["record_id"])
    keep = ~meta.record_id.isin([10565]).to_numpy()
    sd = S[keep, :, 1]; s = np.sqrt((sd[:, in_idx] ** 2).mean(1))
    return np.median(sd / s[:, None], 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=DEFAULT_OUT); a = ap.parse_args()
    out = os.path.abspath(a.out)
    if out.startswith(os.path.abspath(REPO) + os.sep):
        sys.exit("staging directory must be outside the repository")
    for sub in ("sr", "fir", "judge"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    man = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               models=[], fir=[], judge=None, amplitude_prior="amplitude_prior.json", skipped=[])
    amp = {}
    for label, spec, tag in CFG.CONFIGS:
        in_idx = [CFG.LEADS.index(x) for x in spec.split(",")]
        amp[label] = dict(in_idx=in_idx, ratio=[float(v) for v in amplitude_prior(in_idx, CFG.DATA_DIR)])
        cands = [("bima_sr", "BiMa-SR", "regression", [""])]
        if label in CFG.DIFFUSION_CONFIGS:
            cands.append(("dibima_sr", "DiBiMa-SR", "diffusion", ["onestep", "ddim6"]))
        for kind, family, mode, samplers in cands:
            run = f"{kind}_{tag}"; rd = CFG.run_dir(kind, tag)
            ok, missing = finished(rd, samplers)
            if not ok:
                man["skipped"].append(dict(run=run, reason="not finished", missing=missing)); print(f"[skip] {run}: missing {missing}")
                continue
            cfg = json.load(open(os.path.join(rd, "config.json")))
            assert cfg["mode"] == mode and cfg["in_idx"] == in_idx, (run, cfg["mode"], cfg["in_idx"])
            shutil.copy2(os.path.join(rd, "best.pth"), os.path.join(out, "sr", f"{run}.pth"))
            shutil.copy2(os.path.join(rd, "config.json"), os.path.join(out, "sr", f"{run}.config.json"))
            hist = json.load(open(os.path.join(rd, "history.json")))
            best = max(hist, key=lambda r: r["val_pcc_median"])
            import pandas as pd
            sm = pd.read_csv(os.path.join(rd, "summary.csv"))
            sm = sm[sm.group == "all_pred"][["cohort", "method", "pcc_median", "mae_mean", "n_windows"]].to_dict("records")
            man["models"].append(dict(run=run, family=family, mode=mode, config=label, input_leads=spec.split(","),
                                      in_idx=in_idx, weights=f"sr/{run}.pth", config_file=f"sr/{run}.config.json",
                                      n_params=cfg["n_params"], T=cfg.get("T"), samplers=[s for s in samplers if s] or None,
                                      best_epoch=best["epoch"], val_pcc_median=best["val_pcc_median"], test_all_pred=sm))
            print(f"[ok] {run} (best epoch {best['epoch']}, val pooled-median PCC {best['val_pcc_median']:.4f})")
        bd = CFG.run_dir("baselines", tag)
        if os.path.exists(os.path.join(bd, "fir_weights.npz")) and all(os.path.exists(os.path.join(bd, f"done_{c}_fir.json")) for c in CFG.COHORTS):
            d = np.load(os.path.join(bd, "fir_weights.npz"))
            assert list(d["in_idx"]) == in_idx
            np.savez(os.path.join(out, "fir", f"fir_{tag}.npz"), W=d["W"], in_idx=d["in_idx"], L=d["L"],
                     fit_window_index=d["fit_window_index"])
            man["fir"].append(dict(config=label, file=f"fir/fir_{tag}.npz", L=int(d["L"]), shape=list(d["W"].shape)))
            print(f"[ok] fir {tag} W {d['W'].shape}")
        else:
            man["skipped"].append(dict(run=f"baselines_{tag}", reason="not finished")); print(f"[skip] baselines_{tag}")
    jp = CFG.judge_ckpt(); js = os.path.join(os.path.dirname(jp), "summary.json")
    if os.path.exists(jp) and os.path.exists(js):
        shutil.copy2(jp, os.path.join(out, "judge", "judge12.pt"))
        man["judge"] = dict(file="judge/judge12.pt", **json.load(open(js)))
        print("[ok] judge12")
    json.dump(amp, open(os.path.join(out, "amplitude_prior.json"), "w"), indent=1)
    json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    files = sorted(os.path.relpath(os.path.join(dp, f), out) for dp, _, fs in os.walk(out) for f in fs if f != "MANIFEST.sha256")
    with open(os.path.join(out, "MANIFEST.sha256"), "w") as f:
        for rel in files:
            f.write(f"{sha256(os.path.join(out, rel))}  {rel}\n")
    print(f"[done] {len(files)} files in {out}")


if __name__ == "__main__":
    main()
