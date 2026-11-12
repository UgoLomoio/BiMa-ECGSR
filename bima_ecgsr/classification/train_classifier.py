"""Train the 1D-ResNet multi-label ECG classifier (the 12-lead judge) on real PTB-XL windows.

  python -m bima_ecgsr.classification.train_classifier --name judge12 --input-leads all [--resume]

Data   : X_ptbxl_{train,val}.npy (float16 [N,12,2500] per-window per-lead z-scores), meta_ptbxl_*.csv.
         Training windows of ecg_id 10565 (ADC-saturated samples) are excluded.
Targets: 17 sigmoid outputs = NORM, MI, STTC, CD, HYP + 12 panel labels (see classifier.LABELS).
Loss   : masked BCE-with-logits, averaged over valid (window, label) entries; panel label -1 masked;
         superclass terms masked where has_superclass == 0.
Input  : selected leads, re-z-scored per window/lead on the GPU (predict.zscore_windows, the same
         function used at inference). Training augmentation: random 2250-sample (4.5 s) crop per
         window; validation/test use the full 2500-sample window (global pooling).
Optim  : AdamW (lr 1e-3, wd 1e-2, betas 0.9/0.999), OneCycle schedule (cosine, pct_start 0.3) over
         --epochs (30) x steps; batch 128; bf16 autocast; grad-clip 1.0; seed 0.
Select : validation record-level macro AUROC over the 5 superclasses (record prob = mean of its two
         windows; records with has_superclass == 1), evaluated on the full fold-9 set each epoch;
         early stopping with patience 5.
Ckpt   : <runs>/<name>/last.pt every epoch (model+optimizer+scheduler+history), best.pt on improvement,
         log.jsonl, config.json; --resume continues from last.pt. Shuffling/crops are drawn from
         default_rng(seed*1000+epoch), so a resumed run reproduces the same data order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root (script use)
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.classification.classifier import LABELS, LEADS, SUPERCLASSES, build_model, parse_leads  # noqa: E402
from bima_ecgsr.classification.predict import aggregate_records, predict, record_labels, zscore_windows  # noqa: E402
from bima_ecgsr.classification.metrics import auroc  # noqa: E402

DATA = CFG.DATA_DIR
RUNS = CFG.CLS_DIR
EXCLUDE_TRAIN_ECG = (10565,)


def window_targets(meta: pd.DataFrame) -> np.ndarray:
    """[N, 17] float32 targets with NaN = masked."""
    Y = np.full((len(meta), len(LABELS)), np.nan, dtype=np.float32)
    hs = meta["has_superclass"].values == 1
    for j, lab in enumerate(LABELS):
        v = meta[lab].values.astype(np.float32)
        v[v < 0] = np.nan
        if lab in SUPERCLASSES:
            v[~hs] = np.nan
        Y[:, j] = v
    return Y


def val_superclass_auroc(model, Xv, meta_v, lead_idx, device):
    p = predict(model, Xv, lead_idx=lead_idx, batch_size=256, device=device)
    _, pr, codes = aggregate_records(p, meta_v)
    Yr = record_labels(meta_v, codes=codes)
    aucs = {}
    for j, lab in enumerate(SUPERCLASSES):
        m = Yr[:, j] >= 0
        aucs[lab] = auroc(Yr[m, j], pr[m, j])
    return float(np.mean(list(aucs.values()))), aucs


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-leads", default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--crop", type=int, default=2250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--runs", default=RUNS)
    args = ap.parse_args()

    lead_idx = parse_leads(args.input_leads)
    out = os.path.join(args.runs, args.name)
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb * 2**30 / total), 0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ---------------- data (in RAM, float16) ----------------
    t0 = time.time()
    meta_tr = pd.read_csv(os.path.join(args.data, "meta_ptbxl_train.csv"), low_memory=False)
    meta_v = pd.read_csv(os.path.join(args.data, "meta_ptbxl_val.csv"), low_memory=False)
    keep = ~meta_tr["record_id"].isin(EXCLUDE_TRAIN_ECG).values
    Xtr_all = np.load(os.path.join(args.data, "X_ptbxl_train.npy"), mmap_mode="r")
    Xtr = np.ascontiguousarray(Xtr_all[np.flatnonzero(keep)][:, lead_idx])  # [N, C, T] float16
    Ytr = window_targets(meta_tr[keep].reset_index(drop=True))
    Xv = np.ascontiguousarray(np.load(os.path.join(args.data, "X_ptbxl_val.npy"), mmap_mode="r")[:, lead_idx])
    n, C, T = Xtr.shape
    print(f"[data] train {Xtr.shape} (excluded {int((~keep).sum())} windows of ecg {EXCLUDE_TRAIN_ECG}), "
          f"val {Xv.shape}, leads {[LEADS[i] for i in lead_idx]}, {time.time()-t0:.0f}s", flush=True)

    info = dict(name=args.name, input_leads=",".join(LEADS[i] for i in lead_idx), lead_idx=lead_idx,
                labels=list(LABELS), n_train_windows=int(n), crop=args.crop, in_T=int(T),
                excluded_train_ecg=list(EXCLUDE_TRAIN_ECG))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(vars(args), **info), f, indent=1)

    # ---------------- model / optim ----------------
    model = build_model(C).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = n // args.batch_size  # drop last partial batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps,
                                                pct_start=0.3, anneal_strategy="cos")
    start_ep, best, best_ep, bad, history = 0, -1.0, -1, 0, []
    last_p = os.path.join(out, "last.pt")
    if args.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, best_ep, bad, history = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["bad"], ck["history"]
        print(f"[resume] from epoch {ck['epoch']} (best {best:.4f} @ {best_ep}, bad {bad})", flush=True)
        if ck.get("done"):
            print("[resume] run already finished", flush=True)
            return
    print(f"[model] {n_params/1e6:.2f} M params, {steps} steps/epoch", flush=True)

    Ytr_t = torch.from_numpy(Ytr)
    for ep in range(start_ep, args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + ep)
        perm = rng.permutation(n)
        offs = rng.integers(0, T - args.crop + 1, size=n) if args.crop < T else np.zeros(n, int)
        t1, tot, cnt = time.time(), 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        for b in range(steps):
            idx = np.sort(perm[b * args.batch_size:(b + 1) * args.batch_size])
            xb = Xtr[idx]
            if args.crop < T:
                o = offs[idx]
                ar = o[:, None] + np.arange(args.crop)[None]
                xb = np.take_along_axis(xb, ar[:, None, :].repeat(C, 1), axis=2)
            xb = zscore_windows(torch.from_numpy(xb).to(device, non_blocking=True))
            yb = Ytr_t[idx].to(device)
            mask = ~torch.isnan(yb)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(xb)
            loss_el = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(yb), reduction="none")
            loss = (loss_el * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); cnt += 1
        peak_train = torch.cuda.max_memory_allocated() / 2**30
        vmac, vaucs = val_superclass_auroc(model, Xv, meta_v, None, device)
        improved = vmac > best
        if improved:
            best, best_ep, bad = vmac, ep, 0
        else:
            bad += 1
        rec = dict(epoch=ep, train_loss=tot / cnt, val_superclass_macro_auroc=vmac, val_auroc=vaucs,
                   lr=sched.get_last_lr()[0], sec=time.time() - t1, peak_gpu_gb=peak_train,
                   peak_gpu_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        with open(os.path.join(out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        done = bad >= args.patience or ep == args.epochs - 1
        if improved:
            atomic_save(dict(model=model.state_dict(), info=info, epoch=ep, val_macro=vmac), os.path.join(out, "best.pt"))
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep,
                         best=best, best_ep=best_ep, bad=bad, history=history, info=info, done=done), last_p)
        if done:
            break
    summ = dict(name=args.name, input_leads=info["input_leads"], best_epoch=best_ep, best_val_superclass_macro_auroc=best,
                epochs_run=len(history), n_params=n_params,
                max_peak_gpu_gb=max(h["peak_gpu_gb"] for h in history))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("[done]", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
