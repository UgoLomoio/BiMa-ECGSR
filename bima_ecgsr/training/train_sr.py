"""Reduced-lead -> 12-lead ECG reconstruction: training and evaluation (PTB-XL train; PTB-XL test, Georgia, China).

Works for any set of input leads. The global residual is applied by a wrapper (LeadResidualSR) that calls
BiMaSRNet.encoder_forward / decoder_forward(lr=None) and then adds every input lead into its own output channel.
For input = [II] this is numerically the backbone's own lead-II residual path (decoder_forward(..., lr=lr) with
UNMASK_CHANNELS {1: [1]}); see tests/test_sr.py.

Data: memmaps in $BIMA_DATA_DIR (float16 [N,12,2500] per-window per-lead z-scores of 0.5-40 Hz band-passed 5-s windows;
STATS [N,12,2] = (mean_mV, std_mV)).
  INPUT  (k leads): x_mV = z*std + mean for the input leads; remove each lead's mean; divide all input leads of a
         window by one common scale s = sqrt(mean_j std_j^2) (keeps inter-lead amplitude ratios; for k = 1 this is the
         lead's z-score).
  TARGET: all 12 leads as stored (per-lead z-scores).
PTB-XL training windows of ecg_id 10565 (ADC-saturated) are excluded from training and from the FIR fit.

Model: BiMaSRNet with BEST_PARAMS (Mamba v1, dim 64, d_state 16, 4 blocks x 8 layers, residual_internal),
num_channels = k.
  regression (BiMa-SR)   : encoder input = inputs (k ch), one forward pass.
  diffusion  (DiBiMa-SR) : DDPM T=20, squaredcos_cap_v2 + zero-terminal-SNR, x0 ('sample') prediction, trailing
                           spacing; encoder input = [x_t (12), inputs (k)]; inference from x_T ~ N(0, I) only,
                           with one-step (t = T-1) and DDIM-6 (eta 0). Validation uses DDIM-6.
At inference the model sees only the k input leads (and, for diffusion, its own noise); target leads are never inputs.
Recipe: 20 epochs, AdamW lr 1e-3 cosine -> 1e-5 (per step), wd 1e-6, effective batch 32 (4 micro-batches x 8),
bf16 autocast, grad clip 1.0. Model selection: pooled median PCC (median over window x predicted-lead pairs; predicted
leads = all non-input leads) on the fixed 1,024 validation windows (val_select_1024.npy).
Test: best checkpoint on PTB-XL test (fold 10), Georgia, China. Saved per cohort (and per diffusion sampler):
  pred_<cohort>[_<sampler>].npy   float16 [N,12,2500]; input-lead channels REPLACED by the true input z-scores
  metrics_<cohort>[_<sampler>].npz pcc, mae, rmse float32 [N,12] vs. the stored z-score target (all 12 leads)
  summary.csv (groups: all predicted / predicted limb / predicted precordial leads), per_lead.csv
Baselines (--mode baselines, CPU): multi-input linear FIR (lags +-50 per input lead, float64 normal equations in
chunks, fit on 4,000 random PTB-XL train windows, seed 0) on the same normalised inputs; for configs containing I and
II also the exact Einthoven/Goldberger derivation of III, aVR, aVL, aVF from the destandardised inputs (then
z-scored), remaining leads from the FIR ('einthoven_fir').

Usage (from the repository root):
  python -m bima_ecgsr.training.train_sr --mode regression --input-leads II        --name bima_sr_II   --amp --accum 4
  python -m bima_ecgsr.training.train_sr --mode diffusion  --input-leads II        --name dibima_sr_II --amp --accum 4
  python -m bima_ecgsr.training.train_sr --mode baselines  --input-leads I,II,V2   --name baselines_I_II_V2
  smoke: add --epochs 1 --max-steps 20 --n-val 64 --n-test 64 --out-root <scratch>
"""
import argparse, os, sys, json, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root (script use)
from bima_ecgsr import config as CFG  # noqa: E402

DATA = CFG.DATA_DIR
OUT_ROOT = CFG.RUNS_DIR
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB, PREC = list(range(6)), list(range(6, 12))
COHORTS = ['ptbxl_test', 'georgia', 'china']
EXCLUDE_TRAIN_RECORDS = [10565]  # ADC-saturated samples
FIR_L, FIR_NFIT, FIR_SEED = 50, 4000, 0


# ----------------------------------------------------------------------------- leads / normalisation
def parse_leads(s):
    m = {l.lower(): i for i, l in enumerate(LEADS)}
    idx = sorted({m[t.strip().lower()] for t in s.split(',') if t.strip()})
    if not idx:
        raise ValueError('no input leads')
    return idx


def lead_groups(in_idx):
    pred = [c for c in range(12) if c not in in_idx]
    return {'all_pred': pred, 'limb_pred': [c for c in pred if c in LIMB], 'prec_pred': [c for c in pred if c in PREC]}


def norm_inputs_np(z, st, in_idx):
    """z [B,12,L] (z-scores), st [B,12,2] (mean_mV, std_mV) -> [B,k,L] float64, common-scale normalised inputs."""
    z = np.asarray(z, np.float64)[:, in_idx]; st = np.asarray(st, np.float64)[:, in_idx]
    x = z * st[..., 1:2] + st[..., 0:1]
    x = x - x.mean(-1, keepdims=True)
    s = np.sqrt((st[..., 1] ** 2).mean(1))
    return x / np.maximum(s, 1e-6)[:, None, None]


def norm_inputs_t(z, st, in_idx):
    """torch version of norm_inputs_np (float32): z [B,12,L], st [B,12,2] -> [B,k,L]."""
    z = z[:, in_idx].float(); st = st[:, in_idx].float()
    x = z * st[..., 1:2] + st[..., 0:1]
    x = x - x.mean(-1, keepdim=True)
    s = (st[..., 1] ** 2).mean(1).sqrt().clamp_min(1e-6)
    return x / s[:, None, None]


# ----------------------------------------------------------------------------- data
def load_cohort(name, mmap=True):
    X = np.load(os.path.join(DATA, f'X_{name}.npy'), mmap_mode='r' if mmap else None)
    S = np.load(os.path.join(DATA, f'STATS_{name}.npy'))
    return X, S


def train_keep_index():
    import pandas as pd
    m = pd.read_csv(os.path.join(DATA, 'meta_ptbxl_train.csv'), usecols=['window_index', 'record_id'])
    bad = m.record_id.isin(EXCLUDE_TRAIN_RECORDS).to_numpy()
    assert (m.window_index.to_numpy() == np.arange(len(m))).all()
    return np.flatnonzero(~bad)


# ----------------------------------------------------------------------------- metrics / summaries
def metrics_np(pred, tgt):
    """pred, tgt [N,12,L] -> per window x lead PCC, MAE, RMSE (float32)."""
    pred = np.asarray(pred, np.float64); tgt = np.asarray(tgt, np.float64)
    zp = (pred - pred.mean(-1, keepdims=True)) / (pred.std(-1, keepdims=True) + 1e-8)
    zt = (tgt - tgt.mean(-1, keepdims=True)) / (tgt.std(-1, keepdims=True) + 1e-8)
    err = pred - tgt
    return ((zp * zt).mean(-1).astype(np.float32), np.abs(err).mean(-1).astype(np.float32),
            np.sqrt((err ** 2).mean(-1)).astype(np.float32))


def metrics_t(pred, tgt):
    pred = pred.float(); tgt = tgt.float()
    zp = (pred - pred.mean(-1, keepdim=True)) / (pred.std(-1, unbiased=False, keepdim=True) + 1e-8)
    zt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, unbiased=False, keepdim=True) + 1e-8)
    err = pred - tgt
    return (zp * zt).mean(-1), err.abs().mean(-1), err.pow(2).mean(-1).sqrt()


def pooled_median_pcc(pcc, cols):
    return float(np.median(pcc[:, cols])) if cols else float('nan')


def group_rows(cohort, method, pcc, mae, rmse, in_idx):
    rows = []
    for g, cols in lead_groups(in_idx).items():
        if not cols:
            continue
        p, a, r = pcc[:, cols], mae[:, cols], rmse[:, cols]
        rows.append(dict(cohort=cohort, method=method, group=g, leads='|'.join(LEADS[c] for c in cols),
                         n_windows=int(pcc.shape[0]), n_pairs=int(p.size),
                         pcc_median=float(np.median(p)), pcc_q25=float(np.quantile(p, .25)),
                         pcc_q75=float(np.quantile(p, .75)), pcc_mean=float(p.mean()),
                         window_mean_pcc_median=float(np.median(p.mean(1))),
                         mae_median=float(np.median(a)), mae_mean=float(a.mean()), rmse_mean=float(r.mean())))
    return rows


def lead_rows(cohort, method, pcc, mae, rmse, in_idx):
    return [dict(cohort=cohort, method=method, lead=LEADS[c], is_input=c in in_idx,
                 group=('input' if c in in_idx else ('limb' if c in LIMB else 'precordial')),
                 pcc_median=float(np.median(pcc[:, c])), pcc_mean=float(pcc[:, c].mean()),
                 mae_mean=float(mae[:, c].mean()), rmse_mean=float(rmse[:, c].mean())) for c in range(12)]


def write_summaries(out, in_idx):
    """Rebuild summary.csv / per_lead.csv from every metrics_*.npz present in <out> (resume-safe)."""
    import pandas as pd, glob
    rows, per = [], []
    for f in sorted(glob.glob(os.path.join(out, 'metrics_*.npz'))):
        key = os.path.basename(f)[len('metrics_'):-4]
        coh = next(c for c in COHORTS if key == c or key.startswith(c + '_'))
        meth = key[len(coh) + 1:] or 'regression'
        d = np.load(f)
        rows += group_rows(coh, meth, d['pcc'], d['mae'], d['rmse'], in_idx)
        per += lead_rows(coh, meth, d['pcc'], d['mae'], d['rmse'], in_idx)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, 'summary.csv'), index=False)
        pd.DataFrame(per).to_csv(os.path.join(out, 'per_lead.csv'), index=False)
    return rows


# ----------------------------------------------------------------------------- model (torch imported lazily)
def build_net(mode, k, seconds=5, backend=None):
    """backend: Mamba implementation ('mamba_ssm' | 'torch' | None = BIMA_MAMBA_BACKEND / auto)."""
    from bima_ecgsr.models.backbone import BiMaSRNet, BEST_PARAMS as best_params
    return BiMaSRNet(target_channels=12, num_channels=k, fs_lr=500, fs_hr=500, seconds=seconds,
                     residual_global=True, residual_internal=True, use_subpixel=True, sr_type='spatial',
                     use_mamba=best_params['use_mamba'], use_diffusion=(mode == 'diffusion'),
                     n_mamba_layers=best_params['n_mamba_layers'], mamba_dim=best_params['dim'],
                     mamba_d_state=best_params['d_state'], mamba_version=best_params['version'],
                     n_mamba_blocks=best_params['n_mamba_blocks'], use_positional_encoding=False, backend=backend)


def make_wrapper():
    import torch, torch.nn as nn

    class LeadResidualSR(nn.Module):
        """BiMaSRNet + global residual of every input lead into its own output channel.
        Holds no parameters of its own: state_dict keys are those of BiMaSRNet prefixed by 'net.'; we save net."""

        def __init__(self, net, in_idx, mode):
            super().__init__()
            self.net, self.in_idx, self.mode = net, list(in_idx), mode

        def global_residual(self, inp):
            out = inp.new_zeros((inp.shape[0], 12, inp.shape[-1]))
            out[:, self.in_idx] = inp
            return out

        def forward(self, inp, x_t=None, t=None):
            if self.mode == 'diffusion':
                z, res = self.net.encoder_forward(torch.cat([x_t, inp], 1), t=t)
            else:
                z, res = self.net.encoder_forward(inp)
            sr = self.net.decoder_forward(z, residual_premamba=res, lr=None)  # no UNMASK_CHANNELS lookup
            return sr + self.global_residual(inp)

    return LeadResidualSR


def build_scheduler(T):
    from diffusers import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=T, prediction_type='sample', clip_sample=False, timestep_spacing='trailing',
                         beta_schedule='squaredcos_cap_v2', rescale_betas_zero_snr=True)


def predict_batch(model, inp, ddpm=None, steps=1, eta=0.0, gen=None):
    """Sees only the k input leads (and, for diffusion, its own noise)."""
    import torch
    from diffusers import DDIMScheduler
    if model.mode == 'regression':
        return model(inp)
    B, L = inp.shape[0], inp.shape[-1]
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(steps, device=inp.device)
    x = torch.randn((B, 12, L), device=inp.device, generator=gen)
    for t in ddim.timesteps:
        tb = torch.full((B,), int(t), device=inp.device, dtype=torch.long)
        x0 = model(inp, x, tb).float()
        x = ddim.step(x0, t, x, eta=eta, generator=gen).prev_sample
    return x


SAMPLERS = {'onestep': 1, 'ddim6': 6}


def samplers_for(mode):
    return {'': None} if mode == 'regression' else dict(SAMPLERS)


def predict_array(model, X, S, in_idx, bs, ddpm=None, steps=None, seed=0, amp=True, dev='cuda', n=None,
                  out_path=None, chunk=1024, replace_inputs=True):
    """Predict windows [0, n) of memmap X. Returns (pcc, mae, rmse) [n,12] and optionally writes float16 preds."""
    import torch
    n = X.shape[0] if n is None else n
    gen = torch.Generator(device=dev).manual_seed(seed)
    mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16, shape=(n, 12, X.shape[-1])) if out_path else None
    M = [np.zeros((n, 12), np.float32) for _ in range(3)]
    with torch.no_grad():
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            Z = torch.from_numpy(np.ascontiguousarray(X[c0:c1])); St = torch.from_numpy(S[c0:c1])
            outc = np.empty((c1 - c0, 12, X.shape[-1]), np.float16)
            for i in range(0, c1 - c0, bs):
                z = Z[i:i + bs].to(dev).float(); st = St[i:i + bs].to(dev)
                inp = norm_inputs_t(z, st, in_idx)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    p = predict_batch(model, inp, ddpm, steps=steps or 1, gen=gen)
                p = p.float()
                if replace_inputs:
                    p[:, in_idx] = z[:, in_idx]
                for m, v in zip(M, metrics_t(p, z)):
                    m[c0 + i:c0 + i + p.shape[0]] = v.cpu().numpy()
                outc[i:i + p.shape[0]] = p.cpu().numpy().astype(np.float16)
            if mm is not None:
                mm[c0:c1] = outc
    if mm is not None:
        mm.flush(); del mm
    return tuple(M)


# ----------------------------------------------------------------------------- baselines (numpy, CPU)
def _lag(x, L):
    """x [B,k,T] -> [B,T,k*(2L+1)] design (lags -L..L per input lead, zero padded)."""
    B, k, T = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (L, L)))
    return np.stack([xp[:, :, j:j + T] for j in range(2 * L + 1)], -1).transpose(0, 2, 1, 3).reshape(B, T, k * (2 * L + 1))


def fit_fir(in_idx, L=FIR_L, n_fit=FIR_NFIT, seed=FIR_SEED, chunk=50, ridge=1e-3, log=print):
    Xtr, Str = load_cohort('ptbxl_train')
    keep = train_keep_index()
    idx = np.sort(np.random.default_rng(seed).choice(keep, min(n_fit, len(keep)), replace=False))
    K = len(in_idx) * (2 * L + 1); AtA = np.zeros((K, K)); AtY = np.zeros((K, 12))
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        z = np.asarray(Xtr[j], np.float64)
        A = _lag(norm_inputs_np(z, Str[j], in_idx), L).reshape(-1, K)
        AtA += A.T @ A; AtY += A.T @ z.transpose(0, 2, 1).reshape(-1, 12)
    W = np.linalg.solve(AtA + ridge * np.eye(K), AtY)
    log(f'[fir] fit on {len(idx)} train windows (seed {seed}), K={K}, cond(AtA)={np.linalg.cond(AtA):.3e}')
    return W, idx


def einthoven_np(z, st):
    """Exact limb-lead derivation from destandardised I, II (mV): returns z-scored [B,4,T] for III, aVR, aVL, aVF."""
    I = z[:, 0] * st[:, 0, 1:2] + st[:, 0, 0:1]; II = z[:, 1] * st[:, 1, 1:2] + st[:, 1, 0:1]
    D = np.stack([II - I, -(I + II) / 2, I - II / 2, II - I / 2], 1)
    return (D - D.mean(-1, keepdims=True)) / (D.std(-1, keepdims=True) + 1e-8)


def run_baselines(args, in_idx, out, log):
    W, fit_idx = fit_fir(in_idx, log=log)
    np.savez(os.path.join(out, 'fir_weights.npz'), W=W, fit_window_index=fit_idx, in_idx=np.array(in_idx), L=FIR_L)
    methods = ['fir'] + (['einthoven_fir'] if (0 in in_idx and 1 in in_idx) else [])
    for coh in [c for c in args.cohorts.split(',') if c]:
        X, S = load_cohort(coh); n = args.n_test or X.shape[0]; n = min(n, X.shape[0])
        todo = [m for m in methods if not os.path.exists(os.path.join(out, f'done_{coh}_{m}.json'))]
        if not todo:
            log(f'[skip] {coh} done'); continue
        t0 = time.time()
        mms = {m: np.lib.format.open_memmap(os.path.join(out, f'pred_{coh}_{m}.partial.npy'), mode='w+', dtype=np.float16,
                                             shape=(n, 12, X.shape[-1])) for m in todo}
        M = {m: [np.zeros((n, 12), np.float32) for _ in range(3)] for m in todo}
        for c0 in range(0, n, 64):
            c1 = min(n, c0 + 64)
            z = np.asarray(X[c0:c1], np.float64); st = S[c0:c1].astype(np.float64)
            A = _lag(norm_inputs_np(z, st, in_idx), FIR_L)
            P = (A.reshape(-1, A.shape[-1]) @ W).reshape(c1 - c0, -1, 12).transpose(0, 2, 1)
            P[:, in_idx] = z[:, in_idx]
            for m in todo:
                Pm = P.copy()
                if m == 'einthoven_fir':
                    Pm[:, 2:6] = einthoven_np(z, st)
                for arr, v in zip(M[m], metrics_np(Pm, z)):
                    arr[c0:c1] = v
                mms[m][c0:c1] = Pm.astype(np.float16)
        for m in todo:
            mms[m].flush(); del mms[m]
            np.savez_compressed(os.path.join(out, f'metrics_{coh}_{m}.npz'), pcc=M[m][0], mae=M[m][1], rmse=M[m][2])
            os.replace(os.path.join(out, f'pred_{coh}_{m}.partial.npy'), os.path.join(out, f'pred_{coh}_{m}.npy'))
            json.dump(dict(n=n, sec=time.time() - t0), open(os.path.join(out, f'done_{coh}_{m}.json'), 'w'))
            g = group_rows(coh, m, *M[m], in_idx)
            log(f'[test] {coh} {m}: ' + ', '.join(f"{r['group']} PCC med {r['pcc_median']:.4f}" for r in g)
                + f' ({time.time()-t0:.0f} s, n={n})')
    write_summaries(out, in_idx)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['diffusion', 'regression', 'baselines'], required=True)
    ap.add_argument('--input-leads', required=True, help='comma-separated lead names, e.g. I,II,V2')
    ap.add_argument('--name', default=None)
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=32, help='effective batch size (optimizer step)')
    ap.add_argument('--accum', type=int, default=4, help='micro-batches per step (micro-batch = bs/accum)')
    ap.add_argument('--eval-bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--amp', action='store_true', help='bf16 autocast')
    ap.add_argument('--val-steps', type=int, default=6, help='DDIM steps for per-epoch validation (diffusion)')
    ap.add_argument('--n-val', type=int, default=1024, help='first n of val_select_1024 (1024 = all)')
    ap.add_argument('--n-test', type=int, default=0, help='0 = full cohorts; else first n windows of each cohort')
    ap.add_argument('--cohorts', default=','.join(COHORTS))
    ap.add_argument('--max-steps', type=int, default=0, help='stop each epoch after N steps (smoke test)')
    ap.add_argument('--exit-after-epoch', type=int, default=-1, help='testing: exit(3) after this epoch (simulated crash)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--resume', action='store_true', help='resume from <out>/ckpt.pt and skip finished test cohorts')
    ap.add_argument('--out-root', default=OUT_ROOT)
    args = ap.parse_args()

    in_idx = parse_leads(args.input_leads); k = len(in_idx); groups = lead_groups(in_idx)
    tag = '_'.join(LEADS[c] for c in in_idx)
    if args.name is None:
        args.name = {'regression': 'bima_sr_', 'diffusion': 'dibima_sr_', 'baselines': 'baselines_'}[args.mode] + tag
    out = os.path.join(args.out_root, args.name); os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, 'train.log'), 'a')

    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()

    P(f'[start] {time.strftime("%F %T")} mode={args.mode} input={[LEADS[c] for c in in_idx]} idx={in_idx} '
      f'predicted={[LEADS[c] for c in groups["all_pred"]]}')
    cohorts = [c for c in args.cohorts.split(',') if c]
    if args.mode == 'baselines':
        json.dump(vars(args) | dict(in_idx=in_idx, fir_L=FIR_L, fir_nfit=FIR_NFIT, fir_seed=FIR_SEED),
                  open(os.path.join(out, 'config.json'), 'w'), indent=1)
        run_baselines(args, in_idx, out, P); P('[done]', out); return

    import torch, torch.nn as nn
    dev = 'cuda'
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # data: train in RAM (float16, 2.1 GB), val subset in RAM, test cohorts streamed from memmaps
    Xtr_np, Str_np = load_cohort('ptbxl_train', mmap=False)
    keep = torch.from_numpy(train_keep_index())
    Xtr = torch.from_numpy(Xtr_np); Str = torch.from_numpy(Str_np)
    try:
        Xtr = Xtr.pin_memory()
    except Exception as e:  # noqa
        P(f'[warn] pin_memory failed: {e}')
    Xva, Sva = load_cohort('ptbxl_val')
    vsel = np.load(os.path.join(DATA, 'val_select_1024.npy'))[:args.n_val]
    Xv = torch.from_numpy(np.asarray(Xva[vsel])); Sv = torch.from_numpy(Sva[vsel])
    P(f'[data] train {tuple(Xtr.shape)} used {len(keep)} (excluded records {EXCLUDE_TRAIN_RECORDS}: '
      f'{Xtr.shape[0]-len(keep)} windows); val_select {tuple(Xv.shape)}')

    LeadResidualSR = make_wrapper()
    net = build_net(args.mode, k).to(dev)
    model = LeadResidualSR(net, in_idx, args.mode)
    ddpm = build_scheduler(args.T) if args.mode == 'diffusion' else None
    cfg = vars(args) | dict(in_idx=in_idx, predicted_idx=groups['all_pred'], n_params=sum(p.numel() for p in net.parameters()),
                            input_norm='x_mV=z*std+mean (input leads); remove per-lead mean; divide by s=sqrt(mean_j std_j^2)',
                            target='stored per-lead z-scores (12 leads)', excluded_train_records=EXCLUDE_TRAIN_RECORDS)
    from bima_ecgsr.models.backbone import BEST_PARAMS
    cfg['best_params'] = BEST_PARAMS
    if ddpm is not None:
        ab = ddpm.alphas_cumprod
        cfg['sqrt_alphabar_T-1'] = float(ab[-1].sqrt()); cfg['sqrt_alphabar_0'] = float(ab[0].sqrt())
        P(f"[sched] cosine+zSNR T={args.T} sqrt(alphabar_0)={cfg['sqrt_alphabar_0']:.4f} "
          f"sqrt(alphabar_T-1)={cfg['sqrt_alphabar_T-1']:.2e}")
    P(f"[model] mode={args.mode} params={cfg['n_params']/1e6:.3f}M first conv {tuple(net.encoder.encoder[0].weight.shape)}")
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1, default=str)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    spe = math.ceil(len(keep) / args.bs)
    if args.max_steps: spe = min(spe, args.max_steps)
    total = spe * args.epochs
    sched_lr = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * min(s, total) / total))) / args.lr)
    mse = nn.MSELoss()
    best, hist, ep0 = -1e9, [], 0
    ck = os.path.join(out, 'ckpt.pt')
    if args.resume and os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(c['net']); opt.load_state_dict(c['opt']); sched_lr.load_state_dict(c['sched'])
        best, hist, ep0 = c['best'], c['hist'], c['epoch'] + 1
        torch.set_rng_state(c['rng_cpu'].cpu()); torch.cuda.set_rng_state(c['rng_cuda'].cpu())  # map_location moved them
        P(f'[resume] from epoch {ep0} (best val PCC {best:.4f})')

    def val_score():
        model.eval(); pcc = np.zeros((Xv.shape[0], 12), np.float32); maes = np.zeros_like(pcc)
        gen = torch.Generator(device=dev).manual_seed(1234)
        with torch.no_grad():
            for i in range(0, Xv.shape[0], args.eval_bs):
                z = Xv[i:i + args.eval_bs].to(dev).float(); inp = norm_inputs_t(z, Sv[i:i + args.eval_bs].to(dev), in_idx)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                    p = predict_batch(model, inp, ddpm, steps=args.val_steps, gen=gen)
                a, b, _ = metrics_t(p.float(), z); pcc[i:i + len(z)] = a.cpu().numpy(); maes[i:i + len(z)] = b.cpu().numpy()
        cols = groups['all_pred']
        return dict(val_pcc_median=pooled_median_pcc(pcc, cols), val_pcc_median_limb=pooled_median_pcc(pcc, groups['limb_pred']),
                    val_pcc_median_prec=pooled_median_pcc(pcc, groups['prec_pred']), val_mae=float(maes[:, cols].mean()))

    for ep in range(ep0, args.epochs):
        model.train(); perm = keep[torch.randperm(len(keep))]; t0 = time.time(); losses = []
        torch.cuda.reset_peak_memory_stats(); t_ss = None
        for kk in range(spe):
            if kk == 5: torch.cuda.synchronize(); t_ss = time.time()
            idx = perm[kk * args.bs:(kk + 1) * args.bs]; mb = math.ceil(len(idx) / args.accum)
            opt.zero_grad(set_to_none=True); ltot = 0.0
            for j in range(0, len(idx), mb):
                bi = idx[j:j + mb]
                hr = Xtr[bi].to(dev, non_blocking=True).float(); st = Str[bi].to(dev, non_blocking=True)
                inp = norm_inputs_t(hr, st, in_idx)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                    if args.mode == 'diffusion':
                        t = torch.randint(0, args.T, (hr.shape[0],), device=dev)
                        x_t = ddpm.add_noise(hr, torch.randn_like(hr), t)
                        pred = model(inp, x_t, t)
                    else:
                        pred = model(inp)
                loss = mse(pred.float(), hr) * (hr.shape[0] / len(idx))
                loss.backward(); ltot += loss.item()
            if args.clip: nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step(); sched_lr.step(); losses.append(ltot)
            if kk % args.log_every == 0:
                P(f'  ep {ep} step {kk}/{spe} loss {np.mean(losses[-200:]):.4f} lr {sched_lr.get_last_lr()[0]:.2e} '
                  f'{(time.time()-t0)/(kk+1)*1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
        torch.cuda.synchronize(); tr_time = time.time() - t0
        ms_ss = (time.time() - t_ss) / (spe - 5) * 1000 if (t_ss and spe > 5) else tr_time / spe * 1000
        peak_tr = dict(alloc=torch.cuda.max_memory_allocated() / 2**30, reserved=torch.cuda.max_memory_reserved() / 2**30)
        t1 = time.time(); sv = val_score(); val_sec = time.time() - t1
        rec = dict(epoch=ep, train_loss=float(np.mean(losses)), **sv, epoch_min=tr_time / 60, val_sec=val_sec,
                   ms_per_step=ms_ss, peak_alloc_gib=peak_tr['alloc'], peak_reserved_gib=peak_tr['reserved'])
        hist.append(rec); json.dump(hist, open(os.path.join(out, 'history.json'), 'w'), indent=1)
        torch.save(net.state_dict(), os.path.join(out, 'last.pth'))
        if sv['val_pcc_median'] > best:
            best = sv['val_pcc_median']; torch.save(net.state_dict(), os.path.join(out, 'best.pth')); rec['best'] = True
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched_lr.state_dict(), best=best, hist=hist,
                        epoch=ep, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
        P(f"[epoch {ep}] loss {rec['train_loss']:.4f} val pooled-median PCC {sv['val_pcc_median']:.4f} "
          f"(limb {sv['val_pcc_median_limb']:.4f}, prec {sv['val_pcc_median_prec']:.4f}) MAE {sv['val_mae']:.4f} "
          f"{'*best*' if rec.get('best') else ''} ({tr_time/60:.1f} min train, {ms_ss:.0f} ms/step steady, "
          f"peak {peak_tr['alloc']:.2f}/{peak_tr['reserved']:.2f} GiB alloc/reserved, val {val_sec:.0f} s)")
        if ep == args.exit_after_epoch:
            P(f'[exit-after-epoch] simulated crash after epoch {ep}'); log_f.close(); sys.exit(3)

    # ---------------- test with the best checkpoint ----------------
    net.load_state_dict(torch.load(os.path.join(out, 'best.pth'), map_location=dev)); model.eval()
    timing = {}
    for coh in cohorts:
        X, S = load_cohort(coh); n = min(args.n_test or X.shape[0], X.shape[0])
        for sname, steps in samplers_for(args.mode).items():
            key = f'{coh}_{sname}' if sname else coh
            done = os.path.join(out, f'done_{key}.json')
            if args.resume and os.path.exists(done):
                P(f'[skip] {key} already done'); continue
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            tmp = os.path.join(out, f'pred_{key}.partial.npy')
            pcc, mae, rmse = predict_array(model, X, S, in_idx, args.eval_bs, ddpm, steps, seed=0, amp=args.amp, n=n, out_path=tmp)
            sec = time.time() - t0
            np.savez_compressed(os.path.join(out, f'metrics_{key}.npz'), pcc=pcc, mae=mae, rmse=rmse)
            os.replace(tmp, os.path.join(out, f'pred_{key}.npy'))
            timing[key] = dict(n=n, sec=sec, ms_per_window=sec / n * 1000, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30)
            json.dump(timing[key], open(done, 'w'))
            g = group_rows(coh, sname or 'regression', pcc, mae, rmse, in_idx)
            P(f'[test] {key}: ' + ', '.join(f"{r['group']} PCC med {r['pcc_median']:.4f} MAE {r['mae_mean']:.4f}" for r in g)
              + f' (n={n}, {sec/n*1000:.2f} ms/window, peak {timing[key]["peak_alloc_gib"]:.2f} GiB)')
    write_summaries(out, in_idx)
    P('[done]', out)


if __name__ == '__main__':
    main()
