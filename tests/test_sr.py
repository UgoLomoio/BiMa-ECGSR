"""Unit tests for bima_ecgsr/training/train_sr.py (needs the GPU: mamba_ssm kernels).
  python tests/test_sr.py
1. global residual: for every config, model(inp) - decoder_forward(lr=None) == inp in the input-lead channels, 0 elsewhere,
   and channel j of the input lands in output channel in_idx[j] (distinct constants per input channel).
2. for input [II], the wrapper equals the backbone's own lead-II path (regression: decoder_forward(..., lr=lr);
   diffusion: forward_diffusion) that uses backbone.UNMASK_CHANNELS {1: [1]}.
3. input normalisation: k = 1 gives the stored z-score; k > 1 keeps inter-lead amplitude ratios (std ratio = std_mV ratio),
   common scale s = sqrt(mean std^2); torch and numpy versions agree.
4. Einthoven/Goldberger derivation from destandardised I, II reproduces stored III, aVR, aVL, aVF (PTB-XL test).
5. lead parsing / predicted-lead groups.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from bima_ecgsr.training import train_sr as v2  # noqa: E402

CONFIGS = ['II', 'I', 'I,II', 'I,II,V1', 'I,II,V2', 'I,II,V1,V4']
torch.manual_seed(0); dev = 'cuda'
X, S = v2.load_cohort('ptbxl_test')
z = torch.from_numpy(np.array(X[:4])).float().to(dev); st = torch.from_numpy(S[:4]).to(dev)
W = v2.make_wrapper(); n_ok = 0

# 5. parsing
assert v2.parse_leads('V2,i,II') == [0, 1, 7]
g = v2.lead_groups(v2.parse_leads('I,II,V2'))
assert g['limb_pred'] == [2, 3, 4, 5] and g['prec_pred'] == [6, 8, 9, 10, 11] and len(g['all_pred']) == 9
n_ok += 1

for cfg in CONFIGS:
    idx = v2.parse_leads(cfg); k = len(idx)
    for mode in ['regression', 'diffusion']:
        net = v2.build_net(mode, k).to(dev).eval(); m = W(net, idx, mode)
        inp = v2.norm_inputs_t(z, st, idx)
        xt = torch.randn(4, 12, 2500, device=dev); t = torch.full((4,), 19, device=dev, dtype=torch.long)
        with torch.no_grad():
            enc_in = torch.cat([xt, inp], 1) if mode == 'diffusion' else inp
            zz, res = net.encoder_forward(enc_in, t=t if mode == 'diffusion' else None)
            base = net.decoder_forward(zz, residual_premamba=res, lr=None)
            full = m(inp, xt, t) if mode == 'diffusion' else m(inp)
            d = full - base
            other = [c for c in range(12) if c not in idx]
            tol = 1e-5 + 4 * torch.finfo(torch.float32).eps * base.abs().max().item()  # float32 rounding of base + inp
            assert (d[:, idx] - inp).abs().max().item() <= tol, (cfg, mode, 'residual value')
            assert d[:, other].abs().max().item() == 0, (cfg, mode, 'residual spills into non-input channels')
            const = torch.arange(1, k + 1, device=dev, dtype=torch.float)[None, :, None].expand(4, k, 2500)
            r = m.global_residual(const)
            for j, c in enumerate(idx):
                assert torch.all(r[:, c] == j + 1)
            assert torch.all(r[:, other] == 0)
            if cfg == 'II':  # 2. equivalence with the backbone's lead-II path (UNMASK_CHANNELS {1:[1]})
                ref = net.forward_diffusion(xt, t, inp) if mode == 'diffusion' else \
                    net.decoder_forward(*net.encoder_forward(inp), lr=inp)
                assert (full - ref).abs().max().item() <= tol, ('II original path', mode, (full - ref).abs().max().item())
        n_ok += 1
    # 3. normalisation
    a = v2.norm_inputs_t(z, st, idx).cpu().numpy(); b = v2.norm_inputs_np(z.cpu().numpy(), st.cpu().numpy(), idx)
    assert np.allclose(a, b, atol=1e-4)
    sd = S[:4][:, idx, 1].astype(np.float64)
    s = np.sqrt((sd ** 2).mean(1))
    assert np.allclose(b.std(-1), sd / s[:, None], rtol=2e-3), cfg
    if k == 1:
        assert np.allclose(b[:, 0], z[:, idx[0]].cpu().numpy(), atol=2e-3)
    n_ok += 1

# 4. Einthoven
Z = np.asarray(X[:256], np.float64); D = v2.einthoven_np(Z, S[:256].astype(np.float64))
pcc, mae, _ = v2.metrics_np(D, Z[:, 2:6])
print('[einthoven] PTB-XL test first 256: median PCC per lead III,aVR,aVL,aVF', np.round(np.median(pcc, 0), 5),
      'median MAE', np.round(np.median(mae, 0), 4))
assert np.median(pcc) > 0.99
n_ok += 1
print(f'ALL {n_ok} TEST GROUPS PASSED')
