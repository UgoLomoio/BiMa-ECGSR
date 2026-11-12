"""Tests for the pure-PyTorch Mamba backend (bima_ecgsr.models.mamba_torch). Run: python tests/test_mamba_torch.py -v

1. associative (chunked Hillis-Steele) scan and the numba kernel equal the sequential recursion (CPU).
2. the torch backend has exactly the state-dict keys / shapes of the mamba_ssm backend (needs mamba_ssm importable).
3. with trained weights (bima_sr_II, dibima_sr_II best.pth), torch backend on CPU (float32) == mamba_ssm backend on the
   GPU (float32, no autocast, TF32 off) on 2 PTB-XL test windows (needs CUDA + > 4000 MiB free GPU memory).
"""
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.models import mamba_torch as MT  # noqa: E402
from bima_ecgsr.training import train_sr as T  # noqa: E402


def _free_mib():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return 0


def _has_mamba_ssm():
    try:
        import mamba_ssm  # noqa: F401
        return True
    except Exception:
        return False


GPU_OK = torch.cuda.is_available() and _free_mib() > 4000 and _has_mamba_ssm()


class TestScan(unittest.TestCase):
    def _inputs(self, L=700):
        g = torch.Generator().manual_seed(0)
        r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
        return r(2, 24, L), r(2, 24, L), -torch.exp(r(24, 8) * 0.5), r(2, 8, L), r(2, 8, L), r(24), r(2, 24, L), r(24)

    def test_scans_equal_sequential(self):
        u, dl, A, B, C, D, z, db = self._inputs()
        with torch.no_grad():
            ref = MT.selective_scan(u, dl, A, B, C, D, z=z, delta_bias=db, delta_softplus=True, sequential=True)
            for impl in ("torch", "numba"):
                MT.SCAN_IMPL = impl
                for ch in (1, 7, 64, 1024):
                    MT.SCAN_CHUNK = ch
                    y = MT.selective_scan(u, dl, A, B, C, D, z=z, delta_bias=db, delta_softplus=True)
                    self.assertLess((y - ref).abs().max().item(), 1e-5 * (1 + ref.abs().max().item()), (impl, ch))
        MT.SCAN_IMPL, MT.SCAN_CHUNK = "auto", 64

    @unittest.skipUnless(_has_mamba_ssm(), "mamba_ssm not installed")
    def test_state_dict_compatible(self):
        for mode in ("regression", "diffusion"):
            a = T.build_net(mode, 1, backend="torch").state_dict()
            b = T.build_net(mode, 1, backend="mamba_ssm").state_dict()
            self.assertEqual(list(a), list(b))
            self.assertTrue(all(a[k].shape == b[k].shape for k in a))


@unittest.skipUnless(GPU_OK, "needs CUDA, mamba_ssm and > 4000 MiB free GPU memory")
class TestTrainedWeights(unittest.TestCase):
    def test_torch_cpu_equals_mamba_ssm_gpu_fp32(self):
        torch.backends.cudnn.allow_tf32 = False          # cuDNN convolutions default to TF32 on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = False
        X, S = T.load_cohort("ptbxl_test")
        z = torch.from_numpy(np.asarray(X[:2], np.float32)); st = torch.from_numpy(S[:2])
        inp = T.norm_inputs_t(z, st, [1])
        noise = torch.randn(2, 12, 2500, generator=torch.Generator().manual_seed(0))
        Wr = T.make_wrapper()
        for mode, run in (("regression", "bima_sr_II"), ("diffusion", "dibima_sr_II")):
            p = os.path.join(CFG.RUNS_DIR, run, "best.pth")
            if not os.path.exists(p):
                continue
            sd = torch.load(p, map_location="cpu", weights_only=True)
            outs = {}
            for be, dev in (("mamba_ssm", "cuda"), ("torch", "cpu")):
                net = T.build_net(mode, 1, backend=be); net.load_state_dict(sd, strict=True)
                m = Wr(net.to(dev), [1], mode).eval()
                with torch.no_grad():
                    args = (inp.to(dev),) if mode == "regression" else \
                        (inp.to(dev), noise.to(dev), torch.full((2,), 19, dtype=torch.long, device=dev))
                    outs[be] = m(*args).float().cpu()
            d = (outs["torch"] - outs["mamba_ssm"]).abs().max().item()
            print(f"  {run}: max |torch CPU - mamba_ssm GPU| (float32) = {d:.2e} (output max {outs['torch'].abs().max():.2f})")
            self.assertLess(d, 1e-4)


if __name__ == "__main__":
    unittest.main()
