"""Unit tests for bima_ecgsr.classification (CPU only; stdlib unittest). Run: python tests/test_classifier.py -v"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.classification import metrics as M  # noqa: E402
from bima_ecgsr.classification.classifier import LABELS, build_model, parse_leads  # noqa: E402
from bima_ecgsr.classification.predict import (aggregate_records, load_classifier, predict, record_labels,  # noqa: E402
                                               zscore_windows)


class TestModel(unittest.TestCase):
    def test_shape_and_params(self):
        m = build_model(12).eval()
        self.assertEqual(sum(p.numel() for p in m.parameters()), 3364369)   # = judge12/summary.json n_params
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.randn(2, 12, 2500)).shape), (2, len(LABELS)))
        self.assertEqual(len(LABELS), 17)

    def test_parse_leads(self):
        self.assertEqual(parse_leads("all"), list(range(12)))
        self.assertEqual(parse_leads("V2,I,II"), [0, 1, 7])
        with self.assertRaises(ValueError):
            parse_leads("II,II")


class TestInputHandling(unittest.TestCase):
    def test_zscore_and_flat_guard(self):
        x = torch.randn(3, 12, 2500) * 0.7 + 2.0
        z = zscore_windows(x)
        self.assertLess(z.mean(-1).abs().max().item(), 1e-5)
        self.assertLess((z.pow(2).mean(-1).sqrt() - 1).abs().max().item(), 1e-5)
        x[0, 4] = 3.0                                  # flat lead -> centred only, finite
        z = zscore_windows(x)
        self.assertTrue(torch.isfinite(z).all()); self.assertEqual(z[0, 4].abs().max().item(), 0.0)

    def test_predict_scale_invariant(self):
        torch.manual_seed(0); m = build_model(12).eval()
        X = np.random.default_rng(0).normal(size=(4, 12, 2500)).astype(np.float32)
        a = predict(m, X, batch_size=2, device="cpu"); b = predict(m, X * 3.5 + 0.2, batch_size=4, device="cpu")
        self.assertLess(np.abs(a - b).max(), 1e-4)


class TestMetrics(unittest.TestCase):
    def test_auroc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300); s = np.round(rng.normal(size=300) + y, 1)   # with ties
        self.assertAlmostEqual(M.auroc(y, s), roc_auc_score(y, s), places=12)

    def test_weighted_equals_resampled(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 80); s = rng.normal(size=80) + y
        C = M.bootstrap_counts(80, 5, seed=3)
        self.assertEqual(C.shape, (5, 80)); self.assertTrue((C.sum(1) == 80).all())
        w = M.weighted_auroc_many(y, s, C)
        from sklearn.metrics import roc_auc_score
        for b in range(5):
            idx = np.repeat(np.arange(80), C[b])
            self.assertAlmostEqual(w[b], roc_auc_score(y[idx], s[idx]), places=10)

    def test_bootstrap_counts_deterministic(self):
        np.testing.assert_array_equal(M.bootstrap_counts(50, 3, 0), M.bootstrap_counts(50, 3, 0))


class TestAggregation(unittest.TestCase):
    def test_record_mean_and_labels(self):
        meta = pd.DataFrame(dict(cohort="c", record_id=[5, 5, 7, 9, 9], NORM=[1, 1, 0, 0, 0], has_superclass=[1, 1, 1, 0, 0]))
        p = np.arange(5 * 17, dtype=np.float32).reshape(5, 17)
        keys, pr, codes = aggregate_records(p, meta)
        self.assertEqual(keys.record_id.tolist(), [5, 7, 9])
        np.testing.assert_allclose(pr[0], p[:2].mean(0)); np.testing.assert_allclose(pr[1], p[2])
        Y = record_labels(meta, codes=codes)
        self.assertEqual(Y[:, LABELS.index("NORM")].tolist(), [1, 0, -1])       # has_superclass == 0 -> unknown
        self.assertTrue((Y[:, LABELS.index("AFIB")] == -1).all())              # absent column -> unknown


@unittest.skipUnless(os.path.isfile(CFG.judge_ckpt()), "judge checkpoint not found")
class TestJudgeCheckpoint(unittest.TestCase):
    def test_load_strict(self):
        m, info = load_classifier(CFG.judge_ckpt(), device="cpu")
        self.assertEqual(info["lead_idx"], list(range(12)))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 3364369)


if __name__ == "__main__":
    unittest.main()
