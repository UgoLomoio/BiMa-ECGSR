"""Unit tests for bima_ecgsr/data/pipeline.py (stdlib unittest; run from the repo root: python tests/test_pipeline.py -v).

Tests that compare against the built dataset read BIMA_DATA_DIR (bima_ecgsr.config.DATA_DIR) and are
skipped if it has not been built yet. Raw-record tests read the raw cohorts under BIMA_RAW_DIR.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
import wfdb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bima_ecgsr import config as CFG  # noqa: E402
from bima_ecgsr.data import pipeline as P  # noqa: E402

SRC = Path(CFG.RAW_DIR)
CANONICAL = REPO / "bima_ecgsr" / "data" / "constants.py"
GEORGIA_REC = Path(CFG.GEORGIA_DIR) / "g1" / "E00001"
CHINA_REC = Path(CFG.CHINA_DIR) / "WFDBRecords" / "01" / "010" / "JS00001"
PTB_REC = Path(CFG.PTBXL_DIR) / "records500" / "00000" / "00001_hr"
OUT = Path(CFG.DATA_DIR)
ROOTS = {"ptbxl": Path(CFG.PTBXL_DIR), "georgia": Path(CFG.GEORGIA_DIR), "china": Path(CFG.CHINA_DIR) / "WFDBRecords"}
F16_RTOL = 2.0 ** -11          # half-ULP relative rounding error of float16 (normal range)
F16_ATOL = 2.0 ** -25          # subnormal half-ULP


class TestLeadOrder(unittest.TestCase):
    def _check(self, rec):
        x, info, _ = P.read_record(rec)
        sig, fields = wfdb.rdsamp(str(rec))
        self.assertEqual(x.shape, (12, sig.shape[0]))
        for i, lead in enumerate(P.LEADS):
            j = [n.upper() for n in fields["sig_name"]].index(lead.upper())
            np.testing.assert_array_equal(x[i], sig[:, j])
        return fields["sig_name"]

    def test_georgia_header(self):
        names = self._check(GEORGIA_REC)
        print("\n  Georgia header leads:", names)

    def test_china_header(self):
        names = self._check(CHINA_REC)
        print("\n  China header leads:", names)

    def test_ptbxl_header_uppercase_names(self):
        self._check(PTB_REC)

    def test_permuted_header_is_reordered(self):
        rng = np.random.default_rng(1)
        sig = rng.normal(size=(5000, 12)).round(3)
        perm = rng.permutation(12)
        names = [P.LEADS[k].upper() if k % 2 else P.LEADS[k] for k in perm]
        with tempfile.TemporaryDirectory() as d:
            wfdb.wrsamp("perm", fs=500, units=["mV"] * 12, sig_name=names, p_signal=sig[:, perm],
                        fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=d)
            x, _, _ = P.read_record(Path(d) / "perm")
        np.testing.assert_allclose(x, sig.T, atol=1e-3)

    def test_missing_lead_raises(self):
        with self.assertRaises(P.RecordError) as cm:
            P.canonical_lead_index(list(P.LEADS[1:]))
        self.assertEqual(cm.exception.reason, "missing_lead")


class TestFilter(unittest.TestCase):
    def test_magnitude_response(self):
        f = np.array([0.5, 10, 40, 50, 60])
        _, h = ss.sosfreqz(P.SOS, worN=f, fs=P.FS)
        g = np.abs(h) ** 2      # sosfiltfilt applies the filter twice
        print("\n  |H|^2 (sosfiltfilt) at", dict(zip(f.tolist(), np.round(g, 4).tolist())),
              "dB", np.round(20 * np.log10(g), 2).tolist())
        self.assertAlmostEqual(g[0], 0.5, places=3)
        self.assertAlmostEqual(g[1], 1.0, places=3)
        self.assertAlmostEqual(g[2], 0.5, places=3)
        self.assertLess(g[3], 0.2)
        self.assertLess(g[4], 0.07)

    def test_empirical_sines_zero_phase(self):
        t = np.arange(P.N_SAMPLES) / P.FS
        mid = slice(1000, 4000)
        for f0, expect in ((10, 1.0), (0.5, 0.5), (40, 0.5), (50, None), (60, None)):
            y = P.bandpass(np.sin(2 * np.pi * f0 * t)[None])[0]
            gain = np.sqrt(2) * y[mid].std()
            _, h = ss.sosfreqz(P.SOS, worN=[f0], fs=P.FS)
            self.assertAlmostEqual(gain, np.abs(h[0]) ** 2, delta=0.02 if f0 > 1 else 0.05)
            if expect:
                self.assertAlmostEqual(gain, expect, delta=0.05)
        y = P.bandpass(np.sin(2 * np.pi * 10 * t)[None])[0]
        lag = np.argmax(np.correlate(y[mid], np.sin(2 * np.pi * 10 * t)[mid], "full")) - (mid.stop - mid.start - 1)
        self.assertEqual(lag, 0)


class TestWindowsAndZscore(unittest.TestCase):
    def test_zscore_stats_real_records(self):
        for rec in (GEORGIA_REC, CHINA_REC, PTB_REC):
            r = P.process_record(rec)
            self.assertEqual(r["windows"].shape, (2, 12, 2500))
            for k in range(2):
                if r["reasons"][k] is None:
                    w = r["windows"][k].astype(np.float64)
                    np.testing.assert_allclose(w.mean(-1), 0, atol=1e-5)
                    np.testing.assert_allclose(w.std(-1), 1, atol=1e-5)
                    s = P.to_store(r["windows"][k]).astype(np.float64)
                    np.testing.assert_allclose(s.mean(-1), 0, atol=2e-3)
                    np.testing.assert_allclose(s.std(-1), 1, atol=2e-3)

    def test_stats_invert_zscore_in_memory(self):
        for rec in (GEORGIA_REC, CHINA_REC, PTB_REC):
            r = P.process_record(rec)
            self.assertEqual(r["stats"].shape, (2, 12, 2))
            self.assertEqual(r["stats"].dtype, np.float32)
            for k in range(2):
                if r["reasons"][k] is None:
                    f = r["filtered"][k]
                    np.testing.assert_array_equal(r["stats"][k], P.window_stats(f.mean(-1), f.std(-1)))
                    back = P.destandardize(r["windows"][k], r["stats"][k]).astype(np.float64)
                    np.testing.assert_allclose(back, f, rtol=0, atol=1e-5 * (np.abs(f).max() + 1))

    def test_halves_are_contiguous_non_overlapping(self):
        x = np.tile(np.arange(6000, dtype=float), (12, 1))
        w = P.split_windows(P.select_first_10s(x))
        self.assertEqual(w[0, 0, 0], 0)
        self.assertEqual(w[0, 0, -1], 2499)
        self.assertEqual(w[1, 0, 0], 2500)
        self.assertEqual(w[1, 0, -1], 4999)

    def test_flat_nan_short(self):
        rng = np.random.default_rng(0)
        x = rng.normal(scale=0.3, size=(12, 5000))
        x[8] = 1.234                 # V3 flat (constant offset) over the whole record
        r = P.process_signal(x)
        self.assertEqual(r["reasons"], ["flat_lead:V3"] * 2)
        self.assertTrue(np.isnan(r["windows"]).all())
        self.assertIsNone(P.process_signal(np.random.default_rng(1).normal(scale=0.3, size=(12, 5000)))["reasons"][0])
        # note: a lead flat in only one raw half can pass QC after filtering, because the zero-phase
        # band-pass on the full 10 s spreads ringing from the other half (filter-before-split by design)
        x2 = x.copy()
        x2[0, 10] = np.nan
        self.assertEqual(P.process_signal(x2)["reasons"], ["nonfinite_raw"] * 2)
        with self.assertRaises(P.RecordError) as cm:
            P.process_signal(x[:, :2500])
        self.assertEqual(cm.exception.reason, "short_record")


class TestLabels(unittest.TestCase):
    def test_constants_match_canonical(self):
        spec = importlib.util.spec_from_file_location("canon_constants", CANONICAL)
        c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c)
        for name in ("LEADS", "BROAD", "FINE", "SNOMED", "SNOMED_ALIASES", "AMBIGUOUS_FOR_TARGET"):
            self.assertEqual(getattr(P, name), getattr(c, name), name)

    def test_external_rules(self):
        lab = P.external_panel_labels
        self.assertEqual(lab(["713427006"])["CRBBB"], 1)
        self.assertEqual(lab(["59118001"])["CRBBB"], 1)
        self.assertEqual(lab(["713426002"])["CRBBB"], 0)       # incomplete RBBB is not CRBBB
        self.assertEqual(lab(["733534002"])["CLBBB"], 1)
        self.assertEqual(lab(["164909002"])["CLBBB"], 1)
        self.assertEqual(lab(["251120003"])["CLBBB"], 0)       # incomplete LBBB
        self.assertEqual(lab(["427172004"])["PVC"], 1)
        self.assertEqual(lab(["164890007"])["AFIB"], -1)       # China flutter code ambiguous for AFIB
        self.assertEqual(lab(["164890007"])["AFLT"], 1)
        self.assertEqual(lab(["195080001"])["AFLT"], -1)
        self.assertEqual(lab(["63593006"])["PAC"], -1)
        self.assertEqual(lab(["164889003", "164890007"])["AFIB"], 1)   # positive alias wins
        self.assertTrue(all(v == -1 for v in lab([]).values()))
        self.assertTrue(all(v == 0 for v in lab(["426783006"]).values()))

    def test_ptbxl_labels(self):
        db = pd.read_csv(Path(CFG.PTBXL_DIR) / "ptbxl_database.csv", index_col="ecg_id")
        st = pd.read_csv(Path(CFG.PTBXL_DIR) / "scp_statements.csv", index_col=0)
        codes = P.ptbxl_codes(db.loc[1].scp_codes)          # {'NORM': 100, 'LVOLT': 0, 'SR': 0}
        self.assertEqual(P.ptbxl_superclass(codes, st), {"NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0})
        sup = P.ptbxl_superclass({"IMI": 0.0, "CLBBB": 100.0}, st)    # likelihood 0 still counts
        self.assertEqual((sup["MI"], sup["CD"], sup["NORM"]), (1, 1, 0))
        pl = P.ptbxl_panel_labels({"CRBBB": 100.0, "AFIB": 0.0, "IRBBB": 100.0})
        self.assertEqual((pl["CRBBB"], pl["AFIB"], pl["CLBBB"]), (1, 1, 0))

    def test_parse_comments(self):
        d = P.parse_comments(["Age: 83", "Sex: Male", "Dx: 164889003,59118001", "Rx: Unknown"])
        self.assertEqual((d["age"], d["sex"], d["dx"]), (83.0, "M", ["164889003", "59118001"]))
        self.assertTrue(np.isnan(P.parse_comments(["Age: NaN"])["age"]))


@unittest.skipUnless((OUT / "qc_report.json").is_file(), f"built dataset not found in {OUT}")
class TestStoredDataset(unittest.TestCase):
    FILES = {"ptbxl": ("X_ptbxl_test.npy", "meta_ptbxl_test.csv"), "georgia": ("X_georgia.npy", "meta_georgia.csv"),
             "china": ("X_china.npy", "meta_china.csv")}

    def _sample(self, cohort, n=5):
        xf, mf = self.FILES[cohort]
        X = np.load(OUT / xf, mmap_mode="r")
        meta = pd.read_csv(OUT / mf, dtype={"record_id": str, "dx_codes": str})
        self.assertEqual(len(meta), len(X))
        recs = meta.drop_duplicates("record_id").sample(n, random_state=0)
        return X, meta, recs

    def test_float32_vs_stored_float16(self):
        for cohort in self.FILES:
            X, meta, recs = self._sample(cohort)
            for _, rec in recs.iterrows():
                r = P.process_record(ROOTS[cohort] / rec.source_path)
                for _, w in meta[meta.record_id == rec.record_id].iterrows():
                    x32 = r["windows"][w.half]
                    stored = np.asarray(X[w.window_index])
                    np.testing.assert_array_equal(P.to_store(x32), stored)          # bit-exact cast
                    err = np.abs(stored.astype(np.float32) - x32)
                    self.assertTrue((err <= F16_RTOL * np.abs(x32) + F16_ATOL).all(), cohort)

    def test_idempotent_rebuild_5_records(self):
        X, meta, recs = self._sample("china")
        _, _, recs_g = self._sample("georgia", 2)
        for root, rec in [(ROOTS["china"], r) for _, r in recs.iterrows()] + \
                         [(ROOTS["georgia"], r) for _, r in recs_g.iterrows()]:
            a = P.process_record(root / rec.source_path)["windows"]
            b = P.process_record(root / rec.source_path)["windows"]
            self.assertEqual(P.to_store(a).tobytes(), P.to_store(b).tobytes())

    def test_meta_labels_match_source(self):
        for cohort in ("georgia", "china"):
            _, meta, recs = self._sample(cohort)
            for _, rec in recs.iterrows():
                h = wfdb.rdheader(str(ROOTS[cohort] / rec.source_path))
                d = P.parse_comments(h.comments)
                expect = P.external_panel_labels(d["dx"])
                self.assertEqual({k: int(rec[k]) for k in P.PANEL}, expect)
                self.assertEqual(Path(rec.source_path).name, rec.record_id)

    def test_val_select(self):
        v = np.load(OUT / "val_select_1024.npy")
        n = len(np.load(OUT / "X_ptbxl_val.npy", mmap_mode="r"))
        self.assertEqual(len(v), min(1024, n))
        self.assertTrue((np.diff(v) > 0).all() and v.min() >= 0 and v.max() < n)
        np.testing.assert_array_equal(v, np.sort(np.random.default_rng(0).choice(n, min(1024, n), replace=False)))

    STATS = {"ptbxl": "STATS_ptbxl_test.npy", "georgia": "STATS_georgia.npy", "china": "STATS_china.npy"}

    @staticmethod
    def _f16_bound(z, stats):
        """Per-sample bound on |destandardize(stored) - band-passed| from float16 z and float32 stats."""
        z = np.abs(np.asarray(z, np.float64))
        mean, std = np.abs(stats[..., 0:1].astype(np.float64)), stats[..., 1:2].astype(np.float64)
        f32 = 4 * 2.0 ** -24 * (mean + std * (z + 1))           # float32 rounding of stats and of the inversion
        return std * (F16_RTOL * z + F16_ATOL) + f32

    def test_stats_files_aligned(self):
        for xf in sorted(OUT.glob("X_*.npy")):
            X = np.load(xf, mmap_mode="r")
            S = np.load(OUT / xf.name.replace("X_", "STATS_", 1))
            self.assertEqual(S.shape, (len(X), 12, 2), xf.name)
            self.assertEqual(S.dtype, np.float32)
            self.assertTrue(np.isfinite(S).all())
            self.assertTrue((S[..., 1] >= P.FLAT_STD_MV).all())

    def test_destandardize_reproduces_bandpassed(self):
        for cohort in self.FILES:
            X, meta, recs = self._sample(cohort)
            S = np.load(OUT / self.STATS[cohort], mmap_mode="r")
            for _, rec in recs.iterrows():
                r = P.process_record(ROOTS[cohort] / rec.source_path)
                for _, w in meta[meta.record_id == rec.record_id].iterrows():
                    z, st = np.asarray(X[w.window_index]), np.asarray(S[w.window_index])
                    np.testing.assert_array_equal(st, r["stats"][w.half])
                    err = np.abs(P.destandardize(z, st).astype(np.float64) - r["filtered"][w.half])
                    self.assertTrue((err <= self._f16_bound(z, st)).all(), cohort)

    def test_einthoven_ptbxl_from_stats(self):
        """III = II - I survives the linear band-pass; reconstruct from z + stats (float16 error only)."""
        X, meta, _ = self._sample("ptbxl")
        S = np.load(OUT / self.STATS["ptbxl"], mmap_mode="r")
        rows = meta.sample(8, random_state=3)
        worst = []
        for _, w in rows.iterrows():
            z, st = np.asarray(X[w.window_index]), np.asarray(S[w.window_index])
            mv = P.destandardize(z, st).astype(np.float64)
            resid = mv[2] - (mv[1] - mv[0])
            bound = self._f16_bound(z, st)[:3].sum(0)
            f = P.process_record(ROOTS["ptbxl"] / w.source_path)["filtered"][w.half]
            resid_true = f[2] - (f[1] - f[0])              # data residual (1 uV ADC quantisation)
            self.assertTrue((np.abs(resid - resid_true) <= bound).all())
            self.assertLessEqual(np.abs(resid_true).max(), 0.0015)
            self.assertTrue((np.abs(resid) <= 0.0015 + bound).all())
            zres = np.abs(z[2].astype(np.float64) - (z[1].astype(np.float64) - z[0])).max()
            worst.append((np.abs(resid).max(), bound.max(), np.abs(resid_true).max(), zres))
        w = np.array(worst)
        print(f"\n  Einthoven PTB-XL (8 windows): max|III-(II-I)| from stats {w[:, 0].max() * 1000:.2f} uV "
              f"(float16 bound <= {w[:, 1].max() * 1000:.2f} uV, float64 band-passed {w[:, 2].max() * 1000:.2f} uV); "
              f"same relation on z-scores alone: max residual {w[:, 3].min():.2f}-{w[:, 3].max():.2f} SD")

    def test_ptbxl_splits_patient_disjoint(self):
        pats = {s: set(pd.read_csv(OUT / f"meta_ptbxl_{s}.csv", usecols=["patient_id"]).patient_id)
                for s in ("train", "val", "test")}
        self.assertFalse(pats["train"] & pats["val"] or pats["train"] & pats["test"] or pats["val"] & pats["test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
