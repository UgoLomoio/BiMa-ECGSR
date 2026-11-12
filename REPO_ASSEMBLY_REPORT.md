# BiMa-ECGSR — repository assembly report

Assembled 2026-11-11 from the earlier project tree (`<OLD>`) (copied, never moved or edited). No git repository was initialised,
nothing was committed, pushed or uploaded, and no Space or HF repository was created (`space/deploy.py` was only run in
its default dry-run mode). The training queue was not touched; `scripts/queue_sr.sh` / `queue_sr_extra.sh` were never
launched. Run directories under `/media/gtimpano/Data/UGO/dibima_sr12_v2_runs` and `.../dibima_cls_v2_runs` were only read.

## 1. Tree

```
BiMa-ECGSR/
├── bima_ecgsr/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── predict.py
│   │   └── train_classifier.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── constants.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── cls_recon.py
│   │   └── sr_stats.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py
│   │   ├── mamba_backend.py
│   │   ├── mamba_torch.py
│   │   └── scan_numba.py
│   ├── plotting/
│   │   ├── __init__.py
│   │   └── style.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_sr.py
│   ├── __init__.py
│   └── config.py
├── docs/
│   ├── verification/
│   │   ├── equivalence.json
│   │   ├── equivalence_warmup.json
│   │   ├── space_smoke_results_threads2.csv
│   │   └── test_log.txt
│   ├── CLASSIFIER.md
│   ├── PIPELINE.md
│   └── space_app_render.png
├── figures/
│   ├── captions/
│   │   ├── Figure1.md
│   │   └── Graphical_Abstract.md
│   ├── Figure1.pdf
│   ├── Figure1.png
│   ├── figure1.py
│   ├── Figure1.svg
│   ├── Graphical_Abstract.pdf
│   ├── Graphical_Abstract.png
│   ├── graphical_abstract.py
│   └── Graphical_Abstract.svg
├── results/
│   ├── classification/
│   │   ├── auroc_true_records.csv
│   │   └── auroc_true_summary.csv
│   ├── reconstruction_classification/
│   │   ├── auroc_recon_deltas_II.csv
│   │   └── auroc_recon_II.csv
│   ├── noise_schedule.csv
│   └── sr_II_overall.csv
├── scripts/
│   ├── queue_sr.sh
│   ├── queue_sr_extra.sh
│   ├── run_all.sh
│   ├── run_classifier.sh
│   └── smoke_sr.sh
├── source_data/
│   ├── Figure1_a_pipeline.csv
│   ├── Figure1_a_splits.csv
│   ├── Figure1_b_input_configs.csv
│   ├── Figure1_c_architecture.csv
│   ├── Figure1_c_noise_schedule.csv
│   ├── Figure1_d_evaluation.csv
│   ├── Figure1_param_counts.csv
│   ├── Graphical_Abstract_auroc.csv
│   ├── Graphical_Abstract_example.csv
│   └── Graphical_Abstract_fidelity.csv
├── space/
│   ├── examples/  (14 files: 6 PTB-XL WFDB records .hea/.dat, examples.json, ATTRIBUTION.md)
│   ├── tools/
│   │   ├── app_render.png
│   │   ├── render_example.py
│   │   ├── smoke_app.py
│   │   ├── smoke_results_threads2.csv
│   │   └── stage_weights.py
│   ├── app.py
│   ├── deploy.py
│   ├── engine.py
│   ├── LICENSE
│   ├── README.md
│   └── requirements.txt
├── tests/
│   ├── test_classifier.py
│   ├── test_mamba_torch.py
│   ├── test_pipeline.py
│   └── test_sr.py
├── tools/
│   └── verify_equivalence.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```
(`space/tools/app_render.png` and `space/tools/smoke_results_threads2.csv` are local tool outputs, git-ignored; curated
copies are in `docs/`.)

## 2. Provenance (old -> new)

`<OLD>` = the earlier project repository and `<PAPER>` = its paper sub-folder (full paths are in the delegation record and are not repeated here so that this repository stays free of the earlier project's name); other paths are relative to `/home/gtimpano/Scrivania/UGO/` or absolute. Diff = changed lines (+added/-removed) of the
new file against its source.

| source | new path | change | diff |
|---|---|---|---|
| `<OLD>/downstream_task/superresolution/data_v2/pipeline.py` | `bima_ecgsr/data/pipeline.py` | docstring/comment only (3 edits); signal and label code unchanged | +5/-5 |
| `<OLD>/downstream_task/superresolution/data_v2/build_data_v2.py` | `bima_ecgsr/data/build_dataset.py` | paths via config (raw cohort dirs), package import, report titles | +16/-16 |
| `<OLD>/downstream_task/superresolution/data_v2/tests/test_pipeline.py` | `tests/test_pipeline.py` | package import, paths via config, constants checked against vendored file | +18/-17 |
| `LoopedMambaECG/ecg_depth/constants.py` | `bima_ecgsr/data/constants.py` | verbatim copy (sha256 identical) | +0/-0 |
| `/media/gtimpano/Data/UGO/dibima_sr12_v2_data/PIPELINE.md` | `docs/PIPELINE.md` | title + provenance note | +3/-1 |
| `<OLD>/downstream_task/superresolution/train_sr12_v2.py` | `bima_ecgsr/training/train_sr.py` | docstring rewritten, paths via config, build_net uses vendored backbone (+backend arg) | +32/-31 |
| `<OLD>/downstream_task/superresolution/test_sr12_v2.py` | `tests/test_sr.py` | package import, I+II+V1 added (17 -> 20 groups), message wording | +9/-9 |
| `<OLD>/downstream_task/superresolution/models_sr.py` | `bima_ecgsr/models/backbone.py` | vendored subset: SubPixel1D, PositionalEncoding, TemporalEmbedding, SignalEncoder, BidirectionalMamba, DiBiMa_nn -> BiMaSRNet; Lightning wrappers, unused blocks and prints dropped; Mamba backend selectable | +194/-879 |
| `<OLD>/downstream_task/superresolution/utils_sr.py` | `bima_ecgsr/models/backbone.py` | vendored: best_params -> BEST_PARAMS, unmask_channels -> UNMASK_CHANNELS, add_zero_channels | +277/-643 |
| `(mamba_ssm 2.3.1 Mamba / Block / RMSNorm / _init_weights)` | `bima_ecgsr/models/mamba_torch.py, scan_numba.py` | new: pure-PyTorch + numba CPU re-implementation, state-dict compatible |  |
| `<PAPER>/code/sr12_v2/queue_sr12_v2.sh` | `scripts/queue_sr.sh` | repo-relative, env-var paths, BIMA_PYTHON; I+II+V1 jobs added (not launched) | +21/-17 |
| `<PAPER>/code/sr12_v2/queue_sr12_v2_extra.sh` | `scripts/queue_sr_extra.sh` | wrapper: waits for main queue then runs the I+II+V1 jobs of queue_sr.sh (not launched) | +11/-35 |
| `<PAPER>/code/sr12_v2/smoke_sr12_v2.sh` | `scripts/smoke_sr.sh` | repo-relative; SCR must be given; I+II+V1 added | +12/-12 |
| `<OLD>/downstream_task/ecg_classification_v2/classifier.py` | `bima_ecgsr/classification/classifier.py` | docstring line | +1/-1 |
| `<OLD>/downstream_task/ecg_classification_v2/train_classifier.py` | `bima_ecgsr/classification/train_classifier.py` | package imports, paths via config, usage text | +9/-10 |
| `<OLD>/downstream_task/ecg_classification_v2/predict.py` | `bima_ecgsr/classification/predict.py` | relative import, docstring | +7/-7 |
| `<OLD>/downstream_task/ecg_classification_v2/metrics.py` | `bima_ecgsr/classification/metrics.py` | verbatim | +0/-0 |
| `<OLD>/downstream_task/ecg_classification_v2/evaluate.py` | `bima_ecgsr/classification/evaluate.py` | package imports, paths via config, results copy -> results/classification | +11/-10 |
| `<OLD>/downstream_task/ecg_classification_v2/run_all.sh` | `scripts/run_classifier.sh` | rewritten for judge only, repo-relative | +14/-16 |
| `<OLD>/downstream_task/ecg_classification_v2/README.md` | `docs/CLASSIFIER.md` | rewritten (module paths, no history notes) | +28/-25 |
| `<PAPER>/code/sr12_v2/eval_cls_recon.py` | `bima_ecgsr/evaluation/cls_recon.py` | package imports, paths via config | +11/-11 |
| `<PAPER>/code/figures_sr12_v2/prep_v2_stats.py` | `bima_ecgsr/evaluation/sr_stats.py` | paths via config | +7/-4 |
| `<PAPER>/code/figures_sr12/sr12_style.py + its house-style module` | `bima_ecgsr/plotting/style.py` | merged; rcParams/colours/helpers kept; text-rewriting hook removed |  |
| `<PAPER>/code/figures_sr12_v2/graphical_abstract_v2.py` | `figures/graphical_abstract.py` | imports/paths, NAME=Graphical_Abstract | +13/-12 |
| `<PAPER>/code/figures_sr12_v2/figure_sr1_v2.py` | `figures/figure1.py` | imports/paths, NAME=Figure1, param-count helper uses repo module | +21/-18 |
| `<PAPER>/figures/sr12_v2/Figure_SR1_v2.{png,pdf,svg}` | `figures/Figure1.{png,pdf,svg}` | re-rendered from new script (pixel-identical PNG) |  |
| `<PAPER>/figures/sr12_v2/Graphical_Abstract_SR12_v2.{png,pdf,svg}` | `figures/Graphical_Abstract.{png,pdf,svg}` | re-rendered (pixel-identical PNG) |  |
| `<PAPER>/figures/sr12_v2/caption_*.md` | `figures/captions/{Figure1,Graphical_Abstract}.md` | Figure1 caption: source-data file name |  |
| `<PAPER>/results/sr12_v2/sr_II_overall.csv` | `results/sr_II_overall.csv` | byte copy | +0/-0 |
| `<PAPER>/results/sr12_v2/cls/*.csv` | `results/reconstruction_classification/*.csv` | byte copy |  |
| `<OLD>/downstream_task/ecg_classification_v2/results/*.csv` | `results/classification/*.csv` | byte copy |  |
| `<PAPER>/results/<earlier lead-II study>/stats/noise_schedule.csv` | `results/noise_schedule.csv` | byte copy (needed by Figure 1) | +0/-0 |
| `<PAPER>/source_data/sr12_v2/*.csv` | `source_data/{Figure1,Graphical_Abstract}_*.csv` | renamed copies; identical to re-generated |  |
| `<OLD>/hf_space/ (pattern only)` | `space/` | new app/engine/deploy/tools written for this model; weight_file()/private-repo pattern adapted |  |

Vendoring of the model code: only what `train_sr12_v2.py` needs was taken from `models_sr.py` / `utils_sr.py`
(no PyTorch Lightning, UMAP, plotting or dataset utilities). State-dict keys are unchanged (`residual_convs` kept although
unused in `forward`), so the trained `best.pth` files load with `strict=True`. Construction order (and therefore the
initialisation RNG stream) is unchanged.

## 3. Tests (all run from the new repo with the study CUDA conda env; free GPU memory checked first: 16.8 GB)

| suite | result |
|---|---|
| `tests/test_pipeline.py` (data pipeline) | **23/23 OK** |
| `tests/test_sr.py` (reconstruction) | **ALL 20 TEST GROUPS PASSED** (the original 17 + 3 for the added I+II+V1 configuration) |
| `tests/test_classifier.py` (new: model size = 3,364,369 params, z-score / flat-lead guard, scale invariance, AUROC == sklearn, weighted bootstrap == resampled AUROC, record aggregation, strict judge load) | **9/9 OK** |
| `tests/test_mamba_torch.py` (new: associative scan + numba kernel == sequential recursion; state-dict keys == mamba_ssm; trained weights, CPU torch backend vs GPU mamba_ssm, float32, TF32 off) | **3/3 OK**; max abs diff 1.17e-05 (BiMa-SR II), 1.18e-05 (DiBiMa-SR II) |

Log: `docs/verification/test_log.txt`.

## 4. Numerical equivalence with the stored outputs

`tools/verify_equivalence.py` (JSON: `docs/verification/equivalence*.json`). New code, `best.pth` of each run,
mamba_ssm backend on the GPU, bf16 autocast, eval batch 128 and generator seed 0 as in the study test pass (the first
full batch of 128 windows is computed so that the noise draw equals the stored run's; the first 64 windows are
compared; peak GPU memory 1.04-1.07 GiB).

| run : sampler : backend | process | max abs diff (z) | mean abs diff | fraction float16-identical | max abs dPCC |
|---|---|---|---|---|---|
| bima_sr_II:regression:gpu_mamba_ssm | cold process | 0 | 0 | 1 | 0 |
| dibima_sr_II:onestep:gpu_mamba_ssm | cold process | 0.0469 | 0.00131 | 0.436 | 0.00123 |
| dibima_sr_II:ddim6:gpu_mamba_ssm | cold process | 0.0469 | 0.00177 | 0.391 | 0.00158 |
| bima_sr_II:regression:cpu_torch | cold process | 0.0547 | 0.00187 | 0.122 | 0.00643 |
| dibima_sr_II:onestep:cpu_torch | cold process | 0.0548 | 0.00243 | 0.11 | 0.00796 |
| bima_sr_II:regression:gpu_mamba_ssm:warmup | warm-up pass first | 0.0469 | 0.0007 | 0.638 | 0.0017 |
| dibima_sr_II:onestep:gpu_mamba_ssm:warmup | warm-up pass first | 0 | 0 | 1 | 0 |
| dibima_sr_II:ddim6:gpu_mamba_ssm:warmup | warm-up pass first | 0 | 0 | 1 | 0 |

- **BiMa-SR II: bit-identical** to `pred_ptbxl_test.npy` (max abs diff 0.0) in a fresh process.
- **DiBiMa-SR II one-step and DDIM-6: bit-identical** to `pred_ptbxl_test_onestep.npy` / `_ddim6.npy` (0.0) when one
  batch-64 pass runs first (`--warmup`). Without it (or, for BiMa-SR, with it) outputs differ by at most 0.047 z
  (mean 0.0007-0.0018; |dPCC| <= 0.0017). Cause established: the mamba_ssm Triton kernels autotune on first use, so the
  bf16 result depends on the process's call history. The old module (`train_sr12_v2.py`) and the new one give
  identical outputs in the same process (max diff 0.0, cross-check run from a scratch script outside both repos), and
  repeated fresh processes are deterministic (0.0).
- **CPU, pure-PyTorch backend** (float32, first 4 windows; one-step uses the exact stored-run noise): max abs diff 0.055,
  mean 0.0019-0.0024, |dPCC| <= 0.008 vs the stored bf16 GPU outputs. This is bf16-vs-float32 rounding: against the
  float32 GPU path the same backend differs by 1.2e-5 (Section 3).
- **Judge** (`judge12/best.pt`, GPU, bf16), first 64 true PTB-XL test windows vs `pred_true_ptbxl_test.npz`:

| setting | max abs diff window prob. (stored float16) | max abs diff record prob. | complete records |
|---|---|---|---|
| judge12:batch64:cuda | 0.0029 | 0.00119 | 32 |
| judge12:batch256:cuda | 0.000244 | 0 | 32 |

  With the study's prediction batch (256) the record probabilities are identical (0.0); batch 64 changes bf16
  rounding (<= 0.0012).

## 5. Figures re-rendered from the new scripts

`python figures/graphical_abstract.py` and `python figures/figure1.py` (sandbox matplotlib 3.11.2 env, same as the
originals). PNG pixel comparison against the original renders: **Figure 1: 0 differing pixels (4464 x 4251);
Graphical abstract: 0 differing pixels (2881 x 4251)**. SVGs are identical after normalising the creation date and
matplotlib's random clip/hatch ids. All ten source-data CSVs regenerated by the scripts are byte-identical to the
originals. bbox checks: no overlaps. Figure 1 panel b deliberately still shows the five published configurations
(I+II+V1 to be added after training — see open items).

## 6. Banned-term check

The case-insensitive grep with the five patterns of the task specification (earlier project name with and without
leading slash, earlier comparator name, earlier style-module name, and the word stem for data contamination) over the
repo (excluding the sandbox's `.claude/`): **no matches** (re-run after this report was written, see below). Also no
matches for the journal acronym as a word or for the excluded classifier design. Hard-coded locations that remain are documented,
environment-variable-overridable defaults only: `bima_ecgsr/config.py` (`BIMA_DATA_DIR`
`/media/gtimpano/Data/UGO/dibima_sr12_v2_data`, `BIMA_RUNS_DIR` `.../dibima_sr12_v2_runs`, `BIMA_CLS_DIR`
`.../dibima_cls_v2_runs`, `BIMA_RAW_DIR` `/home/gtimpano/Scrivania/UGO/LoopedMambaECG/data`), the same run-dir defaults in
`scripts/queue_sr.sh`, `run_all.sh`, `run_classifier.sh`, and the weights staging default
`/media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging` in `space/deploy.py` / `space/tools/stage_weights.py` /
`smoke_app.py`; plus the default table in README.md. Run-directory names (`dibima_sr_II`, ...) keep the study naming.
No import or `sys.path` entry points outside the repository.

## 7. Hugging Face Space (local only)

`space/`: `app.py` (Gradio 5, "BiMa-ECGSR"), `engine.py`, `README.md` (HF front-matter: sdk gradio 5.50.0, python 3.12,
MIT, PTB-XL CC BY 4.0 attribution), `requirements.txt` (CPU torch, numba, wfdb, diffusers, huggingface_hub),
`deploy.py` (dry-run by default), `tools/stage_weights.py`, `tools/smoke_app.py`, `tools/render_example.py`,
`examples/` (6 unmodified PTB-XL test-fold records + selection rule in `examples.json` + `ATTRIBUTION.md`).

- **CPU Mamba**: the previous Space used ZeroGPU with Triton kernels, so it had no true CPU path. This Space uses a new
  pure-PyTorch Mamba v1 (`bima_ecgsr/models/mamba_torch.py`) whose selective scan runs in a fused numba kernel
  (`scan_numba.py`; chunked associative-scan fallback without numba). It loads the mamba_ssm checkpoints unchanged and
  matches them to 1.2e-5 in float32. `mamba_ssm` is never imported by the app (checked).
- **Weights**: `engine.weight_file()` looks in `$BIMA_WEIGHTS_DIR`, then `space/weights/`, then downloads from the private
  model repo `$BIMA_WEIGHTS_REPO` (default `Ugo96/BiMa-ECGSR-weights`) with the Space secret `HF_TOKEN` into the HF cache.
  Configurations and models are discovered from `MANIFEST.json` at start-up; the dropdown lists only available ones and
  a table lists the others as "weights not yet released" (currently only II is available; I, I+II, I+II+V1, I+II+V2,
  I+II+V1+V4 are disabled).
- **Inputs**: bundled examples or uploads (WFDB `.hea`+`.dat`, uploaded in one go; CSV/NPY 12 x 5000 / 12 x 2500 or
  transposed, mV or µV, optional lead-name header; uploads with only the input leads are accepted). 12-lead records
  with >= 10 s go through `pipeline.process_signal` unchanged (band-pass 0.5-40 Hz on 10 s, two 5-s windows, QC,
  z-score, float16 storage rounding as in the dataset). The app preprocessing reproduces the stored dataset windows
  exactly (max abs diff 0.0 for all 6 examples).
- **Outputs**: 12-lead plot (reconstruction over real, measured input in pink, optional linear FIR dashed), per-lead
  PCC/MAE table (z units), judge probabilities for the 5 superclasses on real / reconstructed / FIR ECGs, the notice
  "Research demonstration — not for clinical use". mV display: measured leads exact (z*std+mean); predicted leads
  z_pred x (median training-set std_lead / s) x s, with s the common input scale of the window (training-set amplitude
  prior, `amplitude_prior.json`).
- **Local test** (`space/tools/smoke_app.py`, sandbox CPU env with torch 2.14.1 CPU + gradio 5.50.0, 2 threads to mimic
  cpu-basic; `app.build()` constructed without launching; standalone bundle from `deploy.build_bundle()` also tested,
  including CSV, NPY (µV), split-folder WFDB and an inputs-only 5-s CSV upload):

| method | runs (6 examples) | preprocessing max diff vs dataset | max abs dPCC per window vs stored | model time (s, 2 windows) | request time (s) |
|---|---|---|---|---|---|
| BiMa-SR | 6 | 0 | 0.00081 | 3.5 | 3.8 |
| DiBiMa-SR (DDIM-6) | 6 | 0 | 0.047 | 19.7 | 20.2 |
| DiBiMa-SR (one-step) | 6 | 0 | 0.018 | 3.3 | 4.2 |

  BiMa-SR reproduces the stored per-window PCC to <= 8e-4. DiBiMa-SR differs more (<= 0.018 one-step, <= 0.047 DDIM-6)
  because the CPU generator draws different noise than the study's CUDA generator; this is sampling variation, not a
  code difference (with the stored noise the CPU result is within 0.008 PCC, Section 4). Render of one example:
  `docs/space_app_render.png` (ecg_id 5784, BiMa-SR from lead II; mean PCC over predicted leads 0.935, FIR 0.784).

## 8. Staged weights

`space/tools/stage_weights.py` run once; output `/media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging` (35 MB,
12 files, `MANIFEST.json` + `MANIFEST.sha256`; verified by `deploy.py` dry-run):

```
10dac7cf91748d7d72eb0e8edbbb1a1971beddc55c81af7cd38349d7413e8142  MANIFEST.json
2dcd75e8ef5dd2214ef921d7307e531c1dff614f97533276a562b6c3e54df7cc  amplitude_prior.json
0db4c528799c7158bc6da79bc9f256126805f3d8fa8c4a0fa919eae38aaabb84  fir/fir_I.npz
c592d4ce2f0ef99bb712497e3741d4e6c7523be2cc0c75439ffb8f254129e1b1  fir/fir_II.npz
993aa19143bb193b3d3225f3b3fdb8246e167fb8042f0722a59a5ccbe234969b  fir/fir_I_II.npz
dd9756cdd9ffe4101f53f7303369760cbcb7011198cb22adaf32fe0ac4bf8087  fir/fir_I_II_V1_V4.npz
82c13a2e7104a3b29e1be6fcae39ea151c1e87d3ee1a3cf49f72414ce182bf64  fir/fir_I_II_V2.npz
68b1274442c3d466c3b63e02db8699fe68839aa2187e90759f166b008d2e54b2  judge/judge12.pt
505cc246a2cce51545b811079405b603f2c03623e151a616d0f482a3c36cf50e  sr/bima_sr_II.config.json
b188ff3274b8eb1cf75911b6557b48ce8db95390f2237a581f2dbb5134c10a88  sr/bima_sr_II.pth
0ad1f70ccadb75b64362a68583e10cfdf235a656d92aa412870fbe3626d42859  sr/dibima_sr_II.config.json
28f1c8337469e55e2b93afc1a942d40cd63355441d9f79dc3f4cc7493ce58d46  sr/dibima_sr_II.pth
```
Staged: `bima_sr_II` (best epoch 9), `dibima_sr_II` (best epoch 10), FIR for II, I, I+II, I+II+V2, I+II+V1+V4, judge12,
amplitude prior for all 6 configurations. Skipped as unfinished: `bima_sr_I` (training at epoch 10/20 at staging time),
`bima_sr_I_II`, `bima_sr_I_II_V2`, `bima_sr_I_II_V1_V4`, `bima_sr_I_II_V1`, `baselines_I_II_V1`.

## 9. Open items

1. After the queue finishes: re-run `space/tools/stage_weights.py` (picks up the new runs), `scripts/run_all.sh`
   (statistics + judge for every configuration), then commit and deploy (`space/deploy.py --yes`).
2. Figure 1 panel b / param counts: add the I+II+V1 row (`CONFIGS` in `figures/figure1.py`) and re-render after
   `bima_sr_I_II_V1` is trained; update the caption.
3. New figures for multi-lead, external validation and classification results are not part of this assembly.
4. The pipeline sha256 quoted in `docs/PIPELINE.md` / `qc_report.json` refers to the original `pipeline.py`; the repo
   copy differs only in its docstring/comments (5 lines). Re-building the dataset would record the new hash.
5. `bima_ecgsr/data/constants.py` is a verbatim copy (keeps the recorded `canonical_constants_sha256`), so it still
   contains one unused tuple (`FAMILIES`).
6. An empty stray directory `figures/{captions}/` was created by a shell brace typo; it is not tracked by git and can be
   removed by the user.
7. Space runtime on cpu-basic is estimated from a 2-thread local run (BiMa-SR about 3-4 s per 10-s record, DDIM-6 about
   20 s); not measured on HF hardware. The Space installs the PyPI CPU torch 2.8 wheel; local tests used conda-forge
   torch 2.14.1 CPU (the PyTorch CPU wheel index was not reachable from the sandbox).
8. `config.py` (bundled into the public Space) contains the local default paths as fallbacks; harmless, but they would
   be visible in the public Space files.

## 10. Deviations

- Added `tools/verify_equivalence.py`, `tests/test_classifier.py`, `tests/test_mamba_torch.py`, `scripts/run_all.sh`
  (no classifier tests existed in the source; `run_all.sh` replaces the classifier's old `run_all.sh`, which became
  `scripts/run_classifier.sh`).
- The SR test suite now has 20 groups (17 original + 3 for I+II+V1).
- The Space's CPU path is a new implementation (pure-PyTorch + numba), not an adaptation of the old Space's fallback,
  because that Space ran Triton kernels on ZeroGPU and had no CPU Mamba path.
- DiBiMa-SR bit-exactness needs a warm-up pass (kernel autotuning); BiMa-SR is bit-exact without it. Both are reported.
- CPU-backend equivalence was checked on 4 windows (not 64) for speed.
- Diffusion Space outputs on CPU use a CPU noise generator (seed 0), so they are not bit-comparable to the stored GPU
  samples.
- `__pycache__` folders created by my own test runs inside the new repo were deleted with `rm -rf` (no other deletions).
