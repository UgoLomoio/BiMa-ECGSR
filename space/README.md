---
title: "BiMa-ECGSR: 12-lead ECG reconstruction from reduced lead sets"
emoji: 🫀
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Reconstruct all 12 ECG leads from 1-4 measured leads
startup_duration_timeout: 1h
suggested_hardware: zero-a10g
tags:
  - ecg
  - electrocardiogram
  - mamba
  - diffusion
  - lead-reconstruction
  - ptb-xl
---

# BiMa-ECGSR: 12-lead ECG reconstruction from reduced lead sets

> **Research demonstration — not for clinical use.** Reconstructed leads are model estimates, not measurements.

Bidirectional-Mamba models reconstruct the full 12-lead ECG (I, II, III, aVR, aVL, aVF, V1-V6) from a reduced set of
measured leads, on 5-s windows at 500 Hz:

- **BiMa-SR**: regression model, one forward pass (about 2.7 M parameters).
- **DiBiMa-SR**: diffusion model (DDPM, T = 20, zero-terminal-SNR cosine schedule, x0-prediction) sampled from pure
  Gaussian noise in one step or with DDIM-6 (lead II input).

Input-lead configurations: II, I, I+II, I+II+V1 (5-electrode bedside telemetry), I+II+V2, I+II+V1+V4. The app
discovers at start-up which trained models are available and only offers those.

## Using the app

Choose a record (a bundled example or an upload) and an input-lead preset: **Smartwatch** (I), **Monitor** (II),
**6-lead device** (I+II), **Telemetry** (I+II+V1), **3-lead set** (I+II+V2), **4-lead set** (I+II+V1+V4). Five tabs:

1. **Reconstruct** — the 12-lead ECG in the clinical 4 x 3 layout plus a 10-s lead II rhythm strip (measured leads
   pink, BiMa-SR orange, real ECG black, optional linear-filter overlay dashed grey); per-lead PCC / MAE when a real
   reference exists; top-3 classifier superclasses; device, Mamba backend and latency; CSV / NPY download (mV).
   DiBiMa-SR (one-step, DDIM-6) is offered for the Monitor (II) preset.
2. **Diagnose & explain** (app only, not part of the paper) — the paper's 12-lead classifier on the real and the
   reconstructed ECG (5 superclasses + 12 panel labels; ground truth of the examples marked); Integrated Gradients
   heatmaps, lead-importance shares, and gradient x input attribution of every reconstructed lead to the measured leads.
3. **Compare input sets** — all six configurations on the same record (chest leads), with this record's chest-lead
   PCC against the paper's PTB-XL test values.
4. **Diffusion uncertainty** — DiBiMa-SR from lead II with N noise draws (mean and min-max band) vs BiMa-SR.
5. **Results & About** — graphical abstract, Tables 1-2 (rendered from the bundled results CSVs), Figure 1, method,
   limitations, licences.

Uploads: WFDB (`.hea` + `.dat`/`.mat`), CSV or NPY (leads x samples or transposed; a header row with lead names is
accepted), either all 12 leads (metrics available) or only the measured leads; CSV / NPY sampling rate is entered
(resampled to 500 Hz) and mV vs µV is detected automatically.

Preprocessing is the study pipeline: first 10 s, zero-phase third-order Butterworth band-pass 0.5-40 Hz, two
non-overlapping 5-s windows, per-window per-lead z-score. The measured input leads are converted back to mV with
their window statistics, mean-centred and divided by one common scale (the RMS of their standard deviations), which
keeps their amplitude ratios. Models predict z-scored leads; for the mV display, predicted leads are scaled by the
median training-set amplitude of each lead relative to the measured inputs.

## Runtime

ZeroGPU: every request runs the networks (reconstruction and the optional classifier) inside one GPU allocation;
preprocessing, the linear FIR baseline, metrics and plotting run on the CPU. On the GPU, Mamba runs with the
`mamba_ssm` CUDA selective-scan kernel when it passes a one-window self-test against a pure-PyTorch implementation
(max |difference| < 1e-3; about 1e-5 in practice); otherwise with the pure-PyTorch implementation on CUDA. Without a
GPU the app runs on the CPU with the pure-PyTorch implementation (numba selective scan). All inference is float32
(no TF32); the device, the backend, the self-test difference and the timings are shown with every result. On ZeroGPU
the GPU is shared, so times vary between requests. Model weights live in a private model repository and are downloaded
at start-up with the Space secret `HF_TOKEN` (`BIMA_WEIGHTS_REPO`, default `Ugo96/BiMa-ECGSR-weights`).

## Data and licences

Bundled examples: eleven records chosen by a fixed rule — within each label group, the record at the 75th percentile
of BiMa-SR per-record PCC for the Telemetry configuration (I+II+V1): PTB-XL v1.0.3 test fold (NORM, MI, STTC, CD, HYP
with a single superclass; CLBBB, CRBBB, AFIB, LVH with a single panel label), one Georgia record and one
Chapman-Shaoxing/Ningbo record (see `examples/examples.json` and `examples/ATTRIBUTION.md`). All three databases are
distributed by PhysioNet under CC BY 4.0: Wagner P, et al. PTB-XL, a large publicly available electrocardiography
dataset. *Scientific Data* 7, 154 (2020); Perez Alday EA, et al. Classification of 12-lead ECGs: the PhysioNet/Computing
in Cardiology Challenge 2020. *Physiol. Meas.* 41, 124003 (2020); Zheng J, et al. A 12-lead electrocardiogram database
for arrhythmia research covering more than 10,000 patients. *Scientific Data* 7, 48 (2020); Goldberger AL, et al.
*Circulation* 101, e215-e220 (2000). The example files are unmodified copies of the original records.

Code: MIT licence (c) Ugo Lomoio.
