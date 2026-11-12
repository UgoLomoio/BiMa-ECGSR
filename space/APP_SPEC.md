# BiMa-ECGSR Space — application specification (approved 2026-11-12)

Public Gradio Space `Ugo96/BiMa-ECGSR`, ZeroGPU hardware, CPU fallback. Weights are downloaded at start-up from the private
model repo `Ugo96/BiMa-ECGSR-weights` with the `HF_TOKEN` Space secret.

## Global
- Persistent banner on every tab: **"Research demonstration — not for clinical use."**
- Names: BiMa-SR (regression), DiBiMa-SR (diffusion), "linear filter", "Einthoven + linear filter". No other project names.
- Colours (paper style): BiMa-SR `#E69F00`, DiBiMa-SR one-step `#0072B2`, DDIM-6 `#56B4E9`, linear filter `#999999` (dashed),
  measured input `#CC79A7`, real ECG black.
- Plotly traces are passed as plain Python lists (no typed arrays).
- All GPU work runs inside `@spaces.GPU` functions. Backend on GPU: `mamba_ssm` if its start-up self-test passes
  (|Δ| < 1e-3 against the PyTorch backend on one window), else the pure-PyTorch Mamba on CUDA; without ZeroGPU the CPU path.
  The device and backend actually used, and the latency, are shown after every run.

## Input-lead presets (all tabs)
| Button | Measured leads | Use case |
|---|---|---|
| Smartwatch | I | wrist single-lead devices |
| Monitor | II | bedside / Holter rhythm strip |
| 6-lead device | I+II | limb-lead devices (6 limb leads) |
| Telemetry | I+II+V1 | 5-electrode bedside telemetry |
| 3-lead set | I+II+V2 | reduced-lead challenge set |
| 4-lead set | I+II+V1+V4 | limb + two chest leads |
Only configurations whose weights are present are enabled (runtime discovery).

## Tab 1 — Reconstruct
Inputs
- Example picker: ~12 bundled records, each labelled with cohort, record id and diagnosis:
  PTB-XL test fold — NORM, MI, STTC, CD, HYP, CLBBB, CRBBB, AFIB (or the closest available), LVH; Georgia × 1; China × 1.
  **Selection rule** (shown in the app): within each label group, the record at the 75th percentile of BiMa-SR per-record PCC
  for the Telemetry configuration (I+II+V1). Only records whose label is the single positive label of its group where possible.
  Licences: PTB-XL, Georgia (PhysioNet/CinC 2020) and Chapman-Shaoxing/Ningbo are CC BY 4.0 — attribution shown under the picker.
- Upload: WFDB (.hea + .dat), CSV or NPY; 500 Hz (resampled otherwise, with a note); either 12 leads (real reference available,
  metrics shown) or only the measured leads (no metrics). Units mV (µV auto-detected). Same preprocessing as the paper:
  0.5–40 Hz zero-phase band-pass, first 10 s → two 5-s windows, per-lead z-score with stored statistics.
- Preset buttons (above). Model: BiMa-SR (default); DiBiMa-SR one-step and DDIM-6 enabled only for Monitor (II).
- Toggles: show linear-filter overlay (default off); run classifier (default on).
Outputs
- 12-lead plot, clinical 4 × 3 layout (I, II, III | aVR, aVL, aVF | V1–V3 | V4–V6) plus a 10-s lead II rhythm strip, 10 s shown
  (two windows). Measured leads pink, reconstruction orange, real black, linear filter dashed grey. Hover shows values.
- Metrics table (when a real reference exists): per predicted lead PCC and MAE (z units) for the model and the linear filter;
  summary rows "all predicted leads" and "predicted chest leads", labelled as in the paper (pooled median over windows × leads).
  Note: limb leads are exact by Einthoven when I and II are measured.
- Compact classifier line: top-3 superclass probabilities on real vs reconstruction, with a "see Diagnose & explain" link
  (the full classifier view is Tab 2; the reconstruction is shared with Tab 2 through gr.State, no second GPU call).
- Run info: device (GPU name or CPU), Mamba backend, model latency (ms per 10-s record), total request time.
- Downloads: reconstructed 12-lead ECG as CSV and NPY (mV; predicted leads rescaled with the training-set amplitude prior, measured
  leads unchanged; the scaling is stated).

## Tab 2 — Diagnose & explain (app only; not part of the paper)
Uses the record and reconstruction of Tab 1 (or runs Tab 1's pipeline if none yet). The classifier is the paper's 12-lead
1D-ResNet (3.36 M parameters) trained only on real PTB-XL ECGs, run unchanged.
Classification
- Side-by-side horizontal bars: NORM, MI, STTC, CD, HYP and the 12 panel labels (AFIB, AFLT, 1AVB, PAC, PVC, SBRAD, STACH, WPW,
  LVH, RVH, CRBBB, CLBBB) on real ECG (if available), BiMa-SR / DiBiMa-SR reconstruction, and the linear filter (toggle).
  Record-level probability = mean of the two 5-s windows, as in the paper. Rows where |Δ| > 0.10 are highlighted.
- Ground-truth labels of bundled examples shown next to the bars.
- Caveat text: "A reconstruction cannot add information absent from the measured leads; agreement with the real ECG means the
  reconstruction is readable by a 12-lead tool, not that it is diagnostic."
Explainability
1. Classifier attribution: Integrated Gradients (32 steps, zero baseline = per-lead mean in z units, both windows) for a class
   selected in a dropdown (default: the top class on the real ECG). Displayed as a time-resolved heatmap under each lead of the
   12-lead plot, for real and reconstructed ECG side by side, with a shared colour scale (|attribution|, normalised per record).
2. Lead importance: bar chart of the share of |attribution| per lead (real vs reconstructed), with measured leads marked pink, and
   the total share falling on measured vs reconstructed leads.
3. Reconstruction attribution (what the reconstruction relies on): for a predicted lead chosen in a dropdown, gradient × input of
   that lead's output (sum over a selectable time window, default the whole 5-s window) with respect to the measured inputs:
   (a) time profile over each measured lead (shows which beats/segments drive it); (b) a lead-to-lead matrix (rows = predicted
   leads, columns = measured leads) of the share of |grad × input| — informative for multi-lead configurations.
- All attributions are computed in one @spaces.GPU call (fp32, the PyTorch Mamba backend if the fused kernel has no backward pass).
- Caveat text: "Saliency maps are heuristic; they show sensitivity of the model output, not a clinical mechanism."

## Tab 3 — Compare input sets
- Same record source (example or upload). One GPU call runs all enabled configurations (BiMa-SR).
- Small multiples: rows = configurations, columns = V1–V6 (measured chest leads drawn pink and marked "measured"), real ECG black.
- Chart: chest-lead PCC (this record) vs configuration for BiMa-SR and the linear baseline (Einthoven + linear filter when I and II
  are measured), with the paper's cohort-level PTB-XL values as faint reference markers (read from results/final/table1_reconstruction.csv).

## Tab 4 — Diffusion uncertainty (lead II only)
- DiBiMa-SR with N noise draws (slider 4–16, default 8; one-step sampler; fixed seeds 0..N-1 shown).
- Plot: per lead, mean of draws with min–max band (blue), BiMa-SR (orange), real (black).
- Table: per-lead mean PCC of the draw mean, spread (mean over time of the max–min range), BiMa-SR PCC.
- Note: in the paper, BiMa-SR is more accurate than DiBiMa-SR; the band shows sampling variability, not calibrated uncertainty.

## Tab 5 — Results & About
- Graphical abstract image; Table 1 (reconstruction, PTB-XL/Georgia/China) and Table 2 (classifier) rendered from the results CSVs
  bundled with the Space (no hand-typed numbers); Figure 1 (method) image.
- Method in five sentences; limitations: z-scored per-lead evaluation (absolute amplitudes not assessed), standard electrode positions
  only (modified Holter / EASI / Frank leads not simulated), external cohorts differ in devices and labels, no prospective or
  clinical validation.
- Dataset licences and attribution; citation (preprint, to be filled in); links to the paper and code (placeholders until public).

## Acceptance checks (local QA)
1. Every tab (incl. the attribution calls) runs on the GPU path (mamba_ssm and PyTorch-CUDA backends) and on the CPU path.
2. For bundled PTB-XL test examples, the reconstruction equals the stored test prediction of that window to |Δ| < 1e-2 (bf16) and
   the app's per-lead PCC matches metrics_ptbxl_test.npz to 3 decimals (BiMa-SR).
3. Upload paths: WFDB, CSV (12-lead and measured-only), NPY (µV), wrong shape / wrong rate → clear error message.
4. Integrated Gradients completeness: sum of attributions ≈ f(x) − f(baseline) within 5% (logit of the selected class);
   reconstruction gradients finite and identical on GPU backends to 1e-3 relative.
5. Screenshots of each tab saved for review.
