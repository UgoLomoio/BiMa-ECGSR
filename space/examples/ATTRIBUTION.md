# Bundled example records

Unmodified copies of eleven public ECG records, all licensed under Creative Commons Attribution 4.0 International
(CC BY 4.0), redistributed with attribution:

- PTB-XL v1.0.3, test fold 10 (nine records, `*_hr.hea/.dat`): Wagner P, Strodthoff N, Bousseljot R-D, Samek W,
  Schaeffter T. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154 (2020).
  https://physionet.org/content/ptb-xl/1.0.3/
- Georgia 12-lead ECG Challenge database (one record, `E*.hea/.mat`), PhysioNet/Computing in Cardiology Challenge
  2020/2021: Perez Alday EA, et al. Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge
  2020. *Physiological Measurement* 41, 124003 (2020).
- Chapman-Shaoxing/Ningbo large-scale 12-lead ECG database, ecg-arrhythmia v1.0.0 (one record, `JS*.hea/.mat`):
  Zheng J, et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients.
  *Scientific Data* 7, 48 (2020); Zheng J, et al. *Scientific Data* 9, 1 (2022). https://physionet.org/content/ecg-arrhythmia/1.0.0/
- PhysioNet: Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101, e215-e220 (2000).

Selection rule (fixed in advance): within each label group, the record whose BiMa-SR per-record PCC for the
Telemetry configuration (I+II+V1; mean over the two 5-s windows of the per-window mean over the nine predicted
leads) is closest to the group's 75th percentile; groups = PTB-XL test records with a single diagnostic superclass
(NORM, MI, STTC, CD, HYP) or a single panel label (CLBBB, CRBBB, AFIB, LVH), and all records of Georgia and of China.
Only records with two valid 5-s windows were eligible. See `examples.json`.
