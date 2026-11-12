"""Deploy the BiMa-ECGSR Space (ZeroGPU) and its private weights repository.

Dry run (default): verifies the staged weights against MANIFEST.sha256, builds the Space upload bundle in a temporary
folder and prints what would be uploaded. Nothing is sent.
  python space/deploy.py [--staging /media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging]
Deploy (explicit):
  HF_TOKEN=<write token> python space/deploy.py --yes [--space-token <read token for the Space secret>]
    1. create (if needed) the PRIVATE model repo Ugo96/BiMa-ECGSR-weights (refuses to upload if it is public) and
       upload the staging folder (weights, MANIFEST.json, MANIFEST.sha256, amplitude prior) plus a model card
    2. create (if needed) the PUBLIC Gradio Space Ugo96/BiMa-ECGSR with --hardware (default zero-a10g = ZeroGPU);
       if that hardware cannot be requested the error is printed and the Space keeps its current hardware
       (the app then runs on the CPU)
    3. set the Space secret HF_TOKEN (read access to the weights repo; default: the deploy token) and the variable
       BIMA_WEIGHTS_REPO
    4. upload the bundle: app.py, engine.py, explain.py, figures.py, README.md, requirements.txt, LICENSE, examples/,
       assets/ (figure images + results CSVs for the Results tab) and the modules of the
       bima_ecgsr package that the app imports (no weights, no data, no tools, no caches)
Tokens are read from the environment ($HF_TOKEN or $HUGGINGFACE_TOKEN) and never printed.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

SPACE = Path(__file__).resolve().parent
REPO = SPACE.parent
SPACE_ID = os.environ.get("BIMA_SPACE_ID", "Ugo96/BiMa-ECGSR")
WEIGHTS_ID = os.environ.get("BIMA_WEIGHTS_REPO", "Ugo96/BiMa-ECGSR-weights")
APP_FILES = ["app.py", "engine.py", "explain.py", "figures.py", "README.md", "requirements.txt"]
PKG_FILES = ["__init__.py", "config.py",
             "data/__init__.py", "data/pipeline.py", "data/constants.py",
             "models/__init__.py", "models/backbone.py", "models/mamba_backend.py", "models/mamba_torch.py",
             "models/scan_numba.py",
             "training/__init__.py", "training/train_sr.py",
             "classification/__init__.py", "classification/classifier.py", "classification/predict.py"]
MODEL_CARD = """---
license: mit
tags:
  - ecg
  - mamba
  - lead-reconstruction
---

# BiMa-ECGSR weights (private)

Checkpoints of the BiMa-ECGSR study (12-lead ECG reconstruction from reduced lead sets with bidirectional Mamba):
BiMa-SR for the input configurations II, I, I+II, I+II+V1, I+II+V2 and I+II+V1+V4, DiBiMa-SR (diffusion) for lead II,
the linear FIR baselines of every configuration, the 12-lead PTB-XL classifier used as the diagnostic judge and the
amplitude prior used for the mV display. All models were trained on PTB-XL v1.0.3 folds 1-8 (CC BY 4.0).

`MANIFEST.json` lists every file with its provenance; `MANIFEST.sha256` holds the checksums
(`sha256sum -c MANIFEST.sha256`). The weights are loaded by the Space
[{space}](https://huggingface.co/spaces/{space}) through its `HF_TOKEN` secret.

Research use only - not for clinical use.
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(dst: Path):
    for f in APP_FILES:
        shutil.copy2(SPACE / f, dst / f)
    shutil.copytree(SPACE / "examples", dst / "examples", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(SPACE / "assets", dst / "assets", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "LICENSE", dst / "LICENSE")
    for f in PKG_FILES:
        (dst / "bima_ecgsr" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "bima_ecgsr" / f, dst / "bima_ecgsr" / f)
    return sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())


def verify_staging(staging: Path):
    lines = (staging / "MANIFEST.sha256").read_text().splitlines()
    bad = [rel for h, rel in (ln.split("  ", 1) for ln in lines) if sha256(staging / rel) != h]
    if bad:
        sys.exit(f"staging checksum mismatch: {bad}")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default=os.environ.get("BIMA_WEIGHTS_STAGING", "/media/gtimpano/Data/UGO/bima_ecgsr_hf_weights_staging"))
    ap.add_argument("--yes", action="store_true", help="actually create the repositories and upload")
    ap.add_argument("--space-token", default=os.environ.get("BIMA_SPACE_READ_TOKEN"),
                    help="token stored as the Space secret HF_TOKEN (read access to the weights repo); default: deploy token")
    ap.add_argument("--hardware", default="zero-a10g", help="Space hardware (default zero-a10g = ZeroGPU; cpu-basic = CPU)")
    ap.add_argument("--skip-weights", action="store_true", help="do not (re-)upload the weights repository")
    ap.add_argument("--message", default="BiMa-ECGSR app (ZeroGPU)")
    a = ap.parse_args()
    staging = Path(a.staging)
    n = verify_staging(staging)
    tmp = Path(tempfile.mkdtemp(prefix="bima_space_bundle_"))
    files = build_bundle(tmp)
    print(f"[weights] {n} files verified in {staging} -> {WEIGHTS_ID} (private model repo)")
    print(f"[space]   {len(files)} files in bundle {tmp} -> {SPACE_ID} (public Gradio Space, {a.hardware})")
    for f in files:
        print("   ", f)
    if not a.yes:
        print("[dry run] nothing uploaded; re-run with --yes to deploy")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN (write access) to deploy")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if not a.skip_weights:
        api.create_repo(WEIGHTS_ID, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(WEIGHTS_ID).private:
            sys.exit(f"{WEIGHTS_ID} exists and is PUBLIC; refusing to upload weights")
        api.upload_file(path_or_fileobj=MODEL_CARD.format(space=SPACE_ID).encode(), path_in_repo="README.md",
                        repo_id=WEIGHTS_ID, repo_type="model", commit_message="Model card")
        c = api.upload_folder(repo_id=WEIGHTS_ID, repo_type="model", folder_path=str(staging),
                              commit_message="BiMa-ECGSR weights (BiMa-SR x6, DiBiMa-SR II, FIR x6, judge12, amplitude prior)")
        print(f"[weights] uploaded: {getattr(c, 'commit_url', c)}")
    try:
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", private=False, exist_ok=True,
                        space_hardware=a.hardware)
    except Exception as exc:
        print(f"[space] create with hardware {a.hardware} failed: {type(exc).__name__}: {exc}")
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    try:
        rt = api.get_space_runtime(SPACE_ID)
        if a.hardware not in (rt.hardware, rt.requested_hardware):
            api.request_space_hardware(SPACE_ID, a.hardware)
            print(f"[space] requested hardware {a.hardware}")
    except Exception as exc:
        print(f"[space] hardware request {a.hardware} failed: {type(exc).__name__}: {exc}")
    api.add_space_secret(SPACE_ID, "HF_TOKEN", a.space_token or token)
    api.add_space_variable(SPACE_ID, "BIMA_WEIGHTS_REPO", WEIGHTS_ID)
    c = api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(tmp), commit_message=a.message)
    print(f"[space] uploaded: {getattr(c, 'commit_url', c)}")
    rt = api.get_space_runtime(SPACE_ID)
    print(f"[done] https://huggingface.co/spaces/{SPACE_ID} stage={rt.stage} hardware={rt.hardware} "
          f"requested={rt.requested_hardware}")


if __name__ == "__main__":
    main()
