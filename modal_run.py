"""Modal wrapper for geometric transportability experiments.

Thin orchestration — all logic lives in src/.

Usage:
    modal run modal_run.py --detach
    modal run modal_run.py --detach --mode multicohort --n-permutations 1000
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
        "networkx", "tqdm", "xgboost", "pyyaml", "requests",
    )
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("data/processed", remote_path="/app/data/processed")
    .add_local_file("config.yaml", remote_path="/app/config.yaml")
)

app = modal.App("geometric-transportability")
vol = modal.Volume.from_name("geometric-transportability-results", create_if_missing=True)

REMOTE_RESULTS = "/results"
TIMEOUT = 86400


@app.function(
    image=image,
    volumes={REMOTE_RESULTS: vol},
    timeout=TIMEOUT,
)
def run_pipeline(mode: str = "two_cohort", backend: str = "raw_features",
                 n_permutations: int = 1000):
    import sys
    import shutil
    import os
    from pathlib import Path
    from datetime import datetime

    os.chdir("/app")
    sys.path.insert(0, "/app")

    print(f"[{datetime.now():%H:%M:%S}] Starting geometric transportability pipeline on Modal")
    print(f"  Mode: {mode}")
    print(f"  Backend: {backend}")
    print(f"  Permutations: {n_permutations}")

    data_dir = Path("/app/data/processed")
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("  ERROR: No processed data found at /app/data/processed/")
        return

    if mode == "multicohort":
        from src.run_multicohort import run
        run(n_perm=n_permutations)
        results_src = Path("/app/results/multicohort")
    else:
        import yaml
        with open("/app/config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["transportability"]["n_permutations"] = n_permutations

        from src.run_all import run
        run(config_path="/app/config.yaml", backend=backend)
        results_src = Path("/app/results")

    results_dst = Path(REMOTE_RESULTS) / f"{mode}_{backend}_{datetime.now():%Y%m%d_%H%M%S}"
    if results_src.exists():
        shutil.copytree(results_src, results_dst)
        print(f"  Results copied to volume: {results_dst}")
        vol.commit()


@app.local_entrypoint()
def main(mode: str = "two_cohort", backend: str = "raw_features",
         n_permutations: int = 1000):
    run_pipeline.remote(mode=mode, backend=backend, n_permutations=n_permutations)
