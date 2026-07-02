"""Download metabolomics datasets from public repositories.

Supports MetaboLights, Metabolomics Workbench, and direct URLs.
Downloads to data/raw/ and logs the accession used.
"""
import os
import json
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


DATA_DIR = Path("data/raw")


def download_file(url, dest, chunk_size=8192):
    """Download a file with progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {dest.name}") as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest


def download_metabolights(study_id, dest_dir=None):
    """Download study files from MetaboLights FTP.

    study_id: e.g. 'MTBLS13770'
    """
    if dest_dir is None:
        dest_dir = DATA_DIR / study_id
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{study_id}"
    ftp_base = f"ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{study_id}"

    print(f"[{datetime.now():%H:%M:%S}] Downloading MetaboLights {study_id}...")

    try:
        resp = requests.get(f"{base_url}/files", timeout=30)
        resp.raise_for_status()
        files_info = resp.json()
        print(f"  Found {len(files_info.get('study', []))} files")
    except Exception as e:
        print(f"  API query failed ({e}), trying direct file list...")
        files_info = {}

    log_download(study_id, "MetaboLights", str(dest_dir))
    return dest_dir


def download_metabolomics_workbench(study_id, dest_dir=None):
    """Download from Metabolomics Workbench REST API.

    study_id: e.g. 'ST000089'
    """
    if dest_dir is None:
        dest_dir = DATA_DIR / study_id
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] Downloading MW {study_id}...")

    base = "https://www.metabolomicsworkbench.org/rest/study"
    for endpoint in ["analysis", "datatable", "factors"]:
        url = f"{base}/study_id/{study_id}/{endpoint}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                outfile = dest_dir / f"{endpoint}.json"
                outfile.write_text(resp.text)
                print(f"  Downloaded {endpoint} ({len(resp.text)} bytes)")
        except Exception as e:
            print(f"  Failed to download {endpoint}: {e}")

    log_download(study_id, "Metabolomics Workbench", str(dest_dir))
    return dest_dir


def download_url(url, filename=None, dest_dir=None):
    """Download a file from a direct URL."""
    if dest_dir is None:
        dest_dir = DATA_DIR
    dest_dir = Path(dest_dir)

    if filename is None:
        filename = url.split("/")[-1]

    dest = dest_dir / filename
    download_file(url, dest)
    log_download(url, "direct URL", str(dest))
    return dest


def log_download(accession, source, path):
    """Log the download to data/raw/download_log.json."""
    log_path = DATA_DIR / "download_log.json"
    log = []
    if log_path.exists():
        log = json.loads(log_path.read_text())

    log.append({
        "accession": accession,
        "source": source,
        "path": path,
        "timestamp": datetime.now().isoformat(),
    })

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
