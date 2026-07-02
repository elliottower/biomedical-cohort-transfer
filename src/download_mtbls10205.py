"""Download and inspect MTBLS10205 (breast cancer, 4 hospitals, 1947 serum samples).

MetaboLights study: non-targeted + targeted LC-MS/MS serum metabolomics for
breast cancer early detection. 4 cohorts across 4 hospitals in Zhejiang Province.

Usage:
    python src/download_mtbls10205.py
"""
import json
import requests
from pathlib import Path
from datetime import datetime


STUDY_ID = "MTBLS10205"
DATA_DIR = Path("data/raw") / STUDY_ID
API_BASE = f"https://www.ebi.ac.uk/metabolights/ws/studies/{STUDY_ID}"
FTP_BASE = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{STUDY_ID}"


def get_study_info():
    """Fetch study metadata from MetaboLights API."""
    print(f"[{datetime.now():%H:%M:%S}] Fetching study info for {STUDY_ID}...")

    try:
        resp = requests.get(f"{API_BASE}", timeout=30)
        if resp.status_code == 200:
            info = resp.json()
            print(f"  Title: {info.get('title', 'N/A')}")
            print(f"  Description: {info.get('description', 'N/A')[:200]}...")
            return info
    except Exception as e:
        print(f"  API query failed: {e}")

    return None


def list_files():
    """List available files in the study."""
    print(f"\n[{datetime.now():%H:%M:%S}] Listing files...")

    try:
        resp = requests.get(f"{API_BASE}/files/tree", timeout=30)
        if resp.status_code == 200:
            files = resp.json()
            print(f"  Found {len(files)} file entries")
            return files
    except Exception as e:
        print(f"  File listing failed: {e}")

    try:
        resp = requests.get(f"{API_BASE}/files", timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    return None


def download_sample_table():
    """Download the sample metadata table (s_*.txt or a_*.txt)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for prefix in ["s_", "a_", "m_", "i_"]:
        for ext in [".txt", ".tsv"]:
            filename = f"{prefix}{STUDY_ID}{ext}"
            url = f"{FTP_BASE}/{filename}"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 100:
                    outpath = DATA_DIR / filename
                    outpath.write_bytes(resp.content)
                    print(f"  Downloaded {filename} ({len(resp.content)} bytes)")
            except Exception:
                pass

    for filename in ["s_MTBLS10205.txt", "a_MTBLS10205.txt",
                     "s_study.txt", "a_study.txt",
                     "m_MTBLS10205.tsv", "i_Investigation.txt"]:
        url = f"{FTP_BASE}/{filename}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
                outpath = DATA_DIR / filename
                outpath.write_bytes(resp.content)
                print(f"  Downloaded {filename} ({len(resp.content)} bytes)")
        except Exception:
            pass


def download_metabolite_tables():
    """Try to download processed metabolite/feature tables."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        "m_MTBLS10205_metabolite_profiling_mass_spectrometry_v2_maf.tsv",
        "m_MTBLS10205_metabolite_profiling_mass_spectrometry_maf.tsv",
    ]
    for filename in candidates:
        url = f"{FTP_BASE}/{filename}"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 100:
                outpath = DATA_DIR / filename
                outpath.write_bytes(resp.content)
                print(f"  Downloaded {filename} ({len(resp.content)} bytes)")
        except Exception:
            pass


def log_download():
    """Log the download."""
    log_path = Path("data/raw/download_log.json")
    log = []
    if log_path.exists():
        log = json.loads(log_path.read_text())

    log.append({
        "accession": STUDY_ID,
        "source": "MetaboLights",
        "url": FTP_BASE,
        "path": str(DATA_DIR),
        "timestamp": datetime.now().isoformat(),
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    info = get_study_info()
    files = list_files()
    download_sample_table()
    download_metabolite_tables()
    log_download()

    print(f"\n[{datetime.now():%H:%M:%S}] Done. Check {DATA_DIR}/ for downloaded files.")
    print("  If feature matrices are not in the standard MetaboLights format,")
    print("  check supplementary materials of the associated publication.")
