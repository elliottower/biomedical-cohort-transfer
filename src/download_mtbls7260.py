"""Download MTBLS7260 (pancreatic cancer, 10 PLCO sites, 1036 serum samples).

3 LC-MS assays (HILIC + reverse-phase). 10 U.S. PLCO screening centers
provide natural site-level batch structure for transportability testing.

Usage:
    python src/download_mtbls7260.py
"""
import json
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


STUDY_ID = "MTBLS7260"
DATA_DIR = Path("data/raw") / STUDY_ID
FTP_BASE = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{STUDY_ID}"
API_BASE = f"https://www.ebi.ac.uk/metabolights/ws/studies/{STUDY_ID}"


def download_file(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        return None
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {dest.name}") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def get_file_list():
    """Get file listing from MetaboLights API."""
    print(f"[{datetime.now():%H:%M:%S}] Fetching file list for {STUDY_ID}...")

    resp = requests.get(f"{API_BASE}/files/tree", timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "study" in data:
            return data["study"]
        return data
    return None


def download_isa_files():
    """Download ISA-Tab metadata files (sample table, assay table, etc.)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    filenames = [
        "i_Investigation.txt",
        f"s_{STUDY_ID}.txt",
        f"a_{STUDY_ID}_metabolite_profiling_mass_spectrometry.txt",
        f"a_{STUDY_ID}_metabolite_profiling_mass_spectrometry_v2_maf.tsv",
        f"m_{STUDY_ID}_metabolite_profiling_mass_spectrometry_v2_maf.tsv",
    ]

    for fn in filenames:
        url = f"{FTP_BASE}/{fn}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 50:
            (DATA_DIR / fn).write_bytes(resp.content)
            downloaded.append(fn)
            print(f"  Downloaded {fn} ({len(resp.content):,} bytes)")

    files = get_file_list()
    if files:
        for entry in files:
            if isinstance(entry, dict):
                fn = entry.get("file", entry.get("name", ""))
            else:
                fn = str(entry)
            if fn and any(fn.endswith(ext) for ext in [".txt", ".tsv", ".csv"]):
                if fn.split("/")[-1] not in downloaded:
                    url = f"{FTP_BASE}/{fn}"
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 50:
                        outpath = DATA_DIR / fn.split("/")[-1]
                        outpath.write_bytes(resp.content)
                        downloaded.append(fn)
                        print(f"  Downloaded {fn} ({len(resp.content):,} bytes)")

    return downloaded


def inspect_downloaded():
    """Inspect what we got."""
    print(f"\n[{datetime.now():%H:%M:%S}] Inspecting downloaded files...")
    for f in sorted(DATA_DIR.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name}: {size:,} bytes")
        if f.suffix in (".txt", ".tsv", ".csv") and size < 500_000:
            with open(f) as fh:
                lines = fh.readlines()
            print(f"    {len(lines)} lines")
            if lines:
                header = lines[0].strip().split("\t")
                print(f"    Columns ({len(header)}): {header[:10]}...")
                if len(lines) > 1:
                    print(f"    First data row: {lines[1].strip()[:200]}...")


def log_download():
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
    download_isa_files()
    inspect_downloaded()
    log_download()
    print(f"\n[{datetime.now():%H:%M:%S}] Done.")
