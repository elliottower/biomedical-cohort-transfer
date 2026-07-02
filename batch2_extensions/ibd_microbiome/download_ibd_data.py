"""Download IBD metagenomics data from public sources.

Tries multiple sources in order:
1. curatedMetagenomicData via R export script
2. Direct download from Zenodo/GitHub if available

Usage: PYTHONPATH=. uv run python batch2_extensions/ibd_microbiome/download_ibd_data.py
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("batch2_extensions/ibd_microbiome/data")


def try_r_export():
    """Try exporting via curatedMetagenomicData R package."""
    print(f"[{datetime.now():%H:%M:%S}] Attempting R export...")
    result = subprocess.run(
        ["Rscript", "batch2_extensions/ibd_microbiome/export_cmd_data.R"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"  R export failed: {result.stderr[:500]}")
        return False


def try_direct_download():
    """Download pre-processed IBD data from known public sources."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # curatedMetagenomicData has pre-exported files on ExperimentHub
    # But those require R. Let's check for any direct TSV sources.

    # Option: Use the HMP2/IBDMDB data
    # The iHMP project has processed species profiles at:
    # https://ibdmdb.org/tunnel/products/HMP2/WGS/1818/taxonomic_profiles.tsv.gz
    urls_to_try = [
        # HMP2 IBDMDB taxonomic profiles
        ("https://ibdmdb.org/tunnel/products/HMP2/WGS/1818/taxonomic_profiles.tsv.gz",
         "hmp2_taxonomic_profiles.tsv.gz"),
        # HMP2 metadata
        ("https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv",
         "hmp2_metadata.csv"),
    ]

    success = False
    for url, filename in urls_to_try:
        outpath = DATA_DIR / filename
        if outpath.exists():
            print(f"  Already have {filename}")
            success = True
            continue
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, outpath)
            print(f"  Saved {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")
            success = True
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

    return success


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    species_file = DATA_DIR / "species_profiles.tsv"
    meta_file = DATA_DIR / "metadata.tsv"

    if species_file.exists() and meta_file.exists():
        print("Data already exists. Skipping download.")
        sys.exit(0)

    if try_r_export():
        if species_file.exists():
            print("R export successful.")
            sys.exit(0)

    print("\nR export failed or incomplete. Trying direct download...")
    if try_direct_download():
        print("\nDirect download complete. May need post-processing.")
        print("Check batch2_extensions/ibd_microbiome/data/ for downloaded files.")
    else:
        print("\nAll download attempts failed.")
        print("To get IBD data manually:")
        print("  1. Install curatedMetagenomicData in R:")
        print("     BiocManager::install('curatedMetagenomicData')")
        print("  2. Run: Rscript batch2_extensions/ibd_microbiome/export_cmd_data.R")
        print("  Or:")
        print("  3. Download from https://ibdmdb.org/ (iHMP/IBDMDB project)")
        sys.exit(1)
