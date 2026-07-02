#!/bin/bash
# Download CRC microbiome meta-analysis data from zellerlab/crc_meta
# Wirbel et al. 2019 (Nature Medicine) + Thomas et al. 2019 (Nature Medicine)
#
# Usage: cd microbiome-crc && bash download.sh

set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Cloning zellerlab/crc_meta (sparse checkout)..."
if [ ! -d "$DATA_DIR/crc_meta" ]; then
    git clone --depth 1 https://github.com/zellerlab/crc_meta.git "$DATA_DIR/crc_meta"
else
    echo "  Already cloned, skipping."
fi

echo ""
echo "Available data files:"
find "$DATA_DIR/crc_meta" -name "*.tsv" -o -name "*.csv" -o -name "*.RData" -o -name "*.rds" | head -30

echo ""
echo "Done. Inspect the files above to find abundance tables and metadata."
echo "Then update run_crc.py's load_crc_cohorts() to match the actual format."
