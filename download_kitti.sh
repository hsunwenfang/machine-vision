#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download KITTI Raw data for the stereo SLAM / SfM pipeline.
#
# Usage:
#   ./download_kitti.sh                         # default: 2011_09_26 drive 0001
#   ./download_kitti.sh 2011_09_26 0001         # explicit date + drive
#   ./download_kitti.sh 2011_09_26 0001 0005    # multiple drives
#
# Downloads into data/kitti/ with the layout expected by KittiLoader:
#   data/kitti/{date}_calib/          ← calibration files
#   data/kitti/{date}_data/
#     {date}_drive_{drive}_extract/   ← images, OXTS, velodyne
#
# Requires: wget (or curl), unzip
# Source: https://www.cvlibs.net/datasets/kitti/raw_data.php
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Configuration --------------------------------------------------------
KITTI_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)/data/kitti"

DATE="${1:-2011_09_26}"
shift 2>/dev/null || true
DRIVES=("${@:-0001}")

# Fallback: if no drives given, default to 0001
if [ ${#DRIVES[@]} -eq 0 ]; then
    DRIVES=("0001")
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# ---- Helper: download + extract ------------------------------------------
fetch() {
    local url="$1"
    local fname
    fname="$(basename "$url")"

    if [ -f "$fname" ]; then
        echo "  [skip] $fname already exists"
    else
        echo "  [download] $fname"
        if command -v wget &>/dev/null; then
            wget -q --show-progress "$url" -O "$fname"
        elif command -v curl &>/dev/null; then
            curl -L --progress-bar "$url" -o "$fname"
        else
            echo "ERROR: neither wget nor curl found" >&2
            exit 1
        fi
    fi
}

extract() {
    local zip="$1"
    local target_dir="$2"

    if [ -d "$target_dir" ]; then
        echo "  [skip] $target_dir already extracted"
    else
        echo "  [extract] $zip → $target_dir"
        unzip -q -o "$zip"
    fi
}

# ---- 1. Calibration -------------------------------------------------------
CALIB_ZIP="${DATE}_calib.zip"
CALIB_DIR="${DATE}_calib"
CALIB_URL="${KITTI_URL}/${CALIB_ZIP}"

echo "=== Calibration ($DATE) ==="
fetch "$CALIB_URL"
extract "$CALIB_ZIP" "$CALIB_DIR"

# The zip extracts to {date}/ — restructure if needed
if [ -d "$DATE" ] && [ ! -d "$CALIB_DIR" ]; then
    mv "$DATE" "$CALIB_DIR"
fi

# ---- 2. Drive data (unsynced / extract) -----------------------------------
for DRIVE in "${DRIVES[@]}"; do
    DRIVE_NAME="${DATE}_drive_${DRIVE}"
    ZIP_NAME="${DRIVE_NAME}/${DRIVE_NAME}_extract.zip"
    LOCAL_ZIP="${DRIVE_NAME}_extract.zip"
    EXTRACT_DIR="${DATE}_data/${DRIVE_NAME}_extract"
    DRIVE_URL="${KITTI_URL}/${ZIP_NAME}"

    echo ""
    echo "=== Drive ${DRIVE} ==="

    fetch "$DRIVE_URL"

    if [ -d "$EXTRACT_DIR" ]; then
        echo "  [skip] $EXTRACT_DIR already extracted"
    else
        echo "  [extract] $LOCAL_ZIP"
        # KITTI zips extract to {date}/{date}_drive_{drive}_extract/
        unzip -q -o "$LOCAL_ZIP"
        # Move to expected layout: {date}_data/{date}_drive_{drive}_extract/
        mkdir -p "${DATE}_data"
        if [ -d "${DATE}/${DRIVE_NAME}_extract" ]; then
            mv "${DATE}/${DRIVE_NAME}_extract" "${DATE}_data/"
        fi
    fi

    # ---- 3. Tracklets (optional, for object annotations) ------------------
    TRACKLET_ZIP="${DRIVE_NAME}/${DRIVE_NAME}_tracklet_labels.zip"
    TRACKLET_LOCAL="${DRIVE_NAME}_tracklet_labels.zip"
    TRACKLET_URL="${KITTI_URL}/${TRACKLET_ZIP}"

    echo "  [tracklets] attempting (optional)..."
    if fetch "$TRACKLET_URL" 2>/dev/null; then
        if [ ! -d "${DATE}_tracklet/${DRIVE_NAME}_sync" ]; then
            mkdir -p "${DATE}_tracklet"
            unzip -q -o "$TRACKLET_LOCAL" -d "${DATE}_tracklet" 2>/dev/null || true
        fi
    fi
done

# ---- Cleanup temp dirs ----------------------------------------------------
rmdir "$DATE" 2>/dev/null || true

# ---- Summary --------------------------------------------------------------
echo ""
echo "=== Done ==="
echo "Data directory: $OUT_DIR"
echo ""
echo "Layout:"
find "$OUT_DIR" -maxdepth 3 -type d | sort | head -30
echo ""
echo "Usage in Python:"
echo "  loader = KittiLoader('$OUT_DIR', date='$DATE', drive='${DRIVES[0]}')"
