#!/bin/bash

# --- Configuration ---
BASE_REMOTE_URL="https://cdaweb.gsfc.nasa.gov/pub/data/rbsp"
LOCAL_BASE_DIR="/home/will/REPT_data/MagData"

SATELLITES=("rbspa" "rbspb")
START_YEAR=2012
END_YEAR=2019

# --- Main Script ---

echo "Starting data download from CDAWeb..."
echo "Local base directory: $LOCAL_BASE_DIR"

# Loop through each satellite
for SAT in "${SATELLITES[@]}"; do
    echo -e "\n--- Processing Satellite: $SAT ---"
    
    # Loop through each year
    for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
        echo "  Processing Year: $YEAR"

        # Construct the remote URL for the year's directory
        # Example: https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/emfisis/magnetometer/4sec/gsm/2012/
        REMOTE_DIR_URL="${BASE_REMOTE_URL}/${SAT}/l3/emfisis/magnetometer/4sec/gsm/${YEAR}/"
        
        # Construct the local directory path
        # Example: /home/will/REPT_data/MagData/rbspa/2012/
        LOCAL_TARGET_DIR="${LOCAL_BASE_DIR}/${SAT}/${YEAR}/"

        echo "    Remote URL: $REMOTE_DIR_URL"
        echo "    Local Dir:  $LOCAL_TARGET_DIR"

        # Create the local directory if it doesn't exist
        mkdir -p "$LOCAL_TARGET_DIR"

        # Use wget to download files recursively
        # -r: recursive download (follows links, but we're giving a directory)
        # -np: no-parent (don't ascend to the parent directory)
        # -nd: no-directories (don't create a hierarchy of directories like 'pub/data/rbsp/...')
        # -A "*.cdf": accept only .cdf files
        # -P "$LOCAL_TARGET_DIR": set the prefix for local files (download into this directory)
        # --show-progress: show progress bar (wget 1.16+ typically)
        # -q --show-progress: quiet output except progress
        
        # Check if the directory already contains .cdf files. If so, skip to avoid re-downloading everything.
        if ls "${LOCAL_TARGET_DIR}"/*.cdf 1> /dev/null 2>&1; then
            echo "    Directory already contains .cdf files. Skipping download for this year."
        else
            echo "    Downloading .cdf files..."
            wget -r -np -nd -A "*.cdf" -P "$LOCAL_TARGET_DIR" "$REMOTE_DIR_URL" --show-progress
            
            # Check wget's exit status
            if [ $? -eq 0 ]; then
                echo "    Download successful for $SAT/$YEAR."
            else
                echo "    WARNING: Download failed for $SAT/$YEAR. Check URL or network connection."
            fi
        fi
    done
done

echo -e "\nAll downloads attempted. Script finished."