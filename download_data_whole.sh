#!/bin/bash

# --- Configuration ---
# Define the new base URL for the REPT data
BASE_REMOTE_URL="https://cdaweb.gsfc.nasa.gov/pub/data/rbsp"
LOCAL_BASE_DIR="/home/will/REPT_data"

# Satellite and year ranges
SATELLITES=("rbspa" "rbspb")
START_YEAR=2012
END_YEAR=2019

# --- Main Script ---

echo "Starting download process for RBSP REPT satellite data."

# Loop through each satellite
for SAT in "${SATELLITES[@]}"; do
    echo -e "\n--- Processing Satellite: $SAT ---"
    
    # Loop through each year
    for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
        echo "  Processing Year: $YEAR"

        # Construct the remote URL for the year's directory
        # Example: https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/rept/sectors/rel03/2012/
        REMOTE_DIR_URL="${BASE_REMOTE_URL}/${SAT}/l3/ect/rept/sectors/rel03/${YEAR}/"
        
        # Construct the local directory path
        # Example: /home/will/REPT_data/2012/
        LOCAL_TARGET_DIR="${LOCAL_BASE_DIR}/${YEAR}/"

        echo "    Remote URL: $REMOTE_DIR_URL"
        echo "    Local Dir:  $LOCAL_TARGET_DIR"

        # Create the local directory if it doesn't exist
        mkdir -p "$LOCAL_TARGET_DIR"

        # Check if the directory already contains .cdf files for this year.
        # This prevents re-downloading a full year if the script is re-run.
        # It's important to use the correct SAT_DIR in the glob check.
        if ls "${LOCAL_TARGET_DIR}/${SAT}_"*.cdf 1> /dev/null 2>&1; then
            echo "    Directory already contains .cdf files for $SAT. Skipping download for this year."
        else
            echo "    Downloading .cdf files..."
            
            # Use wget to download files recursively
            # --cut-dirs=9 is the key change to get the file into the correct local directory
            # https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/rept/sectors/rel03/2012/
            # [1] [2] [3] [4] [5]  [6]    [7]     [8]    [9]
            wget -r -np -nd -A "*.cdf" \
                 --cut-dirs=9 \
                 -P "$LOCAL_TARGET_DIR" \
                 "$REMOTE_DIR_URL" \
                 --show-progress
            
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
