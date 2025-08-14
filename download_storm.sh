#!/bin/bash

# --- Configuration ---
BASE_REMOTE_URL="https://cdaweb.gsfc.nasa.gov/pub/data/rbsp"
LOCAL_BASE_DIR="/home/will/REPT_data"

# Define a specific satellite, time period, and output folder
SATELLITES=("rbspa" "rbspb")
LOCAL_TARGET_DIR="${LOCAL_BASE_DIR}/april2017storm"

# Time period for download (YYYYMMDD format)
# The script will download all files from START_DATE up to (and including) END_DATE.
START_DATE="20170421"
END_DATE="20170425"

# --- Main Script ---

echo "Starting download process for RBSP REPT satellite data."
echo "Satellite: $SAT_NAME"
echo "Time Period: $START_DATE to $END_DATE"
echo "Local directory: $LOCAL_TARGET_DIR"

# Create the main output directory if it doesn't exist
mkdir -p "$LOCAL_TARGET_DIR"

# Convert start and end dates to seconds since the epoch for a simple loop
# date -d "YYYYMMDD" +%s converts a date string to seconds since epoch
START_SECS=$(date -d "$START_DATE" +%s)
END_SECS=$(date -d "$END_DATE" +%s)
ONE_DAY_SECS=86400

# Loop through each satellite
for SAT in "${SATELLITES[@]}"; do
    echo -e "\n--- Processing Satellite: $SAT ---"
    # Loop through each day in the specified time period
    for (( i=START_SECS; i<=END_SECS; i+=ONE_DAY_SECS )); do
        # Convert the current day (in seconds) back to YYYYMMDD and YYYY formats
        DATE_YMD=$(date -d "@$i" +%Y%m%d)
        YEAR=$(date -d "@$i" +%Y)
        
        echo "  Processing date: $DATE_YMD"

        # Construct the remote URL for the year's directory
        # Example: https://cdaweb.gsfc.nasa.gov/pub/data/rbsp/rbspa/l3/ect/rept/sectors/rel03/2017/
        REMOTE_DIR_URL="${BASE_REMOTE_URL}/${SAT}/l3/ect/rept/sectors/rel03/${YEAR}/"
        
        # Construct the filename pattern for the specific day, with a wildcard for the version
        # Example: rbspa_rel03_ect-rept-sci-l3_20170421_v*.cdf
        FILENAME_PATTERN="${SAT}_rel03_ect-rept-sci-l3_${DATE_YMD}_v*.cdf"

        echo "    Remote URL: $REMOTE_DIR_URL"
        echo "    Filename Pattern: $FILENAME_PATTERN"

        # Check if the file already exists locally. This prevents re-downloading.
        if ls "${LOCAL_TARGET_DIR}/${FILENAME_PATTERN}" 1> /dev/null 2>&1; then
            echo "    File already exists locally. Skipping download for this date."
        else
            echo "    Downloading .cdf file for $DATE_YMD..."
            
            # Use wget to download the specific file
            # -r -np -nd: still used for the recursive wget call on the remote URL
            # --cut-dirs=9: cuts the path to save the file in the LOCAL_TARGET_DIR
            # --accept: only downloads the file matching our pattern
            wget -r -np -nd \
                --cut-dirs=9 \
                --accept "$FILENAME_PATTERN" \
                -P "$LOCAL_TARGET_DIR" \
                "$REMOTE_DIR_URL" \
                --show-progress
            
            # Check wget's exit status
            if [ $? -eq 0 ]; then
                echo "    Download successful for $DATE_YMD."
            else
                echo "    WARNING: Download failed for $DATE_YMD. Check URL or network connection."
            fi
        fi
    done
done

echo -e "\nAll downloads attempted. Script finished."
