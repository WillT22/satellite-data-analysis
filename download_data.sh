#!/bin/bash

# Define the base URL (this is the parent of the nsXX directories)
BASE_URL="https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10r1/"
# Define the top-level directory where all data will be stored locally
OUTPUT_DIR="april2017storm" # Using a dedicated folder for clarity

echo "Starting download process for LANL GPS satellite data."

# Create the main output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through satellite numbers
# 'seq -w # #' generates numbers between the two numbers
for i in $(seq -w 50 80); do
    SAT_DIR="ns${i}" # Constructs the directory name, e.g., "ns01", "ns02"
    FULL_SAT_URL="${BASE_URL}${SAT_DIR}/" # Constructs the full URL for this satellite's directory
    
    echo "-> Processing satellite: $SAT_DIR (URL: $FULL_SAT_URL)"

    # wget options for each specific satellite directory:
    # -r: Recursive download within this specific satellite directory.
    # -np: No parent directory (important if there are links going up).
    # -nH: No host directory (prevents creating 'www.ngdc.noaa.gov' folder).
    # --cut-dirs=6: This is still correct for the full URL path:
    #               It removes 'stp/.../version_v1.10r1/' to leave 'nsXX/filename.txt'
    # --reject: Excludes readme and index.html files.
    # --accept: Filters files by the specific pattern (17042#).
    # -P "$OUTPUT_DIR": Saves all content into the main 'gps_satellites_data' folder.
    # "$FULL_SAT_URL": The specific URL for this satellite's directory (the target of this wget call).
    wget -r -np -nH \
         --cut-dirs=6 \
         --reject "*readme*,*README*,index.html" \
         --accept "*17042[1-5]*" \
         -P "$OUTPUT_DIR" \
         "$FULL_SAT_URL"

    # Add a small delay between requests if hammering the server is a concern
    # sleep 1
done

echo "All specified satellite data download attempts complete."
