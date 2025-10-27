#!/bin/bash

# --- CONFIGURATION ---
# Define the start and end dates for the data you need (YYYYMMDD format)
START_DATE="20170421"
END_DATE="20170425"
STORM_ID="april2017storm" # Used for the output directory name

# Define URL and Local Paths
BASE_URL="https://www.ngdc.noaa.gov/stp/space-weather/satellite-data/satellite-systems/lanl_gps/version_v1.10r1/"
OUTPUT_DIR="/home/wzt0020/GPS_data/${STORM_ID}"
# ---------------------

echo "Starting download process for LANL GPS data from ${START_DATE} to ${END_DATE}"
mkdir -p "$OUTPUT_DIR"

# --- EMBEDDED PYTHON SCRIPT TO CALCULATE FILENAME PATTERNS ---
# This function calculates the YYYYMMDD patterns needed for wget's --accept flag.
DATE_PATTERNS=$(python3 << END_PYTHON
import datetime as dt
import numpy as np

# Input dates (from Bash variables)
start_date_str = "$START_DATE"
end_date_str = "$END_DATE"
start_date = dt.datetime.strptime(start_date_str, "%Y%m%d").date()
end_date = dt.datetime.strptime(end_date_str, "%Y%m%d").date()

# FIX: Find the beginning of the week for the START_DATE (assuming Sunday is start of week = 0 in Geodesy)
# Monday is 0, Sunday is 6 (Python's weekday() convention).
# We want to shift back to the nearest Sunday (weekday 6) or the current date if it's Sunday.
# Python's weekday: Monday=0, Sunday=6. We use isoweekday: Monday=1, Sunday=7.

# Calculate days to subtract to reach the previous Sunday (using Monday=1, Sunday=7 convention)
# days_to_subtract = (start_date.isoweekday() % 7) 
# The GPS week typically starts on Sunday (day 0 or day 7) and ends on Saturday.
# A common simplification is to use the start of the week: Sunday is day 6 in weekday().
# If Monday (0) is start, subtract 0. If Sunday (6), subtract 6.

# For standard space physics week (Sunday start, Saturday end):
# Python's weekday(): Mon=0, Tue=1, ..., Sun=6.
# Days to go back: 
# If Monday (0), go back 1 day (to previous Sunday).
# If Sunday (6), go back 0 days.

days_since_sunday = (start_date.weekday() + 1) % 7 # 0=Sunday, 1=Monday, ..., 6=Saturday
days_to_subtract = days_since_sunday 
                                     
# Calculate the actual Sunday start date
true_start_date = start_date - dt.timedelta(days=days_to_subtract)

# Calculate the range of days needed, extending from the calculated Sunday
date_list = [true_start_date + dt.timedelta(days=i) 
             for i in range((end_date - true_start_date).days + 1)]

# The GPS files are named by YYMMDD (Year + Month + Day).
patterns = []
for date in date_list:
    patterns.append(date.strftime("%y%m%d")) 

# Join patterns into a single comma-separated string for Bash consumption
print(",".join(patterns))
END_PYTHON
)
# --- END PYTHON SCRIPT ---

# Check if Python successfully generated patterns
if [ -z "$DATE_PATTERNS" ]; then
    echo "ERROR: Python script failed to generate date patterns or the date range is invalid."
    exit 1
fi

echo "Targeting date patterns: ${DATE_PATTERNS}"

# --- WGET EXECUTION LOOP ---
# Loop through all satellite numbers (ns50 to ns80)
for i in $(seq -w 50 80); do
    SAT_DIR="ns${i}"
    FULL_SAT_URL="${BASE_URL}${SAT_DIR}/"
    LOCAL_SAT_DIR="${OUTPUT_DIR}/${SAT_DIR}"

    echo "-> Processing satellite: $SAT_DIR (URL: $FULL_SAT_URL)"
    
    # We construct a robust accept pattern list for wget
    # Example: If DATE_PATTERNS is 170421,170422, the final accept string is: *170421*,*170422*
    ACCEPT_PATTERNS=$(echo $DATE_PATTERNS | tr ',' '\n' | sed 's/^/*/' | sed 's/$/\*/' | tr '\n' ',')
    # Remove trailing comma for clean execution
    ACCEPT_PATTERNS=${ACCEPT_PATTERNS%,}

    wget -r -np -nH \
         --cut-dirs=6 \
         --reject "*readme*,*README*,index.html" \
         --accept "${ACCEPT_PATTERNS}" \
         -P "$OUTPUT_DIR" \
         "$FULL_SAT_URL"
done

echo "All specified satellite data download attempts complete."

# Remove any empty satellite directories
find "$OUTPUT_DIR" -type d -empty -delete
echo "Cleanup complete."