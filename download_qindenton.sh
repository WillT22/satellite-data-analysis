#!/bin/bash

# Define the base URL of the QinDenton data
BASE_URL="https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/"
# Define the top-level directory where all data will be stored locally
OUTPUT_DIR="QinDenton"

echo "Starting download process to create QinDenton/YEAR/ structure..."

# Create the main top-level directory for all data
# The -p flag ensures that if the directory already exists, no error is thrown.
mkdir -p "$OUTPUT_DIR"

# Loop through each year from 1995 to 2022 to download data
# This loop is crucial for targeting each year's specific directory on the server.
for year in $(seq 1995 2022); do
    echo "Attempting to download 5min and hour data for year: $year"

    # The wget command explained for creating year directories:
    #
    # -r: --recursive
    #     This makes wget follow links into subdirectories (e.g., from the year folder
    #     to download the actual data files within it).
    #
    # -np: --no-parent
    #     Prevents wget from ascending to parent directories (e.g., from 1995/ back to QinDenton/
    #     and then to data_pub/, which we don't want).
    #
    # -nH: --no-host-directories
    #     Prevents wget from creating a top-level directory named after the host
    #     (e.g., 'rbsp-ect.newmexicoconsortium.org'). We want 'QinDenton' as our top level.
    #
    # --cut-dirs=2: This is the most critical part for your desired structure.
    #     It tells wget to 'cut' the first 3 directory components from the *remote URL path*
    #     when constructing the *local file path*.
    #     Let's trace for a URL like: https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/1995/QinDenton_19950101_00_5min.txt
    #     1. data_pub/                     (cut)
    #     2. QinDenton/                    (cut)
    #     What remains of the original path is: 1995/QinDenton_19950101_00_5min.txt
    #     This remaining path is then appended to the directory specified by -P.
    #
    # -A "*_5min.txt","*_hour.txt": --accept-regex
    #     This filters the files, ensuring only those ending with '_5min.txt' or '_hour.txt' are downloaded.
    #
    # -P "$OUTPUT_DIR": --directory-prefix
    #     This specifies the *root* directory where the downloaded files and structure
    #     will be placed. Combined with --cut-dirs=3, it means the '1995/' etc. folders
    #     will be created directly inside this "$OUTPUT_DIR".
    #
    # "${BASE_URL}${year}/": The target URL for this specific wget call.
    #     By appending "$year/" to the base URL, we tell wget to start its recursive
    #     download directly from *within* that year's directory on the server.
    #     Example: https://rbsp-ect.newmexicoconsortium.org/data_pub/QinDenton/1995/
    #     This ensures that only content *from that year's folder* is considered.
    wget -r -np -nH --cut-dirs=2 \
         -A "*_5min.txt","*_hour.txt" \
         -P "$OUTPUT_DIR" \
         "${BASE_URL}${year}/"
    echo "Finished download attempt for year: $year"
done

echo "Download phase complete. Files are now located in QinDenton/YEAR/ directory structure."
echo "Starting final file organization into 5min/ and hour/ subfolders..."

# Change into the main QinDenton directory for post-processing
cd "$OUTPUT_DIR" || { echo "Error: Could not enter $OUTPUT_DIR. Exiting."; exit 1; }

# Loop through each year directory that was created by wget
# The '*' glob expands to directory names in alphabetical order, which sorts years numerically.
for year_dir in */; do
    if [ -d "$year_dir" ]; then # Check if it's a directory
        echo "Organizing files within: $year_dir"
        cd "$year_dir"
        mkdir -p 5min hour
        mv -f *_5min.txt 5min/ 2>/dev/null
        mv -f *_hour.txt hour/ 2>/dev/null
        cd .. # Go back to the QinDenton directory
    fi
done

echo "All data downloaded and organized successfully in the '$OUTPUT_DIR' directory."
