#!/bin/bash

# This script runs the data processing in PARALLEL for multiple directories.
# It launches a separate processing job for each input directory in the background
# and waits for all of them to complete.

# --- Configuration ---
INPUT_DIRS=(
    "/mnt/PLNAS/cenjun/lerobot_data/data_0623/laptop_03/grab_blocks"
    "/mnt/PLNAS/cenjun/lerobot_data/data_0623/pc_02/grab_blocks"
    "/mnt/PLNAS/cenjun/lerobot_data/data_0623/pc_03/grab_blocks"
)
# INPUT_DIRS=(
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0623/laptop_03/grab_strawberries"
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0623/pc_02/grab_strawberries"
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0623/pc_03/grab_strawberries"
# )
# INPUT_DIRS=(
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0703/laptop_03/grab_pen"
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0703/pc_02/grab_pen"
#     "/mnt/PLNAS/cenjun/lerobot_data/data_0703/pc_03/grab_pen"
# )
PYTHON_SCRIPT="save_image_action_mul_2_state.py"

# --- Sanity Checks ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# --- Helper Function ---
# This function encapsulates the logic for processing a single directory.
process_directory() {
    local input_dir="$1"
    
    # Check if the input directory exists
    if [ ! -d "$input_dir" ]; then
        echo "WARNING: Input directory not found, skipping: $input_dir"
        return
    fi
    
    # Generate the output directory by replacing 'lerobot_data' with 'processed_lerobot_data'
    local output_dir="${input_dir/lerobot_data/processed_lerobot_data_rel_state}"

    echo "[JOB STARTED] Processing $input_dir"
    echo "              -> Output will be in $output_dir"
    
    # Execute the python script.
    # The output of this specific job will be printed to the console as it runs.
    python3 "$PYTHON_SCRIPT" --input_dir "$input_dir" --output_dir "$output_dir"
    
    # Check the exit code of the python script
    if [ $? -eq 0 ]; then
        echo "[JOB FINISHED] Successfully processed $input_dir"
    else
        echo "[JOB FAILED] An error occurred while processing $input_dir"
    fi
}

# --- Main Logic ---
echo "======================================================================"
echo "Launching ${#INPUT_DIRS[@]} processing jobs in parallel..."
echo "======================================================================"

# Loop through the directories and launch each processing job in the background
for dir in "${INPUT_DIRS[@]}"; do
    # The '&' at the end runs the function in the background.
    process_directory "$dir" &
done

# The 'wait' command pauses the script here until ALL background jobs started
# from this script have finished.
echo
echo "All jobs launched. Waiting for them to complete..."
echo "NOTE: Output from different jobs will be interleaved below."
echo "======================================================================"
wait

echo "======================================================================"
echo "All parallel processing jobs have completed."
echo "======================================================================"
