#!/bin/bash

base_dir="../processed_data/tokens"

# --------------------------------------------------------------------------
# Dynamically find all directory names starting with "libero" under base_dir
# --------------------------------------------------------------------------
echo "Searching directories from $base_dir..."

# Use the `find` command to locate all matching directories
# -mindepth 1 and -maxdepth 1 ensure that only immediate subdirectories of base_dir are searched
# -type d ensures that only directories are found
# -name "libero*" ensures the directory names start with "libero"
#
# readarray (or mapfile) is a safe way to read the output of a command into an array
# The -t option removes the trailing newline characters from each line
readarray -t full_paths < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -name "libero*")

# Initialize an empty array to store plain directory names
dir_names=()
# Iterate over the paths found and use `basename` to extract the plain directory name
for path in "${full_paths[@]}"; do
    dir_names+=("$(basename "$path")")
done

# Check if any directories were found
if [ ${#dir_names[@]} -eq 0 ]; then
    echo "Warning: No subdirectories starting with 'libero' were found under '$base_dir'."
    exit 0 # Exit normally, as there is no task to execute
fi

echo "Found ${#dir_names[@]} directories, starting processing..."
# --------------------------------------------------------------------------


# Iterate over each directory name in the array
# "${dir_names[@]}" is the standard way to safely traverse all elements in an array
for dir_name in "${dir_names[@]}"
do
    # Construct the full subdirectory path and save path
    sub_record_dir="${base_dir}/${dir_name}"
    save_path="${base_dir}/${dir_name}/record.json"

    # Output the command to be executed (for debugging purposes)
    echo "Executing: python -u concate_record.py --sub_record_dir ${sub_record_dir} --save_path ${save_path}"
    
    # Directly execute the command, which is safer than using eval
    python -u concate_record.py --sub_record_dir "${sub_record_dir}" --save_path "${save_path}"

    echo "------------------------------------"
done

echo "All tasks have been completed."
