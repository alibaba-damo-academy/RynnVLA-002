#!/bin/bash

# Define arrays
types=("goal")
# types=("object")
# types=("spatial")
# types=("10")
splits=("train" "val_ind" "val_ood")

# Define resolution variables, can be 256 or 512
# If you only want to run one, you can keep only one value, e.g., sizes=("256")
# sizes=("256")
sizes=("512")


# Base directory
base_dir="../processed_data"

# Loop through all sizes
for size in "${sizes[@]}"
do
    # Loop through all types
    for type in "${types[@]}"
    do
        # Loop through all splits
        for split in "${splits[@]}"
        do
            # --- First directory type: his_2 ---
            # Construct directory name, replacing 256 with the variable $size
            dir_name_his2="libero_${type}_his_2_${split}_img_only_ck_5_${size}"
            
            # Construct the full command
            command_his2="python -u concate_record.py --sub_record_dir ${base_dir}/${dir_name_his2} --save_path ${base_dir}/${dir_name_his2}/record.json"
            
            # Output the command (optional, for debugging)
            echo "Executing: $command_his2"
            
            # Execute the command
            eval "$command_his2"

            echo "" # Add an empty line to separate output

            # --- Second directory type: his_1 ---
            # Construct directory name, replacing 256 with the variable $size
            dir_name_his1="libero_${type}_his_1_${split}_a2i_${size}"
            
            # Construct the full command
            command_his1="python -u concate_record.py --sub_record_dir ${base_dir}/${dir_name_his1} --save_path ${base_dir}/${dir_name_his1}/record.json"
            
            # Output the command (optional, for debugging)
            echo "Executing: $command_his1"
            
            # Execute the command
            eval "$command_his1"
            echo "------------------------------------" # Separator for different combinations
        done
    done
done
