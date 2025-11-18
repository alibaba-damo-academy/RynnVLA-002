import json
import numpy as np # Still included, though not directly used for numpy operations on data here
import os
import math
import copy
import argparse # Import the argparse module
from tqdm import tqdm

def process_libero_data(
    base_dir: str,
    his: int,
    task_name_for_output: str,
    resolution: int,
    output_dir: str
):
    """
    Processes Libero robot trajectory data to create conversational datasets.
    This version includes state information (state_n.npy) alongside images.
    For each observation at timestep 'i', this script generates a sample
    to predict the action at timestep 'i', which consists of 20 sub-actions.

    Args:
        base_dir (str): The base directory where the Libero datasets are located.
                        Each trajectory folder is expected to contain 'action',
                        'front_image', 'wrist_image', and 'state' subdirectories.
        his (int): The number of historical image frames to include in each conversation.
        task_name_for_output (str): A string used in the output JSON file names to
                                    identify the task type (e.g., 'goal', 'object').
        resolution (int): The image resolution, used in the output JSON file names.
        output_dir (str): The directory where the generated JSON dataset files will be saved.
    """
    # --- CONSTANTS based on the new understanding ---
    ACTION_CHUNK_PREDICTION_HORIZON = 1 # We always predict 1 action chunk (the current one)
    SUB_ACTIONS_PER_CHUNK = 20          # Each action chunk has 20 .npy files

    train_convs = []
    val_convs_ind = []
    val_convs_ood = []

    train_traj_count = 0
    val_ind_traj_count = 0
    val_ood_traj_count = 0

    os.makedirs(output_dir, exist_ok=True)

    task_list = sorted(os.listdir(base_dir))
    split_index_ood = math.ceil(len(task_list) * 1) # Use all tasks for now

    print(f"Processing data from: {base_dir}")
    print(f"Historical frames (his): {his}")
    print(f"Action chunk prediction horizon: {ACTION_CHUNK_PREDICTION_HORIZON}")
    print(f"Sub-actions per chunk: {SUB_ACTIONS_PER_CHUNK}")
    print(f"Output task name: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    for task_id, task in enumerate(tqdm(task_list, desc="Processing Tasks")):
        task_path = os.path.join(base_dir, task)
        task_name_readable = task.replace('_', ' ').replace('-', ' ')
        
        trj_list = sorted(os.listdir(task_path))
        split_index_ind = math.ceil(len(trj_list) * 1)
        
        for i, trj in enumerate(tqdm(trj_list, desc=f"  - Trajectories for {task_name_readable}", leave=False)):
            trj_path = os.path.join(task_path, trj)
            action_base_path = os.path.join(trj_path, 'action')
            imgs_path = os.path.join(trj_path, 'front_image')
            imgs_path_w = os.path.join(trj_path, 'wrist_image')
            state_path = os.path.join(trj_path, 'state') # Added state path
            
            # Check for existence of all required directories
            if not all(os.path.exists(p) for p in [action_base_path, imgs_path, imgs_path_w, state_path]):
                print(f"    Warning: Missing required directories in {trj_path}. Skipping.")
                continue
            
            try:
                # List and parse indices from all data sources
                action_dirs_raw = [d for d in os.listdir(action_base_path) if d.startswith('action_') and os.path.isdir(os.path.join(action_base_path, d))]
                img_files_raw = [f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')]
                state_files_raw = [f for f in os.listdir(state_path) if f.startswith('state_') and f.endswith('.npy')]

                action_indices = sorted([int(d.split('_')[1]) for d in action_dirs_raw])
                img_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw])
                state_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in state_files_raw])

            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not parse indices in {trj_path}. Error: {e}. Skipping.")
                continue

            # Find common indices across all data modalities
            common_indices = sorted(list(set(action_indices) & set(img_indices) & set(state_indices)))
            if not common_indices:
                continue

            img_list, img_list_w, action_list, state_list = [], [], [], []

            for idx in common_indices:
                img_file = os.path.join(imgs_path, f"image_{idx}.png")
                img_file_w = os.path.join(imgs_path_w, f"image_{idx}.png")
                action_dir = os.path.join(action_base_path, f"action_{idx}")
                state_file = os.path.join(state_path, f"state_{idx}.npy") # Added state file path
                
                if os.path.exists(img_file) and os.path.exists(img_file_w) and os.path.isdir(action_dir) and os.path.exists(state_file):
                    try:
                        sub_action_files_raw = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
                        sub_action_files_sorted = sorted(sub_action_files_raw, key=lambda f: int(os.path.splitext(f)[0]))
                        
                        if len(sub_action_files_sorted) == SUB_ACTIONS_PER_CHUNK:
                            sub_action_paths = [os.path.join(action_dir, f) for f in sub_action_files_sorted]
                            img_list.append(img_file)
                            img_list_w.append(img_file_w)
                            action_list.append(sub_action_paths)
                            state_list.append(state_file) # Append the state file path
                        else:
                            print(f"      Warning: Action dir {action_dir} has {len(sub_action_files_sorted)} files, expected {SUB_ACTIONS_PER_CHUNK}. Skipping index {idx}.")
                    except (ValueError, FileNotFoundError) as e:
                         print(f"      Warning: Error processing action dir {action_dir}: {e}. Skipping index {idx}.")

            if not img_list or not action_list or not state_list:
                continue

            # Generate conversation samples. The logic is now 1-to-1.
            # Observation at step 'j' predicts action at step 'j'.
            for j in range(len(action_list)): # Loop through each available timestep
                img_history_start_idx = max(0, j - his + 1)
                img_c = copy.deepcopy(img_list[img_history_start_idx : j + 1])
                img_c_w = copy.deepcopy(img_list_w[img_history_start_idx : j + 1])
                
                # The target action is simply the list of 20 sub-actions at the current step 'j'.
                action_c = copy.deepcopy(action_list[j])
                # The state is the single state file path at the current step 'j'.
                state_c = copy.deepcopy(state_list[j])

                # This check is redundant if the data is clean, but good for safety.
                if len(action_c) != SUB_ACTIONS_PER_CHUNK:
                    continue

                conv = {
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"What action should the robot take to {task_name_readable}?" + "<|state|>" + "<|image|>" * len(img_c) * 2
                        },
                        {
                            "from": "gpt",
                            # The number of action tokens is fixed to the number of sub-actions.
                            "value": "<|action|>" * SUB_ACTIONS_PER_CHUNK
                        },
                    ],
                    "image": img_c + img_c_w,
                    "action": action_c, # This is the list of 20 sub-action paths.
                    "state": state_c # This is the path to the state.npy file.
                }
                
                if task_id < split_index_ood and i < split_index_ind:
                    train_convs.append(conv)
                elif task_id < split_index_ood and i >= split_index_ind:
                    val_convs_ind.append(conv)
                else:
                    val_convs_ood.append(conv)
        
            if task_id < split_index_ood and i < split_index_ind:
                train_traj_count += 1
            elif task_id < split_index_ood and i >= split_index_ind:
                val_ind_traj_count += 1
            else:
                val_ood_traj_count += 1
                
    print("-" * 30)
    print("Saving datasets...")

    # Updated output file naming to reflect state inclusion. "img_only" is changed to "img_state"
    train_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_train_img_state_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json')
    val_ind_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_val_ind_img_state_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json')
    val_ood_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_val_ood_img_state_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json')
    
    with open(train_output_path, 'w') as f:
        json.dump(train_convs, f)
    print(f"Saved train conversations to: {train_output_path}")

    with open(val_ind_output_path, 'w') as f:
        json.dump(val_convs_ind, f)
    print(f"Saved val_ind conversations to: {val_ind_output_path}")

    with open(val_ood_output_path, 'w') as f:
        json.dump(val_convs_ood, f)
    print(f"Saved val_ood conversations to: {val_ood_output_path}")

    print("\n--- Final Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print(f"Validation In-Distribution trajectories: {val_ind_traj_count}, conversations: {len(val_convs_ind)}")
    print(f"Validation Out-of-Distribution trajectories: {val_ood_traj_count}, conversations: {len(val_convs_ood)}")
    print("---------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data (with state info) to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base_dir', '-b', type=str, required=True,
        help='The base directory where the Libero datasets are located.'
    )
    parser.add_argument(
        '--his', '-H', type=int, default=2,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='goal',
        help="A string used in the output JSON file names to identify the task type (e.g., 'goal', 'object')."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=224,
        help='The image resolution, used in the output JSON file names (e.g., 224, 512).'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='./generated_libero_convs/',
        help='The directory where the generated JSON dataset files will be saved. Will be created if it does not exist.'
    )

    args = parser.parse_args()

    process_libero_data(
        base_dir=args.base_dir,
        his=args.his,
        task_name_for_output=args.task_name,
        resolution=args.resolution,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
