import json
import numpy as np # Still included, though not directly used for numpy operations on data here
import os
import math
import copy
import argparse # Import the argparse module

def process_libero_data(
    base_dir: str,
    his: int,
    len_action: int,
    task_name_for_output: str,
    resolution: int,
    with_state: bool,
    img_names: list,
    output_dir: str
):
    """
    Processes Libero robot trajectory data to create conversational datasets for
    training and validation (in-distribution and out-of-distribution).

    Args:
        base_dir (str): The base directory where the Libero datasets are located.
        his (int): The number of historical image frames to include in each conversation.
        len_action (int): The number of future action steps to predict.
        task_name_for_output (str): A string used in the output JSON file names to
                                    identify the task type (e.g., 'goal', 'object').
        resolution (int): The image resolution, used in the output JSON file names.
        output_dir (str): The directory where the generated JSON dataset files and
                          the summary JSON file will be saved.
    """

    train_convs = []
    val_convs_ind = []
    val_convs_ood = []
    all_convs = []

    train_traj_count = 0
    val_ind_traj_count = 0
    val_ood_traj_count = 0

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    task_list = sorted(os.listdir(base_dir))
    # Split for OOD tasks (10% for OOD validation, i.e., first 90% for train/val_ind)
    split_index_ood = math.ceil(len(task_list) * 0.9)

    print(f"Processing data from: {base_dir}")
    print(f"Historical frames (his): {his}")
    print(f"Action prediction length (len_action): {len_action}")
    print(f"Output task name: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"With state: {with_state}")
    print(f"Image list: {img_names}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    for task_id, task in enumerate(task_list):
        task_path = os.path.join(base_dir, task)
        # Assuming task names are like "put_apple_in_bowl" -> "put apple in bowl"
        task_name_readable = task.replace('_', ' ')
        
        trj_list = sorted(os.listdir(task_path))
        # Split for In-Distribution validation within each task (10% for IND validation)
        split_index_ind = math.ceil(len(trj_list) * 0.9)
        
        for i, trj in enumerate(trj_list):
            trj_path = os.path.join(task_path, trj)
            action_path = os.path.join(trj_path, 'action')
            # imgs_path = os.path.join(trj_path, 'imgs')
            imgs_paths = [os.path.join(trj_path, name) for name in img_names]
            if with_state:
                state_path = os.path.join(trj_path, 'eef_gripper_state')    # TODO: 根据需要统一修改 eef_gripper_state
            
            skip_flag = False
            # Check if action and imgs directories exist
            if not os.path.exists(action_path):
                print(f"    Warning: Missing 'action' directory in {trj_path}. Skipping.")
                skip_flag = True

            for imgs_path in imgs_paths:
                if not os.path.exists(imgs_path):
                    print(f"    Warning: Missing 'imgs' directory in {trj_path}. Skipping.")
                    skip_flag = True

            if with_state and (not os.path.exists(state_path)):
                print(f"    Warning: Missing 'state' directory in {trj_path}. Skipping.")
                skip_flag = True

            if skip_flag:
                continue
            
            img_list = []
            action_list = []
            if with_state:
                state_list = []
            
            # Robustly collect image and action file paths by sorting them
            action_files_raw = [f for f in os.listdir(action_path) if f.startswith('action_') and f.endswith('.npy')]
            # img_files_raw = [f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')]
            img_files_raw_list = [[f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')] for imgs_path in imgs_paths]
            if with_state:
                state_files_raw = [f for f in os.listdir(state_path) if f.startswith('eef_gripper_state_') and f.endswith('.npy')]

            # Extract numeric parts and sort
            action_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in action_files_raw])
            # img_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw])
            img_indices_list = [sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw]) for img_files_raw in img_files_raw_list]
            if with_state:
                state_indices = sorted([int(f.split('_')[-1].split('.')[0]) for f in state_files_raw])

            # Find common indices
            common_indices_sets = [set(action_indices)]
            for img_indices in img_indices_list:
                common_indices_sets.append(set(img_indices))
            common_indices = sorted(list(set.intersection(*common_indices_sets)))

            if not common_indices:
                print(f"    Warning: No matching action/image file pairs found in {trj_path}. Skipping.")
                continue

            for idx in common_indices:
                action_file = os.path.join(action_path, f"action_{idx}.npy")
                # img_file = os.path.join(imgs_path, f"image_{idx}.png")
                img_files = [os.path.join(imgs_path, f"image_{idx}.png") for imgs_path in imgs_paths]
                if with_state:
                    state_file = os.path.join(state_path, f"eef_gripper_state_{idx}.npy")
                
                # Double check if files actually exist (though common_indices ensures they should have existed at discovery)
                if os.path.exists(action_file):
                    action_list.append(action_file)
                else:
                    print(f"      Warning: File missing despite index found: {action_file}. Skipping this index.")

                for img_file in img_files:
                    if os.path.exists(img_file):
                        img_list.append(img_file)
                    else:
                        print(f"      Warning: File missing despite index found: {img_file}. Skipping this index.")

                if with_state:
                    if os.path.exists(state_file):
                        state_list.append(state_file)
                    else:
                        print(f"      Warning: File missing despite index found: {state_file}. Skipping this index.")

        
            if not img_list or not action_list:
                print(f"    Warning: No valid image/action pairs found in {trj_path} after filtering. Skipping.")
                continue

            # Generate conversation samples for each step in the trajectory
            for j in range(len(action_list)):
                # Determine the start index for historical images
                img_history_start_idx = max(0, j - his + 1)
                img_c = copy.deepcopy(img_list[img_history_start_idx * len(img_names) : (j + 1) * len(img_names)])
                
                # Determine the end index for future actions
                action_c = copy.deepcopy(action_list[j : min(j + len_action, len(action_list))])
                
                # Skip if we don't have enough future actions for the required len_action
                if len(action_c) < len_action:
                    continue # This sample cannot be fully formed

                if with_state:
                    state_c = copy.deepcopy(state_list[j : j + 1])
                    human_val = f"What action should the robot take to {task_name_readable}?" + "<|state|>" * len(state_c) + "<|image|>" * len(img_c)
                else:
                    human_val = f"What action should the robot take to {task_name_readable}?" + "<|image|>" * len(img_c)
                
                conv = {
                    "conversations":[
                        {
                            "from": "human",
                            "value": human_val
                        },
                        {
                            "from": "gpt",
                            "value": "<|action|>" * len(action_c)
                        },
                    ],
                    "image": img_c,
                    "action": action_c,
                }

                # State
                if with_state:
                    conv["state"] = state_c
                
                # Assign to appropriate dataset split based on task_id and trajectory_id
                if task_id < split_index_ood and i < split_index_ind:
                    train_convs.append(conv)
                elif task_id < split_index_ood and i >= split_index_ind:
                    val_convs_ind.append(conv)
                else:
                    val_convs_ood.append(conv)
                all_convs.append(conv)
        
            # Increment trajectory counts for statistics
            if task_id < split_index_ood and i < split_index_ind:
                train_traj_count += 1
            elif task_id < split_index_ood and i >= split_index_ind:
                val_ind_traj_count += 1
            else:
                val_ood_traj_count += 1
                
    print("-" * 30)
    print("Saving datasets...")

    # Define output file names using the parameters
    img_item = '_'.join([item.replace('imgs_', '') for item in img_names])
    state_item = 'w_state' if with_state else 'wo_state'
    train_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_train_{img_item}_{state_item}_{len_action}_{resolution}.json')
    val_ind_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_val_ind_{img_item}_{state_item}_{len_action}_{resolution}.json')
    val_ood_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_val_ood_{img_item}_{state_item}_{len_action}_{resolution}.json')
    all_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_all_{img_item}_{state_item}_{len_action}_{resolution}.json')

    # Save training set
    with open(train_output_path, 'w') as f:
        json.dump(train_convs, f, indent=2) # Use indent for readability
    print(f"Saved train conversations to: {train_output_path}")

    # Save validation in-distribution set
    with open(val_ind_output_path, 'w') as f:
        json.dump(val_convs_ind, f, indent=2)
    print(f"Saved val_ind conversations to: {val_ind_output_path}")

    # Save validation out-of-distribution set
    with open(val_ood_output_path, 'w') as f:
        json.dump(val_convs_ood, f, indent=2)
    print(f"Saved val_ood conversations to: {val_ood_output_path}")

    print("\n--- Final Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print(f"Validation In-Distribution trajectories: {val_ind_traj_count}, conversations: {len(val_convs_ind)}")
    print(f"Validation Out-of-Distribution trajectories: {val_ood_traj_count}, conversations: {len(val_convs_ood)}")
    print("---------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Required argument
    parser.add_argument(
        '--base_dir', '-b', type=str, required=True,
        help='The base directory where the Libero datasets are located'
    )

    # Optional arguments with default values
    parser.add_argument(
        '--his', '-H', type=int, default=2,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--len_action', '-L', type=int, default=5,
        help='The number of future action steps to predict.'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='goal',
        help="A string used in the output JSON file names to identify the task type (e.g., 'goal', 'object')."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=512,
        help='The image resolution, used in the output JSON file names (e.g., 256, 512).'
    )
    parser.add_argument(
        '--with_state', action='store_true',
        help='If True, with state.'
    )
    parser.add_argument(
        '--img_names', nargs='+', default=['imgs_third_view'], choices=['imgs_wrist', 'imgs_third_view'],
        help='List of image names to include (imgs_wrist and/or imgs_third_view)')
    parser.add_argument(
        '--output_dir', '-o', type=str, default='./generated_libero_convs/',
        help='The directory where the generated JSON dataset files and the summary JSON file will be saved. Will be created if it does not exist.'
    )

    args = parser.parse_args()

    # Call the processing function with parsed arguments
    process_libero_data(
        base_dir=args.base_dir,
        his=args.his,
        len_action=args.len_action,
        task_name_for_output=args.task_name, # Map 'task_name' from args to 'task_name_for_output' in function
        resolution=args.resolution,
        with_state=args.with_state,
        img_names=args.img_names,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()     