import json
import os
import math
import copy
import argparse
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

    print(f"Processing data from: {base_dir}")
    print(f"Historical frames (his): {his}")
    print(f"Action chunk prediction horizon: {ACTION_CHUNK_PREDICTION_HORIZON}")
    print(f"Sub-actions per chunk: {SUB_ACTIONS_PER_CHUNK}")
    print(f"Output task name: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    specific_task_paths = [
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_02/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll-2",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll-1",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_02/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_03/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cap",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/laptop_03/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder-complete",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/pc_02/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/pc_03/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder"
    ]

    # specific_task_paths = [
    #     "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll"
    # ]

    for task_id, task_path in enumerate(tqdm(specific_task_paths, desc="Processing Tasks")):
        base_extracted_dir = "/mnt/PLNAS/cenjun/all_data/extracted/"
        relative_path = os.path.relpath(task_path, base_extracted_dir)
        task_name_readable = os.path.basename(os.path.dirname(relative_path)).replace('_', ' ')
        
        trj_list = sorted(os.listdir(task_path))
        print(trj_list)
        
        for i, trj in enumerate(tqdm(trj_list, desc=f"  - Trajectories for {task_name_readable}", leave=False)):
            print(task_path, trj)
            trj_path = os.path.join(task_path, trj)
            action_base_path = os.path.join(trj_path, 'rel_action')
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
                
                train_convs.append(conv)
        
            train_traj_count += 1
                
    print("-" * 30)
    print("Saving datasets...")

    # Updated output file naming to reflect state inclusion. "img_only" is changed to "img_state"
    train_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_train_img_state_rel_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json')
    
    with open(train_output_path, 'w') as f:
        json.dump(train_convs, f)
    print(f"Saved train conversations to: {train_output_path}")

    print("\n--- Final Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print("---------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data (with state info) to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base_dir', '-b', type=str, default="/mnt/PLNAS/cenjun/all_data/extracted/",
        help='The base directory where the Libero datasets are located.'
    )
    parser.add_argument(
        '--his', '-H', type=int, default=1,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='goal',
        help="A string used in the output JSON file names to identify the task type (e.g., 'goal', 'object')."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=512,
        help='The image resolution, used in the output JSON file names (e.g., 224, 512).'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='/mnt/PLNAS/cenjun/all_data/extracted/convs',
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