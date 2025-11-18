import json
import numpy as np # Keep this import, although action.npy paths are stored, not loaded, in this snippet.
import os
import math
import copy
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Required argument
    parser.add_argument(
        '--base_dir', '-b', type=str, required=True,
        help='The base directory where the Libero datasets are located (e.g., /mnt/nas_jianchong/datasets/EmbodiedAI/Libero/libero_spatial_m_a_i_2_512)'
    )

    # Optional arguments with default values
    parser.add_argument(
        '--his', '-H', type=int, default=2,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='spatial', # Default based on the example base_dir
        help="A string used in the output JSON file names to identify the task type (e.g., 'spatial', 'goal', 'object')."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=512,
        help='The image resolution, used in the output JSON file names (e.g., 256, 512).'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='./generated_libero_convs/',
        help='The directory where the generated JSON dataset files will be saved. Will be created if it does not exist.'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Use arguments
    base_dir = args.base_dir
    his = args.his
    # len_action is defined but not directly used in the conversation generation logic
    # as the current script structure predicts a single next image.
    task_name_for_filename = args.task_name
    resolution_for_filename = args.resolution

    train_convs = []
    val_convs_ind = []
    val_convs_ood = []

    # Initialize trajectory counters for logging
    train_traj_count = 0
    val_ind_traj_count = 0
    val_ood_traj_count = 0

    task_list = sorted(os.listdir(base_dir))
    split_index_ood = math.ceil(len(task_list) * 1)

    print(f"Starting data processing from: {base_dir}")
    print(f"Historical frames (his): {his}")
    print(f"Output files will use task_name: '{task_name_for_filename}' and resolution: {resolution_for_filename}")
    print(f"Output directory: {args.output_dir}")

    for task_id, task in enumerate(task_list):
        task_path = os.path.join(base_dir, task)
        
        trj_list = sorted(os.listdir(task_path))
        split_index_ind = math.ceil(len(trj_list) * 1)
        
        # Track trajectories processed within this task for accumulation
        current_task_train_traj_count = 0
        current_task_val_ind_traj_count = 0
        current_task_val_ood_traj_count = 0
        
        for i, trj in tqdm(enumerate(trj_list)):
            trj_path = os.path.join(task_path, trj)
            action_path = os.path.join(trj_path, 'action')
            imgs_path = os.path.join(trj_path, 'front_image')
            
            img_list = []
            action_list = []
            
            # Determine the number of frames by checking the action directory (assuming paired files)
            num_frames = len(os.listdir(action_path))
            for j in range(num_frames):
                action_file = os.path.join(action_path, f"action_{j}.npy")
                img_file = os.path.join(imgs_path, f"image_{j}.png")
                
                # Check if files exist to ensure data integrity for this frame
                if os.path.exists(action_file) and os.path.exists(img_file):
                    img_list.append(img_file)
                    action_list.append(action_file)
                else:
                    print(f"Warning: Missing file(s) for frame {j} in trajectory {trj_path}. Stopping processing for this trajectory.")
                    break # Stop processing this trajectory if files are missing

            # A conversation requires 'his' history frames + 1 future frame.
            # So, we need at least 'his + 1' total frames in img_list.
            if len(img_list) < his + 1:
                print(f"Warning: Trajectory {trj_path} has only {len(img_list)} frames, which is insufficient for 'his'={his} (requires at least {his+1} frames). Skipping this trajectory.")
                continue # Skip to the next trajectory

            # Iterate through frames to create conversations
            # Loop up to `len(action_list) - 1` because we need `j+1` for the future image.
            for j in range(len(action_list) - 1):
                
                # Historical images: [I_{j-his+1}, ..., I_j]
                img_c_h = copy.deepcopy(img_list[max(j - his + 1, 0) : j + 1])
                # Historical actions: [A_{j-his+1}, ..., A_j]
                action_c = copy.deepcopy(action_list[max(j - his + 1, 0) : j + 1])
                # Future image: [I_{j+1}] (always one next image)
                img_c_f = copy.deepcopy(img_list[j + 1 : j + 2])
                
                conv = {
                    "conversations":[
                        {
                            "from": "human",
                            # Revised prompt to accurately reflect variable 'his' length
                            "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>" * len(img_c_h)
                        },
                        {
                            "from": "gpt",
                            "value": "<|image|>" # The model generates a single image
                        },
                    ],
                    # 'image' field contains all images referenced in the conversation:
                    # [historical_images..., predicted_future_image]
                    "image": img_c_h + img_c_f,
                    # 'action' field contains historical actions corresponding to img_c_h
                    "action": action_c,
                }
                
                # Assign conversations to the correct split
                if task_id < split_index_ood: # In-distribution tasks
                    if i < split_index_ind: # In-distribution trajectories for training
                        train_convs.append(conv)
                    else: # In-distribution trajectories for validation
                        val_convs_ind.append(conv)
                else: # Out-of-distribution tasks
                    val_convs_ood.append(conv)
        
            # Increment trajectory counters for the current task based on their split
            if task_id < split_index_ood:
                if i < split_index_ind:
                    train_traj_count += 1
                else:
                    val_ind_traj_count += 1
            else:
                val_ood_traj_count += 1
                
    # Construct output file paths using argparse values
    train_output_filename = f'libero_{task_name_for_filename}_his_{his}_train_a2i_{resolution_for_filename}.json'
    val_ind_output_filename = f'libero_{task_name_for_filename}_his_{his}_val_ind_a2i_{resolution_for_filename}.json'
    val_ood_output_filename = f'libero_{task_name_for_filename}_his_{his}_val_ood_a2i_{resolution_for_filename}.json'

    train_output_path = os.path.join(args.output_dir, train_output_filename)
    val_ind_output_path = os.path.join(args.output_dir, val_ind_output_filename)
    val_ood_output_path = os.path.join(args.output_dir, val_ood_output_filename)

    # Save datasets
    print(f"\nSaving training data to: {train_output_path}")
    with open(train_output_path, 'w') as f:
        json.dump(train_convs, f, indent=4) # Use indent for readability

    print(f"Saving validation In-Distribution data to: {val_ind_output_path}")
    with open(val_ind_output_path, 'w') as f:
        json.dump(val_convs_ind, f, indent=4)

    print(f"Saving validation Out-of-Distribution data to: {val_ood_output_path}")
    with open(val_ood_output_path, 'w') as f:
        json.dump(val_convs_ood, f, indent=4)

    print("\n--- Dataset Generation Summary ---")
    print(f"Train trajectories: {train_traj_count}, conversations: {len(train_convs)}")
    print(f"Validation In-Distribution trajectories: {val_ind_traj_count}, conversations: {len(val_convs_ind)}")
    print(f"Validation Out-of-Distribution trajectories: {val_ood_traj_count}, conversations: {len(val_convs_ood)}")
    print(f"Total conversations generated: {len(train_convs) + len(val_convs_ind) + len(val_convs_ood)}")


if __name__ == '__main__':
    main()
