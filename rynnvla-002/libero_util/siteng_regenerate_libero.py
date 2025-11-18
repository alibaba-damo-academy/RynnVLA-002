import argparse
import json
import os
import time

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from PIL import Image
import shutil


# from libero_utils import (
#     get_libero_dummy_action,
#     get_libero_env,
# )


# args.image_resolution = 256

def create_directory(path):
    if os.path.exists(path):
        print(f"Warning: Directory '{path}' already exists. Deleting and recreating it.")
        shutil.rmtree(path)  # Recursively delete the directory and its contents
    
    os.makedirs(path) # Create the new (or re-created) directory
    print(f"Directory '{path}' created successfully.")


# def is_noop(action, prev_action=None, threshold=1e-4):
#     """
#     Returns whether an action is a no-op action.

#     A no-op action satisfies two criteria:
#         (1) All action dimensions, except for the last one (gripper action), are near zero.
#         (2) The gripper action is equal to the previous timestep's gripper action.

#     Explanation of (2):
#         Naively filtering out actions with just criterion (1) is not good because you will
#         remove actions where the robot is staying still but opening/closing its gripper.
#         So you also need to consider the current state (by checking the previous timestep's
#         gripper action as a proxy) to determine whether the action really is a no-op.
#     """
#     # Special case: Previous action is None if this is the first action in the episode
#     # Then we only care about criterion (1)
#     if prev_action is None:
#         return np.linalg.norm(action[:-1]) < threshold

#     # Normal case: Check both criteria (1) and (2)
#     gripper_action = action[-1]
#     prev_gripper_action = prev_action[-1]
#     return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def save_data(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0
    
    save_dir = args.save_dir
    
    create_directory(save_dir)

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        # env, task_description = get_libero_env(task, resolution=args.image_resolution)

        # Get dataset for task
        orig_data_path = os.path.join(args.raw_data_dir, f"{task.name}_demo.hdf5")
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        
        cur_task_dir = os.path.join(save_dir, task.name)
        create_directory(cur_task_dir)
                
        for i in range(50):
            # Get demo data
            try:
                demo_data = orig_data[f"demo_{i}"]
            except:
                continue
            orig_actions = demo_data["actions"][()]
            orig_ee_states = demo_data['obs']["ee_states"][()]
            orig_gripper_states = demo_data['obs']["gripper_states"][()]
            orig_robot_states = demo_data["robot_states"][()]
            orig_rgb = demo_data['obs']['agentview_rgb'][()]
            orig_rgb_wrist = demo_data['obs']['eye_in_hand_rgb'][()]
            
            cur_trial_dir = os.path.join(cur_task_dir, f"trj_{i}")
            action_dir = os.path.join(cur_trial_dir, 'action')
            ee_state_dir = os.path.join(cur_trial_dir, 'ee_state')
            gripper_state_dir = os.path.join(cur_trial_dir, 'gripper_state')
            eef_gripper_state_dir = os.path.join(cur_trial_dir, 'eef_gripper_state')
            robot_state_dir = os.path.join(cur_trial_dir, 'robot_state')
            img_dir_third_view = os.path.join(cur_trial_dir, 'imgs_third_view')
            img_dir_wrist = os.path.join(cur_trial_dir, 'imgs_wrist')
            create_directory(action_dir)
            create_directory(ee_state_dir)
            create_directory(gripper_state_dir)
            create_directory(eef_gripper_state_dir)
            create_directory(robot_state_dir)
            create_directory(img_dir_third_view)
            create_directory(img_dir_wrist)

            for j in range(orig_actions.shape[0]):
                action = orig_actions[j]
                action_filename = os.path.join(action_dir, f"action_{j}.npy")
                np.save(action_filename, action)

                ee_state = orig_ee_states[j]
                gripper_state = orig_gripper_states[j]
                
                # Save ee_state separately
                ee_state_filename = os.path.join(ee_state_dir, f"ee_state_{j}.npy")
                np.save(ee_state_filename, ee_state)
                
                # Save gripper_state separately
                gripper_state_filename = os.path.join(gripper_state_dir, f"gripper_state_{j}.npy")
                np.save(gripper_state_filename, gripper_state)
                
                # Save concatenated eef_gripper_state
                combined_state = np.concatenate([ee_state, gripper_state])
                eef_gripper_state_filename = os.path.join(eef_gripper_state_dir, f"eef_gripper_state_{j}.npy")
                np.save(eef_gripper_state_filename, combined_state)

                robot_state = orig_robot_states[j]
                robot_state_filename = os.path.join(robot_state_dir, f"robot_state_{j}.npy")
                np.save(robot_state_filename, robot_state)

                rgb = orig_rgb[j][::-1, ::-1]
                # 确保 RGB 数组的数据类型为 uint8
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                rgb_image = Image.fromarray(rgb)
                rgb_filename = os.path.join(img_dir_third_view, f"image_{j}.png")
                rgb_image.save(rgb_filename)

                rgb = orig_rgb_wrist[j][::-1, ::-1]
                # 确保 RGB 数组的数据类型为 uint8
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                rgb_image = Image.fromarray(rgb)
                rgb_filename = os.path.join(img_dir_wrist, f"image_{j}.png")
                rgb_image.save(rgb_filename)


def cal_libero_all_setting_stats(args):
    # Define all task suites to process
    task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    
    print(f"Computing min/max values for all task suites: {task_suites}")

    # Get benchmark dictionary
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Initialize min and max arrays with appropriate dimensions
    # Assuming combined_state has a fixed dimension across all data
    first_suite = benchmark_dict[task_suites[0]]()
    first_task = first_suite.get_task(0)
    # first_data_path = os.path.join(args.raw_data_dir, f"{first_task.name}_demo.hdf5")
    first_data_path = os.path.join(f'/mnt/PLNAS/cenjun/libero/processed_data/{task_suites[0]}_no_noops_t_256', f"{first_task.name}_demo.hdf5")
    first_data_file = h5py.File(first_data_path, "r")
    first_demo_data = first_data_file["data"]["demo_0"]
    first_ee_state = first_demo_data['obs']["ee_states"][()][0]
    first_gripper_state = first_demo_data['obs']["gripper_states"][()][0]
    first_combined_state = np.concatenate([first_ee_state, first_gripper_state])
    state_dim = first_combined_state.shape[0]
    
    # Initialize global min and max arrays
    global_min = np.full(state_dim, np.inf)
    global_max = np.full(state_dim, -np.inf)

    # Initialize min and max arrays for actions
    first_action = first_demo_data["actions"][()][0]
    action_dim = first_action.shape[0]
    action_min = np.full(action_dim, np.inf)
    action_max = np.full(action_dim, -np.inf)

    # Process each task suite
    for suite_name in task_suites:
        print(f"Processing {suite_name}...")
        task_suite = benchmark_dict[suite_name]()
        num_tasks_in_suite = task_suite.n_tasks

        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task in suite
            task = task_suite.get_task(task_id)

            # Get dataset for task
            # orig_data_path = os.path.join(args.raw_data_dir, f"{task.name}_demo.hdf5")
            orig_data_path = os.path.join(f'/mnt/PLNAS/cenjun/libero/processed_data/{suite_name}_no_noops_t_256', f"{task.name}_demo.hdf5")
            orig_data_file = h5py.File(orig_data_path, "r")
            orig_data = orig_data_file["data"]
            
            for i in range(50):
                # Get demo data
                try:
                    demo_data = orig_data[f"demo_{i}"]
                except:
                    continue
                orig_actions = demo_data["actions"][()]
                orig_ee_states = demo_data['obs']["ee_states"][()]
                orig_gripper_states = demo_data['obs']["gripper_states"][()]

                # Process actions
                for j in range(orig_actions.shape[0]):
                    action = orig_actions[j]
                    # Update action min and max values
                    action_min = np.minimum(action_min, action)
                    action_max = np.maximum(action_max, action)

                # Process combined states
                for j in range(orig_ee_states.shape[0]):
                    ee_state = orig_ee_states[j]
                    gripper_state = orig_gripper_states[j]
                    # Concatenate ee_state and gripper_state
                    combined_state = np.concatenate([ee_state, gripper_state])
                    
                    # Update global min and max values
                    global_min = np.minimum(global_min, combined_state)
                    global_max = np.maximum(global_max, combined_state)
    
    # Print min and max values
    print("\nGlobal min values:", global_min)
    print("Global max values:", global_max)
    print("\nAction min values:", action_min)
    print("Action max values:", action_max)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--image_resolution", type=int, choices=[256, 512],
                        help="Image resolution", required=True)
    parser.add_argument("--raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--save_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    args = parser.parse_args()

    cal_libero_all_setting_stats(args)