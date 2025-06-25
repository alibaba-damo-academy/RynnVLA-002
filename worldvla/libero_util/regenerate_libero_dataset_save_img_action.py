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


from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)


IMAGE_RESOLUTION = 256

def create_directory(path):
    if os.path.exists(path):
        print(f"Warning: Directory '{path}' already exists. Deleting and recreating it.")
        shutil.rmtree(path)  # Recursively delete the directory and its contents
    
    os.makedirs(path) # Create the new (or re-created) directory
    print(f"Directory '{path}' created successfully.")


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
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
        env, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)

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
            orig_rgb = demo_data['obs']['agentview_rgb'][()]
            
            cur_trial_dir = os.path.join(cur_task_dir, f"trj_{i}")
            action_dir = os.path.join(cur_trial_dir, 'action')
            img_dir = os.path.join(cur_trial_dir, 'imgs')
            create_directory(action_dir)
            create_directory(img_dir)

            for j in range(orig_actions.shape[0]):
                action = orig_actions[j]
                action_filename = os.path.join(action_dir, f"action_{j}.npy")
                np.save(action_filename, action)

                rgb = orig_rgb[j][::-1, ::-1]
                # 确保 RGB 数组的数据类型为 uint8
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                rgb_image = Image.fromarray(rgb)
                rgb_filename = os.path.join(img_dir, f"image_{j}.png")
                rgb_image.save(rgb_filename)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--save_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    args = parser.parse_args()

    # Start data regeneration
    main(args)
