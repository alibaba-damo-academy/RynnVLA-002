# process_data.py

import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set environment variable to prevent file locking issues with HDF5 on network file systems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Define a constant for chunk size to manage memory usage
CHUNK_SIZE = 500  # Controls how many frames are read into memory at once

def process_episode_file(episode_path):
    """
    Processes a single HDF5 episode file.
    - Reads front/wrist images and state data.
    - Calculates relative actions (state[t+1] - state[t]).
    - Filters out timesteps where the action is zero.
    - Returns the valid images and actions.
    """
    # try:
    if 1:
        with h5py.File(episode_path, 'r') as root:
            # Check if all required datasets exist
            if 'obs/front_image' not in root or 'obs/wrist_image' not in root or 'obs/state' not in root:
                print(f"Skipping invalid file {episode_path}: missing required datasets.")
                return None

            front_images_dataset = root['obs/front_image']
            wrist_images_dataset = root['obs/wrist_image']
            state_dataset = root['obs/state']
            abs_action = root['action']
            total_frames = len(state_dataset)
            
            if total_frames < 2:
                print(f"Skipping file {episode_path}: not enough frames to compute actions.")
                return None

            front_images_valid = []
            wrist_images_valid = []
            relative_action_valid = []
            relative_action_valid_2 = []
            abs_actions = []
            abs_state = []

            # Process the data in chunks to avoid high memory consumption
            print(f"  Processing {total_frames} frames in chunks of {CHUNK_SIZE}...")
            for start in range(0, total_frames, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, total_frames)
                
                # Load a chunk of states. We need one extra state to calculate the last relative action.
                state_end = min(end + 1, total_frames)
                if start >= state_end -1:
                    continue # not enough states to calculate action
                states_chunk = state_dataset[start:state_end]

                # Calculate relative actions for the chunk
                rel_actions = states_chunk[1:] - states_chunk[:-1]
                rel_actions_2 = abs_action[start:state_end][:-1] - states_chunk[:-1]

                
                # Find indices where the action is non-zero
                rel_actions_sum = np.sum(np.abs(rel_actions), axis=1)
                valid_local_indices = np.where(rel_actions_sum != 0)[0]
                
                if len(valid_local_indices) == 0:
                    continue
                
                # Convert local chunk indices to global file indices
                valid_global_indices = start + valid_local_indices

                # Append valid data from this chunk
                front_images_valid.append(front_images_dataset)
                wrist_images_valid.append(wrist_images_dataset)
                relative_action_valid.append(rel_actions)
                relative_action_valid_2.append(rel_actions_2)
                abs_actions.append(abs_action[:-1])
                abs_state.append(state_dataset[:-1])

            if not front_images_valid:
                print(f"  No valid actions found in {episode_path}.")
                return None

            # Concatenate results from all chunks
            front_images_valid = np.concatenate(front_images_valid, axis=0)
            wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
            relative_action_valid = np.concatenate(relative_action_valid, axis=0)
            relative_action_valid_2 = np.concatenate(relative_action_valid_2, axis=0)
            abs_actions = np.concatenate(abs_actions, axis=0)
            abs_state = np.concatenate(abs_state, axis=0)
            print(relative_action_valid.shape, relative_action_valid_2.shape, abs_action.shape, state_dataset.shape)
            plot_figs(relative_action_valid, relative_action_valid_2, abs_action, state_dataset)

            import pdb; pdb.set_trace()

            return front_images_valid, wrist_images_valid, relative_action_valid
            
    # except Exception as e:
    #     print(f"Error processing file {episode_path}: {e}")
    #     return None

def plot_figs(relative_action_valid, relative_action_valid_2, abs_actions, abs_state):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    print("模拟数据创建完成。")
    print(f"relative_action_valid shape: {relative_action_valid.shape}")
    print(f"relative_action_valid_2 shape: {relative_action_valid_2.shape}")
    print(f"abs_actions shape: {abs_actions.shape}")
    print(f"abs_state shape: {abs_state.shape}")

    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. 可视化并保存第一组: relative_action_valid vs relative_action_valid_2 ---

    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('对比 relative_action_valid 和 relative_action_valid_2 (6个维度)', fontsize=16)
    axes1 = axes1.flatten()

    for i in range(6):
        ax = axes1[i]
        ax.plot(relative_action_valid[:, i], label='relative_action_valid', color='blue', alpha=0.9)
        ax.plot(relative_action_valid_2[:, i], label='relative_action_valid_2', color='red', linestyle='--', alpha=0.8)
        ax.set_title(f'维度 {i+1} 对比')
        ax.set_xlabel('样本索引 (Sample Index)')
        ax.set_ylabel('值 (Value)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 新增代码：保存第一张图片 ---
    # 在 plt.show() 之前调用 savefig
    output_filename1 = 'relative_actions_comparison.png'
    fig1.savefig(output_filename1, dpi=300, bbox_inches='tight')
    print(f"第一张图片已保存为: {output_filename1}")


    # --- 3. 可视化并保存第二组: abs_actions vs abs_state ---

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('对比 abs_actions 和 abs_state (6个维度)', fontsize=16)
    axes2 = axes2.flatten()

    for i in range(6):
        ax = axes2[i]
        ax.plot(abs_actions[:, i], label='abs_actions', color='green')
        ax.plot(abs_state[:, i], label='abs_state', color='purple', linestyle='-.')
        ax.set_title(f'维度 {i+1} 对比')
        ax.set_xlabel('样本索引 (Sample Index)')
        ax.set_ylabel('值 (Value)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 新增代码：保存第二张图片 ---
    output_filename2 = 'absolute_actions_comparison.png'
    fig2.savefig(output_filename2, dpi=300, bbox_inches='tight')
    print(f"第二张图片已保存为: {output_filename2}")




def save_processed_data(output_dir, front_images, wrist_images, actions):
    """Saves the processed data (images and actions) to the output directory."""
    front_image_dir = os.path.join(output_dir, 'front_image')
    wrist_image_dir = os.path.join(output_dir, 'wrist_image')
    action_dir = os.path.join(output_dir, 'action')

    os.makedirs(front_image_dir, exist_ok=True)
    os.makedirs(wrist_image_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)

    print(f"  Saving {len(front_images)} frames to {output_dir}...")
    for i in tqdm(range(len(front_images)), desc="  Saving frames"):
        # This check allows for resuming a partially completed save operation
        if os.path.exists(os.path.join(front_image_dir, f"image_{i}.png")):
            continue

        # Save front image
        Image.fromarray(front_images[i]).save(os.path.join(front_image_dir, f"image_{i}.png"))

        # Save wrist image
        Image.fromarray(wrist_images[i]).save(os.path.join(wrist_image_dir, f"image_{i}.png"))

        # Save action
        np.save(os.path.join(action_dir, f"action_{i}.npy"), actions[i])


def main(args):
    """Main function to walk through directories and process files."""
    for root_dir, _, files in os.walk(args.input_dir):
        # Sort files for deterministic processing order
        for file in sorted(files):
            if file.endswith('.hdf5'):
                episode_path = os.path.join(root_dir, file)
                
                # Construct the output directory path
                rel_path = os.path.relpath(root_dir, args.input_dir)
                trj_num = os.path.splitext(file)[0] # Get filename without extension
                output_dir = os.path.join(args.output_dir, rel_path, trj_num)
                if os.path.exists(output_dir):
                    continue

                print(f"\nProcessing HDF5 file: {episode_path}")
                
                result = process_episode_file(episode_path)

                if result:
                    front_images, wrist_images, actions = result
                    save_processed_data(output_dir, front_images, wrist_images, actions)
                    print(f"Successfully saved data to: {output_dir}")
                else:
                    print(f"Skipped or failed to process: {episode_path}")

    print("\nAll data processing is complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 robot data into images and actions.")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory of the raw HDF5 data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory to save the processed data.')
    
    args = parser.parse_args()
    main(args)