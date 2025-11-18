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
    try:
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

                
                # Find indices where the action is non-zero
                rel_actions_sum = np.sum(np.abs(rel_actions), axis=1)
                valid_local_indices = np.where(rel_actions_sum != 0)[0]
                
                if len(valid_local_indices) == 0:
                    continue
                
                # Convert local chunk indices to global file indices
                valid_global_indices = start + valid_local_indices

                # Append valid data from this chunk
                front_images_valid.append(front_images_dataset[valid_global_indices])
                wrist_images_valid.append(wrist_images_dataset[valid_global_indices])
                relative_action_valid.append(rel_actions[valid_local_indices])

            if not front_images_valid:
                print(f"  No valid actions found in {episode_path}.")
                return None

            # Concatenate results from all chunks
            front_images_valid = np.concatenate(front_images_valid, axis=0)
            wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
            relative_action_valid = np.concatenate(relative_action_valid, axis=0)

            return front_images_valid, wrist_images_valid, relative_action_valid
            
    except Exception as e:
        print(f"Error processing file {episode_path}: {e}")
        return None


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
