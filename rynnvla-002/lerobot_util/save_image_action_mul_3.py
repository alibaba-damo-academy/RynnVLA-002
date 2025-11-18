# process_data.py

import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set environment variable to prevent file locking issues with HDF5 on network file systems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Define constants
CHUNK_SIZE = 500  # Controls how many image frames are read into memory at once
CK = 20           # Command Chunk size: Number of future actions to predict for each state

def process_episode_file(episode_path):
    """
    Processes a single HDF5 episode file.
    - Reads front/wrist images, state data, and absolute actions.
    - For each timestep `t`, calculates a sequence of `CK` relative actions:
      `abs_action[t:t+CK] - state[t]`.
    - Filters out timesteps where the entire future action sequence of length CK is zero.
    - Returns the valid images and the corresponding action sequences.
    """
    try:
        with h5py.File(episode_path, 'r') as root:
            # Check if all required datasets exist
            required_datasets = ['obs/front_image', 'obs/wrist_image', 'obs/state', 'action']
            if not all(d in root for d in required_datasets):
                print(f"Skipping invalid file {episode_path}: missing required datasets.")
                return None

            front_images_dataset = root['obs/front_image']
            wrist_images_dataset = root['obs/wrist_image']
            
            # Load state and action datasets into memory for faster processing.
            # These are typically small and benefit from being in-memory.
            all_states = root['obs/state'][:]
            all_abs_actions = root['action'][:]
            total_frames = len(all_states)

            # Need at least CK frames to look ahead
            if total_frames < CK:
                print(f"Skipping file {episode_path}: not enough frames ({total_frames}) for lookahead CK={CK}.")
                return None

            # --- Pass 1: Find valid indices and compute their action sequences ---
            # A timestep is valid if its corresponding CK-step future action sequence is not all zeros.
            valid_indices = []
            action_sequences_for_valid_indices = []
            print(f"  Scanning {total_frames} frames and computing action sequences (lookahead CK={CK})...")
            
            # Loop must stop early enough to allow for a full action chunk of size CK
            for idx in tqdm(range(total_frames - CK + 1), desc="  Scanning and computing"):
                current_state = all_states[idx]
                # Get the sequence of CK future absolute actions
                action_targets = all_abs_actions[idx : idx + CK]
                
                # Calculate the sequence of relative actions
                rel_actions_sequence = action_targets - current_state[np.newaxis, :]
                rel_actions_sequence_2 = action_targets - current_state[np.newaxis, :]
                rel_actions_sequence_2[:, -1] = action_targets[:, -1]
                
                # Keep the index if the robot is commanded to move at any point in the next CK steps.
                # This filters out timesteps where the robot is idle for the entire CK horizon.
                if np.sum(np.abs(rel_actions_sequence)) != 0:
                    valid_indices.append(idx)
                    action_sequences_for_valid_indices.append(rel_actions_sequence_2)
            
            if not valid_indices:
                print(f"  No valid (non-static over CK horizon) actions found in {episode_path}.")
                return None
            
            print(f"  Found {len(valid_indices)} valid timesteps. Now loading corresponding images in chunks.")

            # --- Pass 2: Load ONLY the images for the valid indices in chunks ---
            front_images_valid = []
            wrist_images_valid = []
            
            for i in tqdm(range(0, len(valid_indices), CHUNK_SIZE), desc="  Loading image chunks"):
                indices_chunk = valid_indices[i : i + CHUNK_SIZE]
                # Advanced indexing with a list of indices loads the data into memory.
                front_images_valid.append(front_images_dataset[indices_chunk])
                wrist_images_valid.append(wrist_images_dataset[indices_chunk])

            # Concatenate image chunks if necessary
            if front_images_valid:
                front_images_valid = np.concatenate(front_images_valid, axis=0)
                wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
            else: # Should not happen if valid_indices is not empty, but as a safeguard.
                return None

            return front_images_valid, wrist_images_valid, action_sequences_for_valid_indices

    except Exception as e:
        print(f"Error processing file {episode_path}: {e}")
        return None


def save_processed_data(output_dir, front_images, wrist_images, action_sequences):
    """
    Saves the processed data. Images are saved directly, and action sequences
    are saved into subdirectories.
    """
    front_image_dir = os.path.join(output_dir, 'front_image')
    wrist_image_dir = os.path.join(output_dir, 'wrist_image')
    action_dir = os.path.join(output_dir, 'action')

    os.makedirs(front_image_dir, exist_ok=True)
    os.makedirs(wrist_image_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)

    print(f"  Saving {len(front_images)} frames to {output_dir}...")
    for i in tqdm(range(len(front_images)), desc="  Saving frames"):
        # Define the action subdirectory for the current frame
        action_sequence_dir = os.path.join(action_dir, f"action_{i}")

        # This check allows for resuming a partially completed save operation.
        # It checks if the action subdirectory for this frame already exists.
        if os.path.exists(action_sequence_dir):
            continue

        # Save front image
        Image.fromarray(front_images[i]).save(os.path.join(front_image_dir, f"image_{i}.png"))

        # Save wrist image
        Image.fromarray(wrist_images[i]).save(os.path.join(wrist_image_dir, f"image_{i}.png"))

        # Create the subdirectory for the action sequence
        os.makedirs(action_sequence_dir, exist_ok=True)
        
        # Get the action sequence for the current frame
        action_seq = action_sequences[i]
        
        # Save each action step in the sequence as a separate .npy file
        for j in range(len(action_seq)):
            np.save(os.path.join(action_sequence_dir, f"{j}.npy"), action_seq[j])


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
                    # Check if the output directory is not empty to be sure
                    if os.listdir(output_dir):
                        print(f"Skipping already processed directory: {output_dir}")
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
    parser = argparse.ArgumentParser(description="Process HDF5 robot data into images and action sequences.")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory of the raw HDF5 data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory to save the processed data.')
    
    args = parser.parse_args()
    main(args)

