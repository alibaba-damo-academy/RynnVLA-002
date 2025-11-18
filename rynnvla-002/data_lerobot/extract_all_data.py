import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    - For each timestep `t`, it extracts the state and calculates a sequence of `CK` relative actions:
      `abs_action[t:t+CK] - state[t]`.
    - Filters out timesteps where the entire future action sequence of length CK is zero.
    - Returns the valid images, the corresponding action sequences, and the corresponding states.
    """
    try:
        with h5py.File(episode_path, 'r') as root:
            required_datasets = ['obs/front_image', 'obs/wrist_image', 'obs/state', 'action']
            if not all(d in root for d in required_datasets):
                print(f"Skipping invalid file {episode_path}: missing required datasets.")
                return None

            front_images_dataset = root['obs/front_image']
            wrist_images_dataset = root['obs/wrist_image']
            
            all_states = root['obs/state'][:]
            all_abs_actions = root['action'][:]
            total_frames = len(all_states)

            if total_frames < CK:
                print(f"Skipping file {episode_path}: not enough frames ({total_frames}) for lookahead CK={CK}.")
                return None

            valid_indices = []
            abs_action_sequences_for_valid_indices = []
            rel_action_sequences_for_valid_indices = []
            states_for_valid_indices = []

            for idx in range(total_frames - CK + 1):
                current_state = all_states[idx]
                action_targets = all_abs_actions[idx : idx + CK]

                rel_actions_sequence = action_targets - current_state[np.newaxis, :]
                rel_actions_sequence_2 = action_targets - current_state[np.newaxis, :]
                rel_actions_sequence_2[:, -1] = action_targets[:, -1]

                if np.sum(np.abs(rel_actions_sequence_2)) != 0:
                    valid_indices.append(idx)
                    abs_action_sequences_for_valid_indices.append(rel_actions_sequence_2)
                    rel_action_sequences_for_valid_indices.append(rel_actions_sequence )
                    states_for_valid_indices.append(current_state)

            if not valid_indices:
                print(f"  No valid (non-static over CK horizon) actions found in {episode_path}.")
                return None

            front_images_valid = []
            wrist_images_valid = []

            for i in range(0, len(valid_indices), CHUNK_SIZE):
                indices_chunk = valid_indices[i:i + CHUNK_SIZE]
                front_images_valid.append(front_images_dataset[indices_chunk])
                wrist_images_valid.append(wrist_images_dataset[indices_chunk])

            if front_images_valid:
                front_images_valid = np.concatenate(front_images_valid, axis=0)
                wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
                states_valid = np.array(states_for_valid_indices)
                return front_images_valid, wrist_images_valid, abs_action_sequences_for_valid_indices, rel_action_sequences_for_valid_indices, states_valid

    except Exception as e:
        print(f"Error processing file {episode_path}: {e}")
        return None


def save_processed_data(output_dir, front_images, wrist_images, abs_action_sequences, rel_action_sequences, states):
    """
    Saves the processed data. Images and states are saved as individual files,
    and action sequences are saved into subdirectories.
    """
    front_image_dir = os.path.join(output_dir, 'front_image')
    wrist_image_dir = os.path.join(output_dir, 'wrist_image')
    abs_action_dir = os.path.join(output_dir, 'abs_action')
    rel_action_dir = os.path.join(output_dir, 'rel_action')
    state_dir = os.path.join(output_dir, 'state')

    os.makedirs(front_image_dir, exist_ok=True)
    os.makedirs(wrist_image_dir, exist_ok=True)
    os.makedirs(abs_action_dir, exist_ok=True)
    os.makedirs(rel_action_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    for i in tqdm(range(len(front_images)), desc="  Saving frames", leave=False):
        abs_action_sequence_dir = os.path.join(abs_action_dir, f"action_{i}")
        rel_action_sequence_dir = os.path.join(rel_action_dir, f"action_{i}")

        if os.path.exists(abs_action_sequence_dir):
            continue

        Image.fromarray(front_images[i]).save(os.path.join(front_image_dir, f"image_{i}.png"))
        Image.fromarray(wrist_images[i]).save(os.path.join(wrist_image_dir, f"image_{i}.png"))
        np.save(os.path.join(state_dir, f"state_{i}.npy"), states[i])

        os.makedirs(abs_action_sequence_dir, exist_ok=True)
        for j in range(len(abs_action_sequences[i])):
            np.save(os.path.join(abs_action_sequence_dir, f"{j}.npy"), abs_action_sequences[i][j])
        
        os.makedirs(rel_action_sequence_dir, exist_ok=True)
        for j in range(len(rel_action_sequences[i])):
            np.save(os.path.join(rel_action_sequence_dir, f"{j}.npy"), rel_action_sequences[i][j])


def generate_output_path(hdf5_path, instruction, base_output_dir):
    """
    Converts an HDF5 path into the output directory by:
    - Replacing `/mnt/PLNAS/cenjun/all_data` with `/mnt/PLNAS/cenjun/all_data/extracted/instruction_path`
    - Where `instruction_path` is the instruction text with spaces replaced by underscores
    """
    instruction_path = instruction.replace(" ", "_").replace("/", "_").replace("\\", "_")
    base_prefix = "/mnt/PLNAS/cenjun/all_data"
    new_base = os.path.join(base_prefix, "extracted", instruction_path)
    
    # Replace old prefix with new one
    rel_path = os.path.relpath(hdf5_path, base_prefix)
    output_dir = os.path.join(base_output_dir, instruction_path, os.path.splitext(rel_path)[0])
    return output_dir


def process_one_file(args_tuple):
    """Wrapper function for multiprocessing."""
    hdf5_path, instruction, base_output_dir = args_tuple
    result = process_episode_file(hdf5_path)
    if result:
        front_images, wrist_images, abs_actions, rel_actions, states = result
        output_dir = generate_output_path(hdf5_path, instruction, base_output_dir)
        os.makedirs(output_dir, exist_ok=True)
        save_processed_data(output_dir, front_images, wrist_images, abs_actions, rel_actions, states)
        return True
    else:
        return False


def main(args):
    """Main function to walk through directories and process files using multiple processes."""
    with open(args.json_path, "r") as f:
        data = json.load(f)

    task_data = data.get("task_data", {})
    futures = []

    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        for task_name, task_info in task_data.items():
            instruction = task_info["instructions"][0]  # Use first instruction per task
            for hdf5_path in task_info["data_path"]:
                if not os.path.exists(hdf5_path):
                    print(f"File does not exist: {hdf5_path}")
                    continue
                futures.append(executor.submit(process_one_file, (hdf5_path, instruction, args.output_dir)))

        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            if future.result():
                success_count += 1

    print(f"\n✅ Successfully processed {success_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 robot data from a JSON config into extracted folders.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON config file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory to save the processed data.')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes to use.')

    args = parser.parse_args()
    main(args)