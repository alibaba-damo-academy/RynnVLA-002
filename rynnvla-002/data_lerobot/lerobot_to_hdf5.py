from typing import Iterator, Tuple, Any, Dict, List

import os
import h5py
import glob
import numpy as np

import argparse
import gc
import logging
import time
from pathlib import Path

import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset    # for original lerobot


# Converts a CHW Tensor to an HWC uint8 NumPy array
def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def save_episode_to_hdf5(
    output_dir: str,
    episode_index: int,
    data: Dict[str, List[np.ndarray]],
    metadata: Dict[str, Any]
):
    """Saves all data for a single episode to an HDF5 file."""
    if not data["actions"]:
        print(f"Skipping empty episode {episode_index}")
        return

    print(f'Saving episode {episode_index}...')
    hdf5_episode_id = f"episode_{episode_index:06d}"
    hdf5_path = os.path.join(output_dir, f"{hdf5_episode_id}.hdf5")

    with h5py.File(hdf5_path, 'w') as f:
        # Save observations
        obs_group = f.create_group("obs")
        obs_group.create_dataset("front_image", data=np.stack(data["front_images"]), dtype='uint8')
        obs_group.create_dataset("wrist_image", data=np.stack(data["wrist_images"]), dtype='uint8')
        obs_group.create_dataset("state", data=np.stack(data["states"]), dtype='float32')

        # Save actions and timestamps
        f.create_dataset("action", data=np.stack(data["actions"]), dtype='float32')
        f.create_dataset("timestamp", data=np.stack(data["timestamps"]), dtype='float32')

        # Save metadata
        f.attrs["task_index"] = metadata["task_index"]
        f.attrs["language_instruction"] = metadata["language_instruction"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot_input_dir", type=str, required=True)
    parser.add_argument("--episode_start_idx", type=int, default=None)
    parser.add_argument("--episode_end_idx", type=int, default=None)    # Note: To process episodes 0-2 (3 total), use episode_end_idx = 2.
    parser.add_argument("--hdf5_output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.hdf5_output_dir, exist_ok=True)

    dataset = LeRobotDataset(None, root=args.lerobot_input_dir, tolerance_s=1e-4)
    first_episode_idx, last_episode_idx = 0, dataset.num_episodes - 1

    start_idx = args.episode_start_idx if args.episode_start_idx is not None else first_episode_idx
    end_idx = args.episode_end_idx if args.episode_end_idx is not None else last_episode_idx

    assert start_idx >= first_episode_idx, f"args.episode_start_idx {start_idx} should be >= {first_episode_idx}"
    assert end_idx <= last_episode_idx, f"args.episode_end_idx {end_idx} should be <= {last_episode_idx}"
    
    from_idx = dataset.episode_data_index["from"][start_idx].item()
    to_idx = dataset.episode_data_index["to"][end_idx].item()

    print(f'Processing data from absolute index {from_idx} to {to_idx}.')
    
    current_episode_index = -1
    episode_data = {
        "front_images": [], "wrist_images": [], "states": [], "actions": [], "timestamps": []
    }
    episode_metadata = {}

    for i_batch in tqdm.tqdm(range(from_idx, to_idx)):
        batch = dataset[i_batch]
        batch_episode_index = batch["episode_index"].item()

        if batch_episode_index != current_episode_index:
            # If we are switching to a new episode, save the data from the previous one.
            if current_episode_index != -1:
                save_episode_to_hdf5(args.hdf5_output_dir, current_episode_index, episode_data, episode_metadata)

            # Reset buffers for the new episode
            episode_data = {key: [] for key in episode_data}
            
            # Update metadata for the new episode
            current_episode_index = batch_episode_index
            episode_metadata["task_index"] = batch["task_index"].item()
            episode_metadata["language_instruction"] = batch['task']

        # Append data for the current step
        episode_data["front_images"].append(to_hwc_uint8_numpy(batch['observation.images.front']))
        episode_data["wrist_images"].append(to_hwc_uint8_numpy(batch['observation.images.wrist']))
        episode_data["states"].append(batch['observation.state'].cpu().numpy().astype(np.float32))
        episode_data["actions"].append(batch['action'].cpu().numpy().astype(np.float32))
        episode_data["timestamps"].append(batch['timestamp'].cpu().numpy().astype(np.float32))

    # --- Added a final save call for the very last episode after the loop finishes ---
    if current_episode_index != -1:
        save_episode_to_hdf5(args.hdf5_output_dir, current_episode_index, episode_data, episode_metadata)
    
    print("Conversion finished.")