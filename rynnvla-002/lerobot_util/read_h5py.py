import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from PIL import Image


episode_file = '/public/hz_oss/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0623/laptop_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll/episode_000000.hdf5'
CHUNK_SIZE = 500  # 控制每批读取多少帧

with h5py.File(episode_file, 'r') as root:
    front_images_dataset = root['obs/front_image']
    wrist_images_dataset = root['obs/wrist_image']
    state_dataset = root['obs/state']

    total_frames = len(state_dataset)

    front_images_valid = []
    wrist_images_valid = []
    relative_action_valid = []

    for start in range(0, total_frames, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_frames)
        states_chunk = state_dataset[start:end]
        rel_actions = states_chunk[1:] - states_chunk[:-1]
        rel_actions_sum = np.sum(np.abs(rel_actions), axis=1)

        valid_local_indices = np.where(rel_actions_sum != 0)[0]
        valid_global_indices = start + valid_local_indices

        front_images_valid.append(front_images_dataset[valid_global_indices])
        wrist_images_valid.append(wrist_images_dataset[valid_global_indices])
        relative_action_valid.append(rel_actions[valid_local_indices])
        print(start)

    # 合并所有 chunk
    front_images_valid = np.concatenate(front_images_valid, axis=0)
    wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
    relative_action_valid = np.concatenate(relative_action_valid, axis=0)

    for i in range(len(front_images_valid)):
      # import pdb; pdb.set_trace()
      rgb_image = Image.fromarray(front_images_valid[i])
      rgb_filename = os.path.join('/public/hz_oss/cenjun/WorldVLA/worldvla/lerobot_util', f"front_image_{i}.png")
      rgb_image.save(rgb_filename)

      rgb_image = Image.fromarray(wrist_images_valid[i])
      rgb_filename = os.path.join('/public/hz_oss/cenjun/WorldVLA/worldvla/lerobot_util', f"wrist_image_{i}.png")
      rgb_image.save(rgb_filename)

      action = relative_action_valid[i]
      action_filename = os.path.join('/public/hz_oss/cenjun/WorldVLA/worldvla/lerobot_util', f"action_{i}.npy")
      np.save(action_filename, action)
      import pdb; pdb.set_trace()






# orig_data = orig_data_file["data"]
import pdb; pdb.set_trace()