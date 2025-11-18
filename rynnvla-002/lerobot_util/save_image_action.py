import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# 原始数据根目录
input_base_dir = '/public/hz_oss/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0623'
input_base_dir = '/mnt/PLNAS/cenjun/lerobot_data/data_0623'

# 输出根目录（与 input_base_dir 对应）
output_base_dir = '/public/hz_oss/cenjun/WorldVLA/worldvla/processed_data_lerobot/data_0623'
output_base_dir = '/mnt/nas_jianchong/cenjun.cj/grasping/World-VLA/worldvla/processed_data/data_0623'
output_base_dir = '/mnt/PLNAS/cenjun/processed_lerobot_data/data_0623'

CHUNK_SIZE = 500  # 控制每批读取多少帧

def process_episode_file(episode_path):
    """处理单个 hdf5 文件"""
    with h5py.File(episode_path, 'r') as root:
        if 'obs/front_image' not in root or 'obs/wrist_image' not in root or 'obs/state' not in root:
            print(f"跳过无效文件 {episode_path}：缺少必要字段")
            return

        front_images_dataset = root['obs/front_image']
        wrist_images_dataset = root['obs/wrist_image']
        state_dataset = root['obs/state']

        total_frames = len(state_dataset)

        front_images_valid = []
        wrist_images_valid = []
        relative_action_valid = []

        for start in tqdm(range(0, total_frames, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, total_frames)
            states_chunk = state_dataset[start:end]
            rel_actions = states_chunk[1:] - states_chunk[:-1]
            rel_actions_sum = np.sum(np.abs(rel_actions), axis=1)

            valid_local_indices = np.where(rel_actions_sum != 0)[0]
            valid_global_indices = start + valid_local_indices

            front_images_valid.append(front_images_dataset[valid_global_indices])
            wrist_images_valid.append(wrist_images_dataset[valid_global_indices])
            relative_action_valid.append(rel_actions[valid_local_indices])

        # 合并 chunk
        front_images_valid = np.concatenate(front_images_valid, axis=0)
        wrist_images_valid = np.concatenate(wrist_images_valid, axis=0)
        relative_action_valid = np.concatenate(relative_action_valid, axis=0)

        return front_images_valid, wrist_images_valid, relative_action_valid

    return None


def save_processed_data(output_dir, front_images, wrist_images, actions):
    """将提取的数据保存到对应的子文件夹"""
    front_image_dir = os.path.join(output_dir, 'front_image')
    wrist_image_dir = os.path.join(output_dir, 'wrist_image')
    action_dir = os.path.join(output_dir, 'action')

    os.makedirs(front_image_dir, exist_ok=True)
    os.makedirs(wrist_image_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)

    for i in tqdm(range(len(front_images))):
        if os.path.exists(os.path.join(front_image_dir, f"image_{i}.png")):
            continue

        # 保存 front image
        rgb_image = Image.fromarray(front_images[i])
        rgb_filename = os.path.join(front_image_dir, f"image_{i}.png")
        rgb_image.save(rgb_filename)

        # 保存 wrist image
        rgb_image = Image.fromarray(wrist_images[i])
        rgb_filename = os.path.join(wrist_image_dir, f"image_{i}.png")
        rgb_image.save(rgb_filename)

        # 保存 action
        action = actions[i]
        action_filename = os.path.join(action_dir, f"action_{i}.npy")
        np.save(action_filename, action)


# 主程序：遍历所有 .hdf5 文件
for root_dir, _, files in os.walk(input_base_dir):
    for file in files:
        if file.endswith('.hdf5'):
            episode_path = os.path.join(root_dir, file)
            trj_num = episode_path.split('/')[-1].split('.')[0]

            # 计算相对路径以构造输出目录
            rel_path = os.path.relpath(root_dir, input_base_dir)
            output_dir = os.path.join(output_base_dir, rel_path, trj_num)

            print(f"Processing: {episode_path}")
            # import pdb; pdb.set_trace()
            result = process_episode_file(episode_path)

            if result is not None:
                front_images, wrist_images, actions = result
                save_processed_data(output_dir, front_images, wrist_images, actions)
                print(f"Saved to: {output_dir}")
            else:
                print(f"Failed to process: {episode_path}")

print("全部数据处理完成。")