import copy
import json
import logging
import os
from pathlib import Path
import pickle
from time import sleep
import traceback
import warnings
import math
import h5py
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml

from libero.libero import benchmark
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class LiberoFinetuneConversation(Dataset):
    def __init__(self, config_path, resolution, with_state=True, with_wrist=True, with_action=True, with_world_model=True):
        logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:")
        logger.info(self.config)

        benchmark_dict = benchmark.get_benchmark_dict()
        logger.info(benchmark_dict)
        logger.info(self.config["META"]["libero_task_suite"])
        self.task_suite = benchmark_dict[self.config["META"]["libero_task_suite"]]()
        self.num_tasks_in_suite = self.task_suite.n_tasks

        if self.config["META"]["split"]=="all":
            self.split_training_set = False
        else:
            self.split_training_set = True
        self.with_state = with_state
        self.with_wrist = with_wrist
        self.with_action = with_action
        self.with_world_model = with_world_model
        self.get_annotation_data(split=self.config["META"]["split"])


    def get_annotation_data(self, split='train'):
        self.data_list = []
        split_index_ood = math.ceil(self.num_tasks_in_suite * 0.9)
        task_dic = {}
        for task_id in range(self.num_tasks_in_suite):
            task = self.task_suite.get_task(task_id).name
            task_dic[task] = task_id
        sorted_task_name = sorted(task_dic.keys())

        for task_id_new in range(len(sorted_task_name)):
            task_name = sorted_task_name[task_id_new]
            task_id = task_dic[task_name]
            orig_data_path = os.path.join(self.config["META"]["raw_data_dir"], f"{task_name}_demo.hdf5")
            orig_data_file = h5py.File(orig_data_path, "r")
            orig_data = orig_data_file["data"]
        
            trj_list = []
            len_trj_list = 0
            for i in range(50):
                # Get demo data
                if f"demo_{i}" in orig_data:
                    len_trj_list+=1
                    trj_list.append(i)
            split_index_ind = math.ceil(len_trj_list * 0.9)
            for i in range(len(trj_list)):
                trj_id = trj_list[i]
                if self.split_training_set:
                    if task_id_new<split_index_ood:
                        if i<split_index_ind:
                            cur_split = 'train'
                        else:
                            cur_split = 'val'
                    else:
                        cur_split = 'val_ood'
                    if split!=cur_split:
                        continue
                demo_data = orig_data[f"demo_{trj_id}"]
                orig_actions = demo_data["actions"][()]
                for j in range(orig_actions.shape[0]):
                    if self.with_action:
                        action_data = self.get_action_data(j, orig_actions.shape[0], orig_actions, trj=trj_id, task_name=task_name, task_id=task_id)
                        if action_data is not None:
                            self.data_list.append(action_data)
                    if self.with_world_model:
                        world_data = self.get_world_model_data(j, orig_actions.shape[0], orig_actions, trj=trj_id, task_name=task_name, task_id=task_id)
                        if world_data is not None:
                            self.data_list.append(world_data)

    def get_action_data(self, action_idx, action_sum, orig_actions, trj, task_name, task_id):
        len_action = self.config["action_model"]["len_action"]
        his = self.config["action_model"]["his"]
        if action_idx>action_sum-len_action:
            return None
        
        data = {}

        img_history_start_idx = max(0, action_idx - his + 1)
        data['image_idx'] = list(range(action_sum)[img_history_start_idx:action_idx+1])
        data['action_ids'] = list(range(action_idx, action_idx + len_action))

        data['task_name'] = task_name
        data['trj'] = trj
        data['task_type'] = 'action'
        data['task_id'] = task_id
        data['state_id'] = action_idx
        return data

    def get_world_model_data(self, action_idx, action_sum, orig_actions, trj, task_name, task_id):
        his = self.config["world_model"]["his"]
        if action_idx>action_sum-his-1:
            return None
        data = {}

        historical_images_idx = list(range(max(action_idx - his + 1, 0), action_idx + 1))
        future_images_idx = list(range(action_idx + 1, action_idx + 2))
        data['image_idx'] = historical_images_idx+future_images_idx
        data['action_ids'] = historical_images_idx
        data['task_name'] = task_name
        data['trj'] = trj
        data['task_type'] = 'world'
        data['task_id'] = task_id
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        action_ids = self.data_list[idx]['action_ids']
        trj = self.data_list[idx]['trj']
        task_id = self.data_list[idx]['task_id']
        task = self.task_suite.get_task(task_id)
        task_name_readable = task.name.replace('_', ' ')
        orig_data_path = os.path.join(self.config["META"]["raw_data_dir"], f"{task.name}_demo.hdf5")
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        demo_data = orig_data[f"demo_{trj}"]
        orig_rgb = demo_data['obs']['agentview_rgb'][()]
        orig_rgb_wrist = demo_data['obs']['eye_in_hand_rgb'][()]
        orig_actions = demo_data["actions"][()]
        orig_ee_states = demo_data['obs']["ee_states"][()]
        orig_gripper_states = demo_data['obs']["gripper_states"][()]
        action = [copy.deepcopy(orig_actions[idx]) for idx in action_ids]

        images = []
        for image_idx in self.data_list[idx]['image_idx']:
            images.append(Image.fromarray(orig_rgb[image_idx][::-1, ::-1].astype(np.uint8)))
            if self.with_wrist:
                images.append(Image.fromarray(orig_rgb_wrist[image_idx][::-1, ::-1].astype(np.uint8)))

        combined_state = []
        if self.data_list[idx]['task_type']=='action':
            if self.with_state:
                state_id = self.data_list[idx]['state_id']
                ee_state = orig_ee_states[state_id]
                gripper_state = orig_gripper_states[state_id]
                combined_state = np.concatenate([ee_state, gripper_state])

                conversations =[
                    {
                        "from": "human",
                        "value": f"What action should the robot take to {task_name_readable}?" + "<|state|>" + "<|image|>" * len(images)
                    },
                    {
                        "from": "gpt",
                        "value": "<|action|>" * len(action)
                    },
                ]
            else:    
              conversations =[
                  {
                      "from": "human",
                      "value": f"What action should the robot take to {task_name_readable}?" + "<|image|>" * len(images)
                  },
                  {
                      "from": "gpt",
                      "value": "<|action|>" * len(action)
                  },
              ]
                  
        elif self.data_list[idx]['task_type']=='world':
            if self.with_wrist:
                conversations = [
                    {
                        "from": "human",
                        # Revised prompt to accurately reflect variable 'his' length
                        "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|image|><|action|>" * len(action)
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|><|image|>" # The model generates a single image
                    },
                ]
            else:
                conversations = [
                    {
                        "from": "human",
                        # Revised prompt to accurately reflect variable 'his' length
                        "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>" * len(action)
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" # The model generates a single image
                    },
                ]
        # print(conversations)
        # print('***********')
        # tokens, labels = self.item_processor.process_item(conv, training_mode=True)
        return conversations, images, action, combined_state

if __name__=='__main__':
    data = LiberoFinetuneConversation('/mnt/damorobot/yuanyq/code/WorldVLA-main/worldvla/configs/libero_256_all/debug.yaml', 256, True)
    print(data.__len__())
    import pdb 
    pdb.set_trace()
