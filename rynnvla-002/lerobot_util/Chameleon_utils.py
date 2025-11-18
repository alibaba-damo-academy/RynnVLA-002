from data_lerobot.pre_tokenize_action import ItemProcessor
from PIL import Image
import os
from transformers import GenerationConfig
import torch
import time

def get_action_Chameleon_dis_awm_ck(model, cur_img, task_description, item_processor, his_img, his_type, action_steps):
    
    if his_type == "1h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>"
                    },
                ],
                "image": [cur_img],
                "action": [],
                }
    elif his_type == "2h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    tokens = item_processor.process_item(conv, training_mode=False)
    

    generation_config = GenerationConfig(max_new_tokens=action_steps*12,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    if 'img_only' in his_type:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    else:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    dis_action = model.generate_dis_ma(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action

def get_action_Chameleon_dis_awm_ck_wrist(model, cur_img, img1, task_description, item_processor, his_img, his_type, action_steps):
    
    if his_type == "1h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|><|image|>"
                    },
                ],
                "image": [cur_img] + [img1],
                "action": [],
                }
    elif his_type == "2h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    tokens = item_processor.process_item(conv, training_mode=False)

    generation_config = GenerationConfig(max_new_tokens=action_steps*12,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    if 'img_only' in his_type:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    else:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    dis_action = model.generate_dis_ma(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action

def get_action_Chameleon_dis_awm_ck_wrist_action_head(model, cur_img, img1, task_description, item_processor, his_img, his_type, action_steps, state=None):
    
    print("task_description: ", task_description)
    if his_type == "1h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|><|image|>"
                    },
                ],
                "image": [cur_img] + [img1],
                "action": [],
                }
    elif his_type == "1h_1a_img_only_state":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|state|><|image|><|image|>"
                    },
                ],
                "image": [cur_img] + [img1],
                "action": [],
                "state": state,
                }
    elif his_type == "2h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a_img_only":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?" + "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    tokens = item_processor.process_item(conv, training_mode=False)

    tokens.append(10004)

    generation_config = GenerationConfig(max_new_tokens=2,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    if 'img_only' in his_type:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    else:
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    dis_action = model.generate_action_head(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action
