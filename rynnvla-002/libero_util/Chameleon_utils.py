from data.pre_tokenize_action import ItemProcessor
from PIL import Image
import os
from transformers import GenerationConfig
import torch
import time

def get_action_Chameleon(model, observation, task_description, item_processor):
    img_input = observation['full_image']
    rgb_image = Image.fromarray(img_input)
    # print(rgb_image.size)
    # new_size = (320, 224)
    # rgb_image = rgb_image.resize(new_size)
    # rgb_filename = os.path.join("image_test.png")
    # rgb_image.save(rgb_filename)
    
    conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "What action should the robot take to " + task_description + " <|image|>"
                },
                # {
                #     "from": "gpt",
                #     "value": ""
                # },
            ],
            "image": [rgb_image],
            "action": [],
            }

    tokens = item_processor.process_item(conv, training_mode=False)
    
    # import pdb; pdb.set_trace()

    # conv = {
    #         "conversations":[
    #             {
    #                 "from": "human",
    #                 "value": "What action should the robot take to " + task_description + " <|image|>"
    #             },
    #             # {
    #             #     "from": "gpt",
    #             #     "value": ""
    #             # },
    #         ],
    #         "image": ["image_test.png"],
    #         "action": [],
    #         }
    # tokens = item_processor.process_item(conv, training_mode=False)
        
    generation_config = GenerationConfig(max_new_tokens=8192,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    ct_action, dis_action = model.generate_ct(input_ids, generation_config)
    
    return ct_action, dis_action


def get_action_Chameleon_dis(model, cur_img, task_description, item_processor, his_img, his_action, his_type, action_steps):
    # img_input = observation['full_image']
    # rgb_image = Image.fromarray(img_input)
    # print(rgb_image.size)
    # new_size = (320, 224)
    # rgb_image = rgb_image.resize(new_size)
    # rgb_filename = os.path.join("image_test.png")
    # rgb_image.save(rgb_filename)
    
    if his_type == "1h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " <|image|>"
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": ""
                    # },
                ],
                "image": [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " <|image|>" + " <|image|>" * len(his_img[-3:])
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": ""
                    # },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    elif his_type == "4ha_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " "  + "<|image|>" + "<|action|>" * len(his_action[-3:])
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": ""
                    # },
                ],
                "image": [cur_img],
                "action": his_action[-3:],
                }
    elif his_type == "4h_4a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " " + "<|image|><|action|>" * len(his_action[-3:]) + "<|image|>"
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": ""
                    # },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": his_action[-3:],
                }
    elif his_type == "4h_4a_awm":
        if len(his_action) == 0:
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " " + "<|image|>"
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": "<|action|>" + "<|image|><|action|>" * len(his_action[-3:])
                    # },
                ],
                "image": [cur_img],
                "action": [],
                }
        else:
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + " " + "<|image|>"
                    },
                    {
                        "from": "gpt",
                        "value": "<|action|><|image|>" * len(his_action[-3:])
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": his_action[-3:],
                }
    tokens = item_processor.process_item(conv, training_mode=False)
    
    # import pdb; pdb.set_trace()

    # conv = {
    #         "conversations":[
    #             {
    #                 "from": "human",
    #                 "value": "What action should the robot take to " + task_description + " <|image|>"
    #             },
    #             # {
    #             #     "from": "gpt",
    #             #     "value": ""
    #             # },
    #         ],
    #         "image": ["image_test.png"],
    #         "action": [],
    #         }
    # tokens = item_processor.process_item(conv, training_mode=False)
        
    # generation_config = GenerationConfig(max_new_tokens=8192,
    #                                     max_length=model.config.max_position_embeddings,
    #                                     temperature=1,
    #                                     top_k=None,
    #                                     do_sample=False,
    #                                     eos_token_id=[8710],
    #                                 )
    generation_config = GenerationConfig(max_new_tokens=action_steps*12,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    dis_action = model.generate_dis(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action
    

def get_action_Chameleon_dis_awm(model, cur_img, task_description, item_processor, his_img, his_action, his_type, action_steps):
    # img_input = observation['full_image']
    # rgb_image = Image.fromarray(img_input)
    # print(rgb_image.size)
    # new_size = (320, 224)
    # rgb_image = rgb_image.resize(new_size)
    # rgb_filename = os.path.join("image_test.png")
    # rgb_image.save(rgb_filename)
    
    if his_type == "1h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    },
                ],
                "image": [cur_img],
                "action": [],
                }
    elif his_type == "2h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    elif his_type == "2h_2a":
        if len(his_action):
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                # "action": his_action[-1:],
                "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
        else:
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": his_action[-1:],
                # "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
    elif his_type == "4h_4a":
        conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "What action should the robot take to " + task_description + "?"
                },
                {
                    "from": "gpt",
                    "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-3:])
                },
            ],
            "image": his_img[-3:] + [cur_img],
            "action": his_action[-3:],
            }
    elif his_type == "1h_1a_img_only":
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
    
    # import pdb; pdb.set_trace()

    # conv = {
    #         "conversations":[
    #             {
    #                 "from": "human",
    #                 "value": "What action should the robot take to " + task_description + " <|image|>"
    #             },
    #             # {
    #             #     "from": "gpt",
    #             #     "value": ""
    #             # },
    #         ],
    #         "image": ["image_test.png"],
    #         "action": [],
    #         }
    # tokens = item_processor.process_item(conv, training_mode=False)
        
    # generation_config = GenerationConfig(max_new_tokens=8192,
    #                                     max_length=model.config.max_position_embeddings,
    #                                     temperature=1,
    #                                     top_k=None,
    #                                     do_sample=False,
    #                                     eos_token_id=[8710],
    #                                 )
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
    dis_action = model.generate_dis(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action

def get_action_Chameleon_dis_awm_ck(model, cur_img, task_description, item_processor, his_img, his_action, his_type, action_steps):
    # img_input = observation['full_image']
    # rgb_image = Image.fromarray(img_input)
    # print(rgb_image.size)
    # new_size = (320, 224)
    # rgb_image = rgb_image.resize(new_size)
    # rgb_filename = os.path.join("image_test.png")
    # rgb_image.save(rgb_filename)
    
    if his_type == "1h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    },
                ],
                "image": [cur_img],
                "action": [],
                }
    elif his_type == "2h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    elif his_type == "2h_2a":
        if len(his_action):
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                # "action": his_action[-1:],
                "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
        else:
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": his_action[-1:],
                # "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
    elif his_type == "4h_4a":
        conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "What action should the robot take to " + task_description + "?"
                },
                {
                    "from": "gpt",
                    "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-3:])
                },
            ],
            "image": his_img[-3:] + [cur_img],
            "action": his_action[-3:],
            }
    elif his_type == "1h_1a_img_only":
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
    
    # import pdb; pdb.set_trace()

    # conv = {
    #         "conversations":[
    #             {
    #                 "from": "human",
    #                 "value": "What action should the robot take to " + task_description + " <|image|>"
    #             },
    #             # {
    #             #     "from": "gpt",
    #             #     "value": ""
    #             # },
    #         ],
    #         "image": ["image_test.png"],
    #         "action": [],
    #         }
    # tokens = item_processor.process_item(conv, training_mode=False)
        
    # generation_config = GenerationConfig(max_new_tokens=8192,
    #                                     max_length=model.config.max_position_embeddings,
    #                                     temperature=1,
    #                                     top_k=None,
    #                                     do_sample=False,
    #                                     eos_token_id=[8710],
    #                                 )
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

def get_action_Chameleon_dis_awm_ck_action_head(model, cur_img, task_description, item_processor, his_img, his_action, his_type, action_steps):
    
    if his_type == "1h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>"
                    },
                ],
                "image": [cur_img],
                "action": [],
                }
    elif his_type == "2h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-1:]) + "<|image|>"
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": [],
                }
    elif his_type == "4h_1a":
        conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" * len(his_img[-3:]) + "<|image|>"
                    },
                ],
                "image": his_img[-3:] + [cur_img],
                "action": [],
                }
    elif his_type == "2h_2a":
        if len(his_action):
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                # "action": his_action[-1:],
                "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
        else:
            conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "What action should the robot take to " + task_description + "?"
                    },
                    {
                        "from": "gpt",
                        "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-1:])
                    },
                ],
                "image": his_img[-1:] + [cur_img],
                "action": his_action[-1:],
                # "action": [[0.5, 0, 0, 0, 0, 0, -1]],
                }
    elif his_type == "4h_4a":
        conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "What action should the robot take to " + task_description + "?"
                },
                {
                    "from": "gpt",
                    "value": "<|image|>" + "<|action|><|image|>" * len(his_action[-3:])
                },
            ],
            "image": his_img[-3:] + [cur_img],
            "action": his_action[-3:],
            }
    elif his_type == "1h_1a_img_only":
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
    
    generation_config = GenerationConfig(max_new_tokens=1,
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

def get_action_Chameleon_dis_awm_g_video(model, task_description, item_processor, his_img, his_action, his_type):
    
    conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "Generate the image based on the current image and the action." + "<|image|><|action|>"
                },
                # {
                #     "from": "gpt",
                #     "value": "<|image|>"
                # },
            ],
            "image": his_img[-1:],
            "action": his_action[-1:],
        }
    tokens = item_processor.process_item(conv, training_mode=False)
    
    generation_config = GenerationConfig(max_new_tokens=3000,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    # generation_config = GenerationConfig(max_new_tokens=400,
    #                                 max_length=model.config.max_position_embeddings,
    #                                 temperature=1,
    #                                 top_k=50,
    #                                 do_sample=True,
    #                                 eos_token_id=[8710],
    #                             )
    # input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    g_image_tokens = model.generate_img(input_ids, generation_config)
    # import pdb; pdb.set_trace()
    # g_image = item_processor.decode_image(g_image_tokens[0, :-10].cpu().tolist())
    g_image = item_processor.decode_image(g_image_tokens[0, :-1].cpu().tolist())
        
    return g_image

def get_action_Chameleon_dis_awm_g_video_wrist(model, task_description, item_processor, his_img, his_img_wrist, his_action, his_type):
    
    conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "Generate the image based on the current image and the action." + "<|image|><|image|><|action|>"
                },
                # {
                #     "from": "gpt",
                #     "value": "<|image|>"
                # },
            ],
            "image": his_img[-1:] + his_img_wrist[-1:],
            "action": his_action[-1:],
        }
    tokens = item_processor.process_item(conv, training_mode=False)
    
    generation_config = GenerationConfig(max_new_tokens=3000,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    # generation_config = GenerationConfig(max_new_tokens=400,
    #                                 max_length=model.config.max_position_embeddings,
    #                                 temperature=1,
    #                                 top_k=50,
    #                                 do_sample=True,
    #                                 eos_token_id=[8710],
    #                             )
    # input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    g_image_tokens = model.generate_img(input_ids, generation_config)

    # import pdb; pdb.set_trace()
    # g_image = item_processor.decode_image(g_image_tokens[0, :-10].cpu().tolist())
    # torch.set_printoptions(profile="full")
    # print('g_image_tokens', g_image_tokens, g_image_tokens.shape)
    # g_image_front = item_processor.decode_image(g_image_tokens[0, :-1].cpu().tolist())
    # g_image_wrist = item_processor.decode_image(g_image_tokens[0, :-1].cpu().tolist())

    # 定义图像的开始和结束 token ID
    IMG_START_TOKEN = 8197
    IMG_END_TOKEN = 8196

    # 获取批次中第一个样本的 token 序列 (形状变为 [sequence_length])
    tokens_sequence = g_image_tokens[0]

    # 找到所有开始和结束 token 的位置（索引）
    start_indices = torch.where(tokens_sequence == IMG_START_TOKEN)[0]
    end_indices = torch.where(tokens_sequence == IMG_END_TOKEN)[0]

    # 检查是否找到了足够的 token 对
    if len(start_indices) >= 2 and len(end_indices) >= 2:
        # --- 提取第一张图像的 tokens ---
        # 从第一个 start_token 到第一个 end_token (包含 end_token)
        front_start_idx = start_indices[0]
        front_end_idx = end_indices[0]
        # 切片时，结束索引需要+1，因为Python切片是“左闭右开”的
        g_image_front_tokens = tokens_sequence[front_start_idx : front_end_idx + 1]

        # --- 提取第二张图像的 tokens ---
        # 从第二个 start_token 到第二个 end_token (包含 end_token)
        wrist_start_idx = start_indices[1]
        wrist_end_idx = end_indices[1]
        g_image_wrist_tokens = tokens_sequence[wrist_start_idx : wrist_end_idx + 1]

        print("成功提取到两组图像 tokens:")
        print("Front image tokens:", g_image_front_tokens)
        print("Wrist image tokens:", g_image_wrist_tokens)

        # --- 使用提取出的 tokens 解码图像 ---
        # 注意：这里我们使用了提取出的新变量 g_image_front_tokens 和 g_image_wrist_tokens
        # .cpu() 是为了确保张量在CPU上，.tolist() 将其转换为Python列表
        g_image_front = item_processor.decode_image(g_image_front_tokens.cpu().tolist())
        g_image_wrist = item_processor.decode_image(g_image_wrist_tokens.cpu().tolist())

        # 这里可以接续保存或显示图像的代码
        # g_image_front.save("generated_front_image.png")
        # g_image_wrist.save("generated_wrist_image.png")

    else:
        print(f"错误：未能找到两对图像开始/结束标记。")
        print(f"找到了 {len(start_indices)} 个开始标记 ({IMG_START_TOKEN})。")
        print(f"找到了 {len(end_indices)} 个结束标记 ({IMG_END_TOKEN})。")
        g_image_front = None
        g_image_wrist = None
        
    return g_image_front, g_image_wrist

def reconstruct_img(item_processor, img):
    conv = {
                "conversations":[
                    {
                        "from": "human",
                        "value": "<|image|>"
                    },
                    # {
                    #     "from": "gpt",
                    #     "value": "<|image|>"
                    # },
                ],
                "image": img,
                "action": [],
            }
    tokens = item_processor.process_item(conv, training_mode=False)
    g_image = item_processor.decode_image(tokens[1:-1])

    return g_image


def parse_his_type(his_type: str):
    try:
        # 1. 按下划线分割字符串
        parts = his_type.split('_')

        # 2. 进行基本格式校验
        # 至少需要 'his', number, view, state_prefix, 'state' 这5个部分
        if len(parts) < 5 or parts[0] != 'his' or parts[-1] != 'state':
            print(f"警告: 字符串 '{his_type}' 格式不符合预期。")
            return None

        # 3. 提取 'his' (历史步数)
        # parts[1] 应该是代表历史步数的数字
        his = int(parts[1])

        # 4. 提取 'with_state' (是否包含状态)
        # parts[-2] 应该是 'w' 或 'wo'
        state_prefix = parts[-2]
        if state_prefix == 'w':
            with_state = True
        elif state_prefix == 'wo':
            with_state = False
        else:
            # 如果不是 'w' 或 'wo'，则格式错误
            print(f"警告: 在 '{his_type}' 中找到无效的状态前缀 '{state_prefix}'。")
            return None
        
        # 更简洁的写法:
        # with_state = (state_prefix == 'w')

        # 5. 提取 'views' (视图列表)
        # 'views' 是位于 'his' 部分之后、'state' 部分之前的所有内容
        # 即从第3个元素 (索引2) 到倒数第2个元素 (切片不包含末尾)
        views = parts[2:-2]

        return {
            'his': his,
            'views': views,
            'with_state': with_state
        }

    except (ValueError, IndexError) as e:
        # 捕获可能的错误，例如 int() 转换失败或列表索引越界
        print(f"解析字符串 '{his_type}' 时出错: {e}")
        return None

def get_action_Chameleon_dis_awm_ck_discrete_action(model, cur_img, cur_wrist_img, task_description, item_processor, his_img, his_wrist_img, cur_state, his_action, his_type, action_steps):
    
    his_type_dict = parse_his_type(his_type)

    if his_type_dict['his'] == 2 and len(his_type_dict['views']) == 3:
        img_c = his_img[-1:] + his_wrist_img[-1:] + [cur_img] + [cur_wrist_img]
    elif his_type_dict['his'] == 1 and len(his_type_dict['views']) == 3:
        img_c = [cur_img] + [cur_wrist_img]
    elif his_type_dict['his'] == 2 and len(his_type_dict['views']) == 2:
        img_c = his_img[-1:] + [cur_img]
    elif his_type_dict['his'] == 1 and len(his_type_dict['views']) == 2:
        img_c = [cur_img]
    else:
        raise ValueError(f"{his_type} not valid!")

    if his_type_dict['with_state']:
        human_val = f"What action should the robot take to {task_description}?" + "<|state|>" * 1 + "<|image|>" * len(img_c)
    else:
        human_val = f"What action should the robot take to {task_description}?" + "<|image|>" * len(img_c)

    conv = {
        "conversations":[
            {
                "from": "human",
                "value": human_val
            },
        ],
        "image": img_c,
        "action": [],
    }

    if his_type_dict['with_state']:
        conv['state'] = cur_state

    tokens = item_processor.process_item(conv, training_mode=False)
    print(cur_state, tokens)
    # exit() 
    
    generation_config = GenerationConfig(max_new_tokens=action_steps*12,  #     # 9
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    # if 'img_only' in his_type:
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    # else:
    #     input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    dis_action = model.generate_dis_ma(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action

def get_action_Chameleon_dis_awm_ck_continous_action(model, cur_img, cur_wrist_img, task_description, item_processor, his_img, his_wrist_img, cur_state, his_action, his_type, action_steps):
    
    his_type_dict = parse_his_type(his_type)

    print('his_type_dict:', his_type_dict)

    if his_type_dict['his'] == 2 and len(his_type_dict['views']) == 3:
        img_c = his_img[-1:] + his_wrist_img[-1:] + [cur_img] + [cur_wrist_img]
    elif his_type_dict['his'] == 1 and len(his_type_dict['views']) == 3:
        img_c = [cur_img] + [cur_wrist_img]
    elif his_type_dict['his'] == 2 and len(his_type_dict['views']) == 2:
        img_c = his_img[-1:] + [cur_img]
    elif his_type_dict['his'] == 1 and len(his_type_dict['views']) == 2:
        img_c = [cur_img]
    else:
        raise ValueError(f"{his_type} not valid!")

    if his_type_dict['with_state']:
        human_val = f"What action should the robot take to {task_description}?" + "<|state|>" * 1 + "<|image|>" * len(img_c)
    else:
        human_val = f"What action should the robot take to {task_description}?" + "<|image|>" * len(img_c)

    conv = {
        "conversations":[
            {
                "from": "human",
                "value": human_val
            },
        ],
        "image": img_c,
        "action": [],
    }

    if his_type_dict['with_state']:
        conv['state'] = cur_state
    
    # print(cur_state, conv)

    tokens = item_processor.process_item(conv, training_mode=False)
    tokens.append(10004)

    print('tokens:', tokens)     # add 10004
    # exit() 
    
    generation_config = GenerationConfig(max_new_tokens=1,
                                        max_length=model.config.max_position_embeddings,
                                        temperature=1,
                                        top_k=None,
                                        do_sample=False,
                                        eos_token_id=[8710],
                                    )
    # if 'img_only' in his_type:
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    # else:
    #     input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)[:,:-1]
    dis_action = model.generate_action_head(input_ids, generation_config)
    
    # import pdb; pdb.set_trace()
    
    return dis_action



