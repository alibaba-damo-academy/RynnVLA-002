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

def get_action_Chameleon_dis_awm_g_video(model, task_description, item_processor, his_img, his_action, his_type):
    

    if his_type == "2h_2a":
        conv = {
            "conversations":[
                {
                    "from": "human",
                    "value": "What action should the robot take to " + task_description + "?"
                },
                {
                    "from": "gpt",
                    "value": "<|image|><|action|>"
                },
            ],
            "image": his_img[-1:],
            "action": his_action[-1:],
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
                    "value": "<|image|><|action|>" * len(his_action[-3:])
                },
            ],
            "image": his_img[-3:],
            "action": his_action[-3:],
            }
    elif his_type == "1a2i":
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
    
    generation_config = GenerationConfig(max_new_tokens=1200,
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
