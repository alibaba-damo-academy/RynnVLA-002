<p align="center">
    <img src="rynnvla-002/assets/logo.png?raw=true" width="80" style="margin-bottom: 0.1;"/>
<p>

<h3 align="center"><a href="" style="color:#9C276A">
RynnVLA-002: A Unified Vision-Language-Action and World Model</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>


<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-AD1C18.svg?logo=arXiv)]() 
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/collections/jcenaa/worldvla-685b9df63bdfe8cb67cc71b2)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE) 
</h5>

## 📰 News



## 🌟 Introduction
RynnVLA-002 is an autoregressive action world model that unifies action and image understanding and generation. RynnVLA-002 intergrates Vision-Language-Action (VLA) model (action model) and world model in one single framework.

<div style="text-align: center;">
  <img src="rynnvla-002/assets/overview.png" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>
<br>

### Action Model Results (Text + Image -> Action)
Action Model generates actions given the text instruction and image observations.

|         |         |         |  
| :-----: | :-----: | :-----: |  
| ![Open drawer](rynnvla-002/assets/action_model_open_the_middle_drawer_of_the_cabinet.gif) | ![Pick up soup](rynnvla-002/assets/action_model_pick_up_the_alphabet_soup_and_place_it_in_the_bask.gif) | ![Pick up bowl](rynnvla-002/assets/action_model_pick_up_the_black_bowl_between_the_plate_and_the_r.gif) |
| Input: Open the middle drawer of the cabinet. | Input: Pick up the alphabet soup and place it in the basket. | Input: Pick up the black bowl between the plate and the ramekin and place it on the plate. |
<br>

### World Model Results (Action + Image -> Image)
World Model generates the next frame given the current frame and action control.

|         |         |         |  
| :-----: | :-----: | :-----: |  
| ![Open drawer](rynnvla-002/assets/world_model_open_the_top_drawer_and_put_the_bowl_inside.gif) | ![Pick up soup](rynnvla-002/assets/world_model_push_the_plate_to_the_front_of_the_stove.gif) | ![Pick up bowl](rynnvla-002/assets/world_model_put_the_bowl_on_the_stove.gif) |
| Input: Action sequence of "Open the top drawer and put the bowl inside". | Input: Action sequence of "Push the plate to the front of the stove". | Input: Action sequence of "Put the bowl on the stove". |
<br>

## 🛠️ Requirements and Installation
```
conda env create -f environment.yml
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

## :earth_americas: Model Zoo

| Model (256 * 256)    |    HF Link        |    Success Rate (%)     |
| :--------------------: | :------------------------------------------------------------: | :--------------------: |
| LIBERO-Spatial       | [jcenaa/WorldVLA-ActionModel-LIBERO-Spatial-256](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Spatial-256/tree/main) | 85.6 |
| LIBERO-Object       | [jcenaa/WorldVLA-ActionModel-LIBERO-Object-256](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Object-256/tree/main) | 89.0 |
| LIBERO-Goal | [jcenaa/WorldVLA-ActionModel-LIBERO-Goal-256](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Goal-256/tree/main) | 82.6 |
| LIBERO-Long | [jcenaa/WorldVLA-ActionModel-LIBERO-10-256](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-10-256/tree/main) | 59.0 |
<br>

| Model (512 * 512)    |    HF Link        |    Success Rate (%)     |
| :--------------------: | :------------------------------------------------------------: | :--------------------: |
| LIBERO-Spatial       | [jcenaa/WorldVLA-ActionModel-LIBERO-Spatial-512](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Spatial-512/tree/main) | 87.6 |
| LIBERO-Object       | [jcenaa/WorldVLA-ActionModel-LIBERO-Object-512](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Object-512/tree/main) | 96.2 |
| LIBERO-Goal | [jcenaa/WorldVLA-ActionModel-LIBERO-Goal-512](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-Goal-512/tree/main) | 83.4 |
| LIBERO-Long | [jcenaa/WorldVLA-ActionModel-LIBERO-10-512](https://huggingface.co/jcenaa/WorldVLA-ActionModel-LIBERO-10-512/tree/main) | 60.0 |


## 🗝️ Training
We evaluate four tasks of the LIBERO benchmark, including [spatial, obejct, goal, 10]. Here we take LIEBRO goal and 256 resolution as an example.

We offer two types of training pipelines:

- `Pretokenize`: This pipeline preprocesses all the training data by tokenizing it into tokens before the training begins.
- `NoPretokenize`: This pipeline performs tokenization dynamically during the training process.

Both pipelines begin by filtering out no-operation actions like [OpenVLA](https://github.com/openvla/openvla).

```bash
cd rynnvla-002/libero_util
python regenerate_libero_dataset_filter_no_op.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir ../processed_data/Libero/libero_goal \
    --libero_target_dir ../processed_data/libero_goal_no_noops_t_256 \
    --image_resolution 256
```

After filtering, you can choose between the `Pretokenize` or `NoPretokenize` training pipeline. The `Pretokenize` pipeline offers faster training speeds, while the `NoPretokenize` option eliminates the need for preprocessing.


### Pipeline1: Pretokenize

#### Step 0: Lerobot to HDF5

We use HDF5 format data. Therefore, if you collect data in Lerobot format, you can follow the following command to process it into HDF5 format:
```
cd rynnvla-002/libero_util
python lerobot_to_hdf5.py \
    --lerobot_input_dir {lerobot_input_dir}
    --hdf5_output_dir {hdf5_output_dir}
```

#### Step 1: Libero Data Preparation

After filtering out no-operation actions, save all images and actions.
```bash
python regenerate_libero_dataset_save_img_action_state_wrist.py \
    --libero_task_suite libero_goal \
    --image_resolution 256 \
    --raw_data_dir ../processed_data/libero_goal_no_noops_t_256 \
    --save_dir ../processed_data/libero_goal_image_state_action_t_256
``` 
Next, generate the conversations data for the Chameleon model. The action model conversations are in the following format:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "What action should the robot take to open the middle drawer of the cabinet?<|state|><|image|><|image|><|image|><|image|>"
    },
    {
      "from": "gpt",
      "value": "<|action|><|action|><|action|><|action|><|action|>"
    }
  ],
  "image": [
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_0.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_0.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_1.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_1.png"
  ],
  "action": [
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_1.npy",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_2.npy",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_3.npy",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_4.npy",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_5.npy"
  ]
}
```
The world model conversations are in the following format:
```json
{
  "conversations": [
    {
        "from": "human",
        "value": "Generate the next image based on the provided sequence of historical images and corresponding actions.<|image|><|image|><|action|>"
    },
    {
        "from": "gpt",
        "value": "<|image|><|image|>"
    }
  ],
  "image": [
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_0.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_0.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_third_view/image_1.png",
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/imgs_wrist/image_1.png"
  ],
  "action": [
    "../processed_data/libero_goal_image_state_action_t_256/open_the_middle_drawer_of_the_cabinet/trj_0/action/action_0.npy"
  ]
},
```
To validate the world model performance, we split all the libero dataset into train/val_ind/val_ood json files.
```bash
cd rynnvla-002/data
python action_state_model_conv_generation.py \
    --base_dir ../processed_data/libero_goal_image_state_action_t_256 \
    --his 2 \
    --len_action 5 \
    --task_name goal \
    --resolution 256 \
    --with_state \
    --img_names imgs_third_view imgs_wrist \
    --output_dir ../processed_data/convs
python world_model_bi_views_conv_generation.py \
    --base_dir ../processed_data/libero_goal_image_state_action_t_256 \
    --his 1 \
    --task_name goal \
    --resolution 256 \
    --output_dir ../processed_data/convs
```
Finally, tokenize all the conversations into tokens and save them.
```bash
cd rynnvla-002/data
python pretoken_state_action_model.py --task goal --resolution 256 --with_state --img_names imgs_third_view imgs_wrist --his 2 --len_action 5 --tokenizer_path Alpha-VLLM/Lumina-mGPT-7B-768
python pretoken_world_model.py --task goal --resolution 256 --img_name imgs_third_view imgs_wrist --tokenizer_path Alpha-VLLM/Lumina-mGPT-7B-768
bash concate_record_libero.sh
python concate_action_world_model_data_libero.py --source_dir_patterns libero_goal_his_2_{}_third_view_wrist_w_state_5_256 libero_goal_his_1_{}_third_view_wrist_a2i_256 --all_patterns libero_goal_his_2_third_view_wrist_w_state_5_256_abiw
```

#### Step 2: Prepare data configs
Set the correct data path in the config files in `rynnvla-002/configs/libero_goal/his_2_third_view_wrist_w_state_5_256_pretokenize.yaml`.

#### Step 3: Start training
Now you can start training with your training scripts:
```bash
# Libero goal, 256 resolution
cd rynnvla-002/exps_pretokenize
bash libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.sh
```




### Pipeline2: NoPretokenize
#### Step 1: Prepare data configs
Set the correct data path in the config files in `rynnvla-002/configs/libero_goal/his_2_third_view_wrist_w_state_5_256_nopretokenize.yaml`.

#### Step 2: Start training
```bash
# Libero goal, 256 resolution
cd rynnvla-002/exps_nopretokenize
bash libero_goal_his_2_third_view_wrist_w_state_5_256_abiw.sh
```


## ✅ Evaluation
### Step 1: Prepare evaluation scripts
Set the `checkpoint_path` in the bash files in `rynnvla-002/evals_libero/` to the model path. You can download our trained in Model Zoo or train yourself.

### Step 2: Start evaluation
```bash
# Libero goal, 256 resolution, continous
cd rynnvla-002/evals_libero
bash eval_libero_goal_his_2_third_view_wrist_w_state_5_256_abiw_continous.sh
# Libero goal, 256 resolution, discrete
cd rynnvla-002/evals_libero
bash eval_libero_goal_his_2_third_view_wrist_w_state_5_256_abiw_discrete.sh
```



## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>
If you find the project helpful for your research, please consider citing our paper:
```bibtex
@article{cen2025WorldVLA,
  title={WorldVLA: Towards Autoregressive Action World Model},
  author={Cen, Jun and Yu, Chaohui and Yuan, Hangjie and Jiang, Yuming and Huang, Siteng and Guo, Jiayan and Li, Xin and Song, Yibing and Luo, Hao and Wang, Fan and Zhao, Deli and Chen, Hao},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

## Acknowledgment <a name="acknowledgment"></a>
This project builds upon [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT), [Chemeleon](https://github.com/facebookresearch/chameleon), and [OpenVLA](http://github.com/openvla/openvla). We thank these teams for their open-source contributions.
