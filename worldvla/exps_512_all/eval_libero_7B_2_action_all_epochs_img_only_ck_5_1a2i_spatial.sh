#!/bin/bash
apt update
apt install libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6 -y
# apt install ffmpeg libsm6 libxext6 libgl1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/workspace/workgroup/cenjun.cj/conda/lumina_libero_a800/lib/libstdc++.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/nas_jianchong/cenjun.cj/grasping/conda_env/lumina_libero_h20/lib/libstdc++.so.6

lr=5e-6
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../configs/libero_512_all/his_2_all_spatial_img_only_ck_5_1a2i.yaml
data_config_val_ind=../configs/libero_512_all/his_2_val_ind_spatial_img_only_ck_5_1a2i.yaml
data_config_val_ood=../configs/libero_512_all/his_2_val_ood_spatial_img_only_ck_5_1a2i.yaml

base_exp_name=eval_libero_ts_spatial_his2imgonly_lr5e6_bs8_ck_5_1a2i_w004_all
base_output_dir=output_libero_512/"$base_exp_name"
mkdir -p "$base_output_dir"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29506)) ../eval_solver_libero_512.py \
    --device 1 \
    --task_suite_name libero_spatial \
    --his 2h_1a_img_only \
    --no_auto_resume \
    --resume_path "../ckpts/chameleon/512_all/libero_spatial" \
    --eval_only True \
    --model_size 7B \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 10 \
    --warmup_epochs 0.01 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 4 \
    --data_config_train $data_config_train \
    --data_config_val_ind $data_config_val_ind \
    --data_config_val_ood $data_config_val_ood \
    --cache_ann_on_disk \
    --num_workers 8 \
    --output_dir "$base_output_dir" \
    --checkpointing \
    --max_seq_len 8192 \
    --unmask_image_logits \
    --dropout ${dropout} \
    --z_loss_weight ${z_loss_weight} \
    --ckpt_max_keep 0 \
    2>&1 | tee -a "$base_output_dir"/output.log &
