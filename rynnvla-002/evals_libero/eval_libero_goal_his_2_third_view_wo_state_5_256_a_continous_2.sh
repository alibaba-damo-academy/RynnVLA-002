#!/bin/bash
apt update
apt install libegl-dev xvfb libgl1-mesa-dri libgl1-mesa-dev libgl1-mesa-glx libstdc++6 -y
# apt install ffmpeg libsm6 libxext6 libgl1
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/
# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/workspace/workgroup/cenjun.cj/conda/lumina_libero_a800/lib/libstdc++.so.6
# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /mnt/nas_jianchong/cenjun.cj/grasping/conda_env/lumina_libero_h20/lib/libstdc++.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/pai/bin/../lib/libstdc++.so.6

lr=5e-6
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../configs_libero/libero_goal/his_2_third_view_wo_state_5_256_a.yaml
data_config_val_ind=../configs_libero/libero_goal/his_2_third_view_wo_state_5_256_a.yaml
data_config_val_ood=../configs_libero/libero_goal/his_2_third_view_wo_state_5_256_a.yaml
time_horizon=5
epoch_num=24
task_suite=libero_goal
exp_name=his_2_third_view_wo_state_5_256_a
his_setting=his_2_third_view_wo_state
eval_setting=continous
# base_exp_name=libero_goal_his_2_third_view_wo_state_5_256_a
checkpoint_path=/mnt/damorobot/cenjun/libero_outputs_new/"$task_suite"/"$exp_name"/"epoch$epoch_num"

base_output_dir=/mnt/damorobot/cenjun/libero_evals_new_2/"$task_suite"/"$exp_name"/"epoch_$epoch_num"/"$eval_setting"
mkdir -p "$base_output_dir"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29521)) ../eval_solver_libero_continous_wo_state.py \
    --device 2 \
    --task_suite_name $task_suite \
    --his $his_setting \
    --no_auto_resume \
    --resume_path $checkpoint_path \
    --eval_only True \
    --model_size 7B \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs $epoch_num \
    --warmup_epochs 0.01 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 4 \
    --action_dim 7 \
    --time_horizon $time_horizon \
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