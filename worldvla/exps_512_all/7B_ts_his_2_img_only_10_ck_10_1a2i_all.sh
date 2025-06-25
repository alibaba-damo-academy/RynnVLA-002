#!/bin/bash

lr=5e-6
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../configs/libero_512_all/his_2_all_10_img_only_ck_10_1a2i.yaml
data_config_val_ind=../configs/libero_512_all/his_2_val_ind_10_img_only_ck_10_1a2i.yaml
data_config_val_ood=../configs/libero_512_all/his_2_val_ood_10_img_only_ck_10_1a2i.yaml

exp_name=libero_ts_10_his2imgonly_lr5e6_bs8_ck_10_1a2i_w004_all
output_dir=output_libero_512
mkdir -p "$output_dir"/"$exp_name"

# torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK pretrain_solver_awm_w_ck.py \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=30001 ../pretrain_solver_awm_w_ck.py \
--disable_length_clustering \
--ablation 0 \
--model_size 7B \
--batch_size 8 \
--accum_iter 1 \
--epochs 40 \
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
--output_dir "$output_dir"/"$exp_name" \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
--ckpt_max_keep 0 \
2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "exp name: $exp_name"
