#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "master_addr: $MASTER_ADDR"
echo "master_port: $MASTER_PORT"
echo "rank: $RANK"

# lr=5e-6
lr=1e-5
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config_train=../../configs/libero_10/his_2_third_view_wrist_w_state_10_256_nopretokenize.yaml
data_config_val_ind=../../configs/libero_10/his_2_third_view_wrist_w_state_10_256_nopretokenize.yaml
data_config_val_ood=../../configs/libero_10/his_2_third_view_wrist_w_state_10_256_nopretokenize.yaml
time_horizon=10

exp_name=his_2_third_view_wo_wrist_w_state_10_256_abiw
output_dir=/mnt/workspace/workgroup/yuanyq/libero_output/libero_10
mkdir -p "$output_dir"/"$exp_name"

# torchrun --nnodes=1 --nproc_per_node=8 --master_port=30001 ../pretrain_solver_awm_w_ck_action_head.py \
torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK ../pretrain_solver_awm_w_ck_action_head.py \
--disable_length_clustering \
--train_only True \
--preprocess false \
--with_action \
--with_world_model \
--resolution 256 \
--init_from ../../ckpts/starting_point \
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
--action_dim 7 \
--time_horizon $time_horizon \
--data_config_train $data_config_train \
--data_config_val_ind $data_config_val_ind \
--data_config_val_ood $data_config_val_ood \
--num_workers 16 \
--output_dir "$output_dir"/"$exp_name" \
--checkpointing \
--max_seq_len 4096 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
--ckpt_max_keep 0 \
2>&1 | tee -a "$output_dir"/"$exp_name"/output.log

echo "exp name: $exp_name" 