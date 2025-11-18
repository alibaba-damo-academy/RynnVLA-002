import functools
import logging
import math
from typing import List

import torch
from torch import nn

from .chameleon import ChameleonForConditionalGeneration
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


__all__ = ["ChameleonXLLMXForConditionalGeneration_ck"]

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        time_horizon=15,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*action_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )
        
    def __call__(self, x):
        return self.predict_action(x)

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.time_horizon, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action

class ActionHead(nn.Module):
    def __init__(self, action_dim=7, time_horizon=8, hidden_size_factor=0.25, num_encoder_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.num_encoder_layers = num_encoder_layers
        self.hidden_size = 4096
        self.reduced_hidden_size = int(self.hidden_size * hidden_size_factor)
        self.action_token_embeddings = nn.Embedding(1, time_horizon * action_dim * self.hidden_size)
        nn.init.normal_(self.action_token_embeddings.weight, std=0.02)
        self.hidden_projection = nn.Linear(self.hidden_size, self.reduced_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_hidden_size,
            nhead=4,
            dim_feedforward=self.reduced_hidden_size * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.reduced_hidden_size)
        )
        self.output_projection = L1RegressionActionHead(
            self.reduced_hidden_size, self.reduced_hidden_size, self.time_horizon, self.action_dim
        )
        
    def forward(self, hidden_states, input_ids, attention_mask=None, target_token_id=10004, eval=False):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - 从模型得到的hidden states
            input_ids: [batch_size, seq_len] - 对应的input_ids
            attention_mask: [batch_size, seq_len] - 注意力掩码（可选）
            target_token_id: int - 目标token id (默认10004)
        
        Returns:
            actions: 预测的动作序列
        """
        # # 检查输入是否有NaN
        # print("=== NaN Debug Info ===")
        # print(f"Input hidden_states has NaN: {torch.isnan(hidden_states).any()}")
        # if torch.isnan(hidden_states).any():
        #     print(f"NaN positions in hidden_states: {torch.isnan(hidden_states).nonzero()}")
        
        batch_size = hidden_states.shape[0]
        action_tokens = self.action_token_embeddings.weight.view(1, self.time_horizon * self.action_dim, self.hidden_size).expand(batch_size, -1, -1)
        
        # print(f"Action tokens has NaN: {torch.isnan(action_tokens).any()}")
        # if torch.isnan(action_tokens).any():
        #     print(f"NaN positions in action_tokens: {torch.isnan(action_tokens).nonzero()}")
        
        # 第一步：提取每一行第一个target_token_id之前的token的hidden states
        extracted_hidden_states = []
        extracted_attention_masks = []

        flag = True
        
        for i in range(batch_size):
            # 找到第一个target_token_id的位置
            target_positions = (input_ids[i] == target_token_id).nonzero(as_tuple=True)[0]
            if len(target_positions) > 1 or eval:
                # 取第一个target_token_id之前的所有token
                end_pos = target_positions[0].item()
            else:
                continue
            
            # print(f"Batch {i}: end_pos = {end_pos}")
            
            # 提取对应的hidden states
            extracted_hidden = hidden_states[i, :end_pos, :]  # [end_pos, hidden_size]
            # print(f"Batch {i}: extracted_hidden has NaN: {torch.isnan(extracted_hidden).any()}")
            extracted_hidden_states.append(extracted_hidden)
            
            # 提取对应的attention mask（如果提供）
            if attention_mask is not None:
                extracted_mask = attention_mask[i, :end_pos]
                extracted_attention_masks.append(extracted_mask)
        
        if len(extracted_hidden_states) == 0:
            extracted_hidden_states.append(hidden_states[0, 0:1, :])
            flag = False
                
        # 第二步：为每一行添加action_tokens
        combined_states_list = []
        combined_attention_masks = []
        max_length = 0
        
        for i in range(len(extracted_hidden_states)):
            # 将当前样本的hidden states与action tokens拼接
            combined_hidden = torch.cat([extracted_hidden_states[i], action_tokens[i]], dim=0)
            # print(f"Batch {i}: combined_hidden has NaN: {torch.isnan(combined_hidden).any()}")
            combined_states_list.append(combined_hidden)
            
            # 处理attention mask
            if attention_mask is not None:
                # 为action tokens创建mask (全为1)
                action_tokens_mask = torch.ones(self.time_horizon * self.action_dim, 
                                            device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([extracted_attention_masks[i], action_tokens_mask], dim=0)
                combined_attention_masks.append(combined_mask)
            
            # 记录最大长度
            max_length = max(max_length, combined_hidden.shape[0])
                
        # 第三步：补全所有序列到相同长度
        padded_hidden_states = []
        padded_attention_masks = []
        
        for i in range(len(extracted_hidden_states)):
            current_length = combined_states_list[i].shape[0]
            if current_length < max_length:
                # 用零向量补全hidden states
                padding = torch.zeros(max_length - current_length, self.hidden_size, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
                padded_hidden = torch.cat([combined_states_list[i], padding], dim=0)
            else:
                padded_hidden = combined_states_list[i]
            
            # print(f"Batch {i}: padded_hidden has NaN: {torch.isnan(padded_hidden).any()}")
            padded_hidden_states.append(padded_hidden)
            
            # 补全attention mask
            if attention_mask is not None:
                current_mask_length = combined_attention_masks[i].shape[0]
                if current_mask_length < max_length:
                    mask_padding = torch.zeros(max_length - current_mask_length, 
                                            device=attention_mask.device, dtype=attention_mask.dtype)
                    padded_mask = torch.cat([combined_attention_masks[i], mask_padding], dim=0)
                else:
                    padded_mask = combined_attention_masks[i]
                padded_attention_masks.append(padded_mask)
        
        # 堆叠成batch
        processed_hidden_states = torch.stack(padded_hidden_states, dim=0)  # [batch_size, max_length, hidden_size]
        # print(f"Processed hidden_states has NaN: {torch.isnan(processed_hidden_states).any()}")
        
        if attention_mask is not None:
            processed_attention_mask = torch.stack(padded_attention_masks, dim=0)  # [batch_size, max_length]
        else:
            processed_attention_mask = torch.ones(len(extracted_hidden_states), processed_hidden_states.shape[1], 
                                                device=processed_hidden_states.device)
        
        # print(f"Processed attention_mask has NaN: {torch.isnan(processed_attention_mask).any()}")
        
        # 投影到较小的维度
        projected_states = self.hidden_projection(processed_hidden_states)
        # print(f"Projected states has NaN: {torch.isnan(projected_states).any()}")
        
        # 检查hidden_projection层的权重
        # if hasattr(self.hidden_projection, 'weight'):
        #     print(f"Hidden projection weight has NaN: {torch.isnan(self.hidden_projection.weight).any()}")
        #     if hasattr(self.hidden_projection, 'bias') and self.hidden_projection.bias is not None:
        #         print(f"Hidden projection bias has NaN: {torch.isnan(self.hidden_projection.bias).any()}")
        
        # 通过transformer encoder
        transformer_output = self.transformer_encoder(
            projected_states,
            src_key_padding_mask=(1 - processed_attention_mask).bool()
        )
        # print(f"Transformer output has NaN: {torch.isnan(transformer_output).any()}")
        
        # 检查transformer encoder的参数
        # for name, param in self.transformer_encoder.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"Transformer encoder parameter {name} has NaN")
        
        # 第四步：提取action tokens对应的输出
        action_outputs = []
        for i in range(len(extracted_hidden_states)):
            # 计算当前样本原始序列长度
            original_length = extracted_hidden_states[i].shape[0]
            # action tokens在transformer输出中的位置
            action_start = original_length
            action_end = action_start + self.time_horizon * self.action_dim
            
            # 边界检查
            if action_end > transformer_output.shape[1]:
                print(f"Warning: action_end ({action_end}) > sequence length ({transformer_output.shape[1]}) for batch {i}")
                action_end = transformer_output.shape[1]
            
            # 提取action tokens对应的输出
            action_output_i = transformer_output[i, action_start:action_end, :]  # [time_horizon * action_dim, reduced_hidden_size]
            # print(f"Batch {i}: action_output has NaN: {torch.isnan(action_output_i).any()}")
            action_outputs.append(action_output_i)
                
        # 将所有action outputs堆叠
        action_outputs_tensor = torch.stack(action_outputs, dim=0)  # [batch_size, time_horizon * action_dim, reduced_hidden_size]
        # print(f"Action outputs tensor has NaN: {torch.isnan(action_outputs_tensor).any()}")
        
        # 生成最终的动作预测
        actions = self.output_projection(action_outputs_tensor)
        actions = actions.reshape(-1, self.action_dim)
        # print(f"Final actions has NaN: {torch.isnan(actions).any()}")
        
        # 检查output_projection层的权重
        # if hasattr(self.output_projection, 'weight'):
        #     print(f"Output projection weight has NaN: {torch.isnan(self.output_projection.weight).any()}")
        #     if hasattr(self.output_projection, 'bias') and self.output_projection.bias is not None:
        #         print(f"Output projection bias has NaN: {torch.isnan(self.output_projection.bias).any()}")
        
        # print("=== End NaN Debug Info ===")
        
        return actions, flag




class ChameleonXLLMXForConditionalGeneration_ck_action_head_progress(ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        self.init_input_ids = None
        # self.action_dim = 7
        # self.action_head = ActionHead(action_dim=self.action_dim, time_horizon=5, hidden_size_factor=0.25, num_encoder_layers=2)
        # self.action_dim = 6
        # self.action_head = ActionHead(action_dim=self.action_dim, time_horizon=20, hidden_size_factor=0.25, num_encoder_layers=2)
        self.action_dim = config.action_dim
        self.action_head = ActionHead(action_dim=config.action_dim, time_horizon=config.time_horizon, hidden_size_factor=0.25, num_encoder_layers=2)
        self.post_init()
        

    def forward(self, input_ids=None, labels=None, training=False, att_mask=True, **kwargs):

        if not training:
            # import pdb; pdb.set_trace()
            if self.init_input_ids is None:
                self.init_input_ids = input_ids
            else:
                self.init_input_ids = torch.cat([self.init_input_ids, input_ids], dim=-1)
            if not att_mask:
                attention_mask = None
            else:
                attention_mask = self.generate_att_mask_3_progress(self.init_input_ids)
                kwargs['attention_mask'] = attention_mask.squeeze()[-1:]
            # print(self.init_input_ids)
            print(kwargs['attention_mask'])
            # import pdb; pdb.set_trace()
            result = ChameleonForConditionalGeneration.forward(
                self, input_ids=input_ids, **kwargs
            )
            return result

        # import pdb; pdb.set_trace()
        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        if not att_mask:
            attention_mask = None
        else:
            attention_mask = self.generate_att_mask_3_progress(input_ids)
        # import pdb; pdb.set_trace()
                
        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, attention_mask=attention_mask, **kwargs
        )

        # import pdb; pdb.set_trace()

        c_loss = result[0]

        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[1]                   # [8, 1266, 65536]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)
        
        if 'output_hidden_states' in kwargs:
            # c_loss, additional_loss_dict, logits, hidden_states, labels_c
            hidden_states = result[2][-1]  # [batch_size, seq_len, hidden_dim]
            
            # 调用ActionHead来预测动作
            predicted_actions, actions_flag = self.action_head(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=None,
                target_token_id=10004
            )

            if actions_flag == False:
                return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, predicted_actions.mean()*0
            
            # print(f"Predicted actions shape: {predicted_actions.shape}")
            # print(f"Predicted actions: {predicted_actions}")

            labels_action_dis, sequences = self.get_action_hs_label(result[2][-1], labels)
            labels_action_ct = self.decode_token_ids_to_actions(labels_action_dis)

            loss_ct = torch.nn.functional.l1_loss(predicted_actions, labels_action_ct)

            # print(f"Predicted actions shape: {predicted_actions.shape}", f"GT actions shape: {labels_action_ct.shape}")

            # import pdb; pdb.set_trace()
            
            return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, loss_ct
        else:
            return c_loss, additional_loss_dict
    
    def get_action_hs_label(self, hidden_states, labels_c):

        # 找到所有符合条件的序列
        sequences = self.find_sequences(labels_c)

        # 初始化结果张量
        labels_action = torch.zeros(len(sequences), self.action_dim, dtype=torch.long, device=self.device)
        
        # 填充结果张量
        for i, (batch, start) in enumerate(sequences):
            labels_action[i] = labels_c[batch, start:start+self.action_dim]
        
        return labels_action, sequences
    
    def find_sequences(self, tensor_input):
        # 找到所有以 10004 开始，15005 结束的序列
        start_indices = (tensor_input[:, :-1*self.action_dim+1] == 10004).nonzero(as_tuple=True)
        valid_sequences = []
        for batch, start in zip(*start_indices):
            if tensor_input[batch, start+self.action_dim+1] == 15004:
                valid_sequences.append((batch, start+1))
        return valid_sequences


    def generate_att_mask_3(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 创建初始的下三角矩阵作为基础注意力掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()
        
        # 找到所有特殊标记的位置
        image_start = (input_ids == 8197)  # 图像块开始标记
        image_end = (input_ids == 8196)    # 图像块结束标记
        action_start = (input_ids == 10004)  # 动作块开始标记
        action_end = (input_ids == 15004)    # 动作块结束标记

        # 找到每个batch中所有的图像块和动作块的起始和结束位置
        image_blocks = []
        action_blocks = []
        for batch_idx in range(batch_size):
            # 找到当前batch的图像块起始和结束位置
            image_starts = torch.where(image_start[batch_idx])[0]
            image_ends = torch.where(image_end[batch_idx])[0]
            
            # 如果图像块的起始和结束位置不匹配
            if len(image_starts) > len(image_ends):
                # 将当前batch的最后一个位置作为缺失的结束标记
                last_position = seq_len - 1
                image_ends = torch.cat([image_ends, torch.tensor([last_position], dtype=torch.long, device=self.device)])
            elif len(image_starts) < len(image_ends):
                image_ends = image_ends[:-1]
            
            # 确保图像块的起始和结束位置匹配
            if len(image_starts) != len(image_ends):
                raise ValueError("Mismatched image start and end tokens in batch.")
            
            # 存储图像块的起始和结束位置
            image_blocks.append(list(zip(image_starts.cpu().numpy(), image_ends.cpu().numpy())))
            
            # 找到当前batch的动作块起始和结束位置
            action_starts = torch.where(action_start[batch_idx])[0]
            action_ends = torch.where(action_end[batch_idx])[0]
            
            # 如果动作块的起始和结束位置不匹配
            if len(action_starts) > len(action_ends):
                # 将当前batch的最后一个位置作为缺失的结束标记
                last_position = seq_len - 1
                action_ends = torch.cat([action_ends, torch.tensor([last_position], dtype=torch.long, device=self.device)])
            elif len(action_starts) < len(action_ends):
                action_ends = action_ends[:-1]
            
            # 确保动作块的起始和结束位置匹配
            if len(action_starts) != len(action_ends):
                raise ValueError("Mismatched action start and end tokens in batch.")
            
            # 存储动作块的起始和结束位置
            action_blocks.append(list(zip(action_starts.cpu().numpy(), action_ends.cpu().numpy())))

        # 遍历每个batch并更新mask
        for batch_idx in range(batch_size):
            # 获取当前batch的图像块和动作块
            current_image_blocks = image_blocks[batch_idx]
            current_action_blocks = action_blocks[batch_idx]
            
            # 找到最后一个图像块的结束位置
            if current_image_blocks:
                last_image_end = current_image_blocks[-1][1]  # 最后一个图像块的结束位置
            else:
                last_image_end = -1  # 如果没有图像块，则认为所有动作块都在图像块之后
            
            # 遍历当前batch的所有动作块
            for block_start, block_end in current_action_blocks:
                # 判断当前动作块是否在最后一个图像块之后
                if block_start > last_image_end:
                    # 找到当前动作块之前的所有动作块
                    previous_action_blocks = [
                        (s, e) for s, e in current_action_blocks if e < block_start
                    ]
                    
                    # 如果存在之前的动作块，将当前动作块与这些动作块之间的注意力设为0
                    for prev_start, prev_end in previous_action_blocks:
                        mask[batch_idx, block_start:block_end + 1, prev_start:prev_end + 1] = 0
                else:
                    # 如果当前动作块不在最后一个图像块之后，则保持注意力为1
                    pass  # 默认情况下已经是1，无需额外操作
        
        return mask
    
    def generate_att_mask_3_progress(self, input_ids):
        """
        生成一个复杂的注意力掩码，结合了因果掩码和针对特殊token块（图像、动作、状态）的特定规则。

        规则：
        1. 基础是因果注意力掩码（token只能关注自身和之前的token）。
        2. 在最后一个图像块之后出现的动作块，不能关注任何之前的动作块。
        3. 对于每个状态块，位于该状态块之后、但在下一个动作块开始之前的所有token，都不能关注该状态块。
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. 创建初始的下三角矩阵作为基础因果注意力掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()
        
        # 2. 找到所有特殊标记的位置
        image_start_id = 8197
        image_end_id = 8196
        action_start_id = 10004
        action_end_id = 15004
        state_start_id = 15504
        state_end_id = 16004
        
        image_start = (input_ids == image_start_id)
        image_end = (input_ids == image_end_id)
        action_start = (input_ids == action_start_id)
        action_end = (input_ids == action_end_id)
        state_start = (input_ids == state_start_id)
        state_end = (input_ids == state_end_id)

        # 3. 找到每个batch中所有特殊块的起始和结束位置
        image_blocks = []
        action_blocks = []
        state_blocks = []
        for batch_idx in range(batch_size):
            # --- 图像块 ---
            image_starts = torch.where(image_start[batch_idx])[0]
            image_ends = torch.where(image_end[batch_idx])[0]
            if len(image_starts) > len(image_ends):
                last_position = torch.tensor([seq_len - 1], dtype=torch.long, device=self.device)
                image_ends = torch.cat([image_ends, last_position])
            elif len(image_starts) < len(image_ends):
                image_starts = torch.cat([torch.tensor([0], dtype=torch.long, device=self.device), image_starts]) # 假设缺失的开始在最前面
            
            if len(image_starts) != len(image_ends): raise ValueError("Mismatched image start/end tokens.")
            image_blocks.append(list(zip(image_starts.cpu().numpy(), image_ends.cpu().numpy())))
            
            # --- 动作块 ---
            action_starts = torch.where(action_start[batch_idx])[0]
            action_ends = torch.where(action_end[batch_idx])[0]
            if len(action_starts) > len(action_ends):
                last_position = torch.tensor([seq_len - 1], dtype=torch.long, device=self.device)
                action_ends = torch.cat([action_ends, last_position])
            elif len(action_starts) < len(action_ends):
                action_starts = torch.cat([torch.tensor([0], dtype=torch.long, device=self.device), action_starts])

            if len(action_starts) != len(action_ends): raise ValueError("Mismatched action start/end tokens.")
            action_blocks.append(list(zip(action_starts.cpu().numpy(), action_ends.cpu().numpy())))

            # --- 状态块 ---
            state_starts = torch.where(state_start[batch_idx])[0]
            state_ends = torch.where(state_end[batch_idx])[0]
            if len(state_starts) > len(state_ends):
                last_position = torch.tensor([seq_len - 1], dtype=torch.long, device=self.device)
                state_ends = torch.cat([state_ends, last_position])
            elif len(state_starts) < len(state_ends):
                state_starts = torch.cat([torch.tensor([0], dtype=torch.long, device=self.device), state_starts])

            if len(state_starts) != len(state_ends): raise ValueError("Mismatched state start/end tokens.")
            state_blocks.append(list(zip(state_starts.cpu().numpy(), state_ends.cpu().numpy())))


        # 4. 遍历每个batch并根据规则更新mask
        for batch_idx in range(batch_size):
            current_image_blocks = image_blocks[batch_idx]
            current_action_blocks = action_blocks[batch_idx]
            current_state_blocks = state_blocks[batch_idx]
            
            # --- 规则：最后一个图像块之后的动作块，不能关注之前的动作块 ---
            last_image_end = current_image_blocks[-1][1] if current_image_blocks else -1
            
            for block_start, block_end in current_action_blocks:
                if block_start > last_image_end:
                    previous_action_blocks = [(s, e) for s, e in current_action_blocks if e < block_start]
                    for prev_start, prev_end in previous_action_blocks:
                        # 将当前动作块对之前动作块的注意力设置为0
                        mask[batch_idx, block_start:block_end + 1, prev_start:prev_end + 1] = 0
                # else: 默认因果掩码已生效，无需操作
            
            # --- 新增规则：状态块之后的、下一个动作块之前的token，不能关注该状态块 ---
            for state_s, state_e in current_state_blocks:
                # 找到紧跟在该状态块之后的第一个动作块的起始位置
                next_action_start_pos = seq_len  # 默认为序列末尾
                for action_s, _ in sorted(current_action_blocks): # 按位置排序以确保找到的是第一个
                    if action_s > state_e:
                        next_action_start_pos = action_s
                        break
                
                # 定义需要被屏蔽的查询token范围 (query tokens)
                # 从状态块结束后一位，到下一个动作块开始前一位
                query_start = state_e + 1
                query_end = next_action_start_pos
                
                # 定义被屏蔽的键/值token范围 (key/value tokens)，即当前状态块
                key_start = state_s
                key_end = state_e + 1
                
                # 如果存在这样的查询范围，则将它们对状态块的注意力设置为0
                if query_start < query_end:
                    mask[batch_idx, query_start:query_end, key_start:key_end] = 0
        
        return mask
    
    def generate_img(self, input_ids, generation_config):
        # res = ChameleonForConditionalGeneration.generate(
        #     self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True, use_cache=True, past_key_values=past_key_values
        # )
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:]
        # dis_tokens = res['sequences']
        # import pdb; pdb.set_trace()
        return dis_tokens
    
    def generate_dis_ma(self, input_ids, generation_config):
        self.init_input_ids = None
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:][0]
        decoded_actions = self.decode_token_ids_to_actions(dis_tokens)
        
        action_sequences = []
        for i, token in enumerate(dis_tokens):
            if token == 10004:
                start_index = i
            elif token == 15004:
                end_index = i
                if start_index is not None:
                    action_sequences.append(decoded_actions[start_index+1:end_index])
                start_index = None
                
        return action_sequences
    
    def generate_action_head(self, input_ids, generation_config):
        """
        生成一个token（期望为10004），然后使用action_head预测动作
        """
        self.init_input_ids = None
        
        # 生成一个token（期望为10004）
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, 
            output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        
        # 获取生成的token
        generated_token = res['sequences'][:, input_ids.shape[1]:]  # [batch_size, 1]
        # print(input_ids, res['sequences'])
        print(f"Generated token: {generated_token}")  # 调试信息，确认是否为10004
        
        # 构建完整的input_ids（原始输入 + 生成的token）
        full_input_ids = res['sequences']  # [batch_size, original_length + 1]
        
        # 获取最后一层的hidden states
        # hidden_states是一个tuple，每个元素对应一个生成步骤的hidden states
        # 我们需要最后一步的最后一层hidden states
        # last_step_hidden_states = res['hidden_states'][-1][-1]  # [batch_size, seq_len, hidden_dim]

        new_token_hidden_states_list = [
            step_hidden_states[-1] for step_hidden_states in res['hidden_states']
        ]
        last_step_hidden_states = torch.cat(new_token_hidden_states_list, dim=1)
        print(last_step_hidden_states.shape, full_input_ids.shape)
        print(last_step_hidden_states[0,0], last_step_hidden_states[0,-2], last_step_hidden_states[0,-1])
        
        # 使用action_head预测动作
        predicted_actions, actions_flag = self.action_head(
            hidden_states=last_step_hidden_states,
            input_ids=full_input_ids,
            attention_mask=None,
            target_token_id=10004,
            eval=True
        )
        
        # 检查是否成功预测动作
        if not actions_flag:
            print("Warning: Action prediction failed, returning zero actions")
            return torch.zeros(self.action_head.time_horizon, self.action_head.action_dim, device=input_ids.device)
        
        # 将predicted_actions重新reshape为[time_horizon, action_dim]
        predicted_actions = predicted_actions.reshape(self.action_head.time_horizon, self.action_head.action_dim)
        
        print(f"Predicted actions shape: {predicted_actions.shape}")
        # print(f"Predicted actions: {predicted_actions}")
        
        return predicted_actions



    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens, self.action_head]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        return modules
    
    def decode_token_ids_to_actions(self, dis_action):
        bins = torch.linspace(-1, 1, 256, device=dis_action.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        discretized_actions = dis_action - 1 - 10004
        discretized_actions = torch.clamp(discretized_actions - 1, min=0, max=bin_centers.shape[0] - 1).long()
        return bin_centers[discretized_actions]