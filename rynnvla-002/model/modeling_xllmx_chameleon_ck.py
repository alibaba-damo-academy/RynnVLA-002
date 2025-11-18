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


class ChameleonXLLMXForConditionalGeneration_ck(ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        self.init_input_ids = None

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
                attention_mask = self.generate_att_mask_3(self.init_input_ids)
                kwargs['attention_mask'] = attention_mask.squeeze()[-1:]
            # print(self.init_input_ids)
            # print(kwargs['attention_mask'])
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
            attention_mask = self.generate_att_mask_3(input_ids)
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
            return c_loss, additional_loss_dict, result[1], result[2][-1], labels
        else:
            return c_loss, additional_loss_dict

    
    def generate_att_mask_2(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 创建初始的下三角矩阵作为基础注意力掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()
        
        # 找到所有特殊标记的位置
        image_start = (input_ids == 8197)  # 图像块开始标记
        image_end = (input_ids == 8196)    # 图像块结束标记
        action_start = (input_ids == 10004)  # 动作块开始标记
        action_end = (input_ids == 15004)    # 动作块结束标记

        # 找到每个batch中所有的动作块和图像块的起始和结束位置
        action_blocks = []
        img_blocks = []
        for batch_idx in range(batch_size):
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
            
            # 将动作块的起始和结束位置存储为元组
            action_blocks.append(list(zip(action_starts.cpu().numpy(), action_ends.cpu().numpy())))

            # 找到当前batch的图像块起始和结束位置
            img_starts = torch.where(image_start[batch_idx])[0]
            img_ends = torch.where(image_end[batch_idx])[0]
            
            # 确保图像块的起始和结束位置匹配
            if len(img_starts) != len(img_ends):
                raise ValueError("Mismatched image start and end tokens in batch.")
            
            # 将图像块的起始和结束位置存储为元组
            img_blocks.append(list(zip(img_starts.cpu().numpy(), img_ends.cpu().numpy())))

        # 遍历每个batch并更新mask
        for batch_idx in range(batch_size):
            # 获取当前batch的动作块和图像块
            current_action_blocks = action_blocks[batch_idx]
            current_img_blocks = img_blocks[batch_idx]

            # 找到第一个图像块（如果有）
            first_img_block = None
            if len(current_img_blocks) > 0:
                first_img_block = current_img_blocks[0]

            # 遍历每个动作块
            for block_start, block_end in current_action_blocks:
                # 找到当前动作块之前的所有图像块和动作块
                previous_blocks = [
                    (s, e) for s, e in (current_img_blocks + current_action_blocks) if e < block_start
                ]
                
                # 遍历之前的块，将它们与当前动作块之间的注意力设为0
                for prev_start, prev_end in previous_blocks:
                    # 如果是第一个图像块，则跳过
                    if (prev_start, prev_end) == first_img_block:
                        continue
                    
                    # Mask掉当前动作块与之前的块之间的注意力
                    mask[batch_idx, block_start:block_end + 1, prev_start:prev_end + 1] = 0

        return mask

    

    def generate_att_mask(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 创建初始的下三角矩阵作为基础注意力掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()
        
        # 找到所有特殊标记的位置
        image_start = (input_ids == 8197)  # 图像块开始标记
        image_end = (input_ids == 8196)    # 图像块结束标记
        action_start = (input_ids == 10004)  # 动作块开始标记
        action_end = (input_ids == 15004)    # 动作块结束标记

        # 找到每个batch中所有的动作块的起始和结束位置
        action_blocks = []
        for batch_idx in range(batch_size):
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
            
            # 将动作块的起始和结束位置存储为元组
            action_blocks.append(list(zip(action_starts.cpu().numpy(), action_ends.cpu().numpy())))
        
        # 遍历每个batch并更新mask
        for batch_idx in range(batch_size):
            for block_start, block_end in action_blocks[batch_idx]:
                # 找到当前动作块之前的所有动作块
                previous_action_blocks = [
                    (s, e) for s, e in action_blocks[batch_idx] if e < block_start
                ]
                
                # 如果存在之前的动作块，将当前动作块与这些动作块之间的注意力设为0
                for prev_start, prev_end in previous_action_blocks:
                    mask[batch_idx, block_start:block_end + 1, prev_start:prev_end + 1] = 0
        return mask


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



    
    def generate_dis(self, input_ids, generation_config):
        # import pdb; pdb.set_trace()
        self.init_input_ids = None
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:]
        # import pdb; pdb.set_trace()
        # dis_action = self.decode_token_ids_to_actions(dis_tokens)[0,1:-2].squeeze()
        dis_action = self.decode_token_ids_to_actions(dis_tokens)[0,1:8].squeeze()
        # import pdb; pdb.set_trace()
        return dis_action
    
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
    
    def generate_dis_ma_fast(self, input_ids, generation_config):
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:][0]
        
        res_tokens = None
        for i, token in enumerate(dis_tokens):
            if token == 10004:
                start_index = i
            elif token == 15004:
                end_index = i
                if start_index is not None:
                    res_tokens = dis_tokens[start_index+1:end_index]
                
        return res_tokens


    
    def process_tensor(self, input_ids):
        # 找到每一行中第一个等于8710的位置
        mask = input_ids == 8710
        first_occurrence = mask.to(torch.float).argmax(dim=1)
        
        # 创建一个与 input_ids 相同形状的 mask
        row_indices = torch.arange(input_ids.size(0)).unsqueeze(1).to(self.device)
        col_indices = torch.arange(input_ids.size(1)).unsqueeze(0).to(self.device)
        
        # 将第一次出现8710之后的位置全部设为True
        mask = col_indices > first_occurrence.unsqueeze(1)
        
        # 使用这个mask将对应位置的值存储起来
        stored_values = [row[mask[i]] for i, row in enumerate(input_ids)]
        
        # 使用这个mask将对应位置设为0
        result = input_ids.clone()
        result[mask] = 0
    
        return result, stored_values

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens]
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
