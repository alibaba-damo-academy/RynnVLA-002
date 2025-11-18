import torch
import pickle


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

    print(image_blocks, action_blocks, state_blocks)

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

# 为了让函数可以独立运行，我们创建一个模拟的 `self` 对象
class MockModel:
    def __init__(self, device='cpu'):
        self.device = device
    
    generate_att_mask_3_progress = generate_att_mask_3_progress

# --- 示例用法 ---
if __name__ == '__main__':
    model = MockModel()
    
    with open('/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_abs_state_256_minmax/tokens/libero_goal_his_1_train_img_state_abs_ck_1_512/new_files_task_2/0.pkl', 'rb') as f:
      data = pickle.load(f)
      print(data)
    
    # 生成掩码
    attention_mask = model.generate_att_mask_3_progress(torch.tensor(data['token']).unsqueeze(0).cuda())
    # print(attention_mask[0].long())
    print(attention_mask[0][19], )
    print(attention_mask[0][20], )
    print('570: ', attention_mask[0][570], )
    print('571: ', attention_mask[0][571], )
    print('572: ', attention_mask[0][572], )
    print('573: ', attention_mask[0][573], )
    print(attention_mask[0][581], )
    print(attention_mask[0][582], )
    
    