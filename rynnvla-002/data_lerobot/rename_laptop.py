import json
import os
import pickle
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
from tqdm import tqdm
import math

# --- 用户配置区域 ---

# 请将这个路径替换为您文件的确切路径
input_file_path = "/mnt/PLNAS/cenjun/all_data/extracted/convs/libero_goal_his_1_train_img_state_rel_ck_1_512.json"
input_file_path_2 = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record_unique.json"
print(input_file_path)

# --- Token 替换规则 ---
original_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 29413, 16604, 16399, 16402, 16414]
blocks_replacement_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 27795, 25426, 16414]
strawberries_replacement_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 27795, 32723, 40410, 16414]

# 预先计算前缀长度
ORIGINAL_PREFIX_LEN = len(original_prefix)

original_label = [-100 for i in range(ORIGINAL_PREFIX_LEN)]
blocks_replacement_label = [-100 for i in range(len(blocks_replacement_prefix))]
strawberries_replacement_label = [-100 for i in range(len(strawberries_replacement_prefix))]


# --- 多进程配置 ---
NUM_PROCESSES = 128


def process_chunk_and_modify(task_chunk):
    """
    工作函数，由每个子进程执行。
    它接收一个任务块，处理文件，并返回统计结果。
    """
    local_counts = Counter()
    
    for condition_item, token_record in task_chunk:
        try:
            conversation_value = condition_item.get('conversations', [{}])[0].get('value')
            
            if conversation_value != 'What action should the robot take to laptop 03?<|state|><|image|><|image|>':
                continue

            image_path = condition_item.get('image', [None])[0]
            if image_path is None:
                continue

            original_pkl_path = token_record['file']
            replacement_prefix = None
            replacement_prefix_label = None
            scene_type = None

            if 'blocks' in image_path:
                replacement_prefix = blocks_replacement_prefix
                replacement_prefix_label = blocks_replacement_label
                scene_type = 'blocks'
            elif 'strawberries' in image_path:
                replacement_prefix = strawberries_replacement_prefix
                replacement_prefix_label = strawberries_replacement_label
                scene_type = 'strawberries'
            
            if replacement_prefix is None:
                continue

            # --- 文件读写和修改逻辑 ---
            # 读取原始pickle文件
            with open(original_pkl_path, 'rb') as f_in:
                tokens_data = pickle.load(f_in)
            
            original_tokens = tokens_data['token']
            original_labels = tokens_data['label']
            
            # 安全检查：确认文件的前缀与预期一致
            if len(original_tokens) >= ORIGINAL_PREFIX_LEN and original_tokens[:ORIGINAL_PREFIX_LEN] == original_prefix:
                # 替换前缀
                new_tokens = replacement_prefix + original_tokens[ORIGINAL_PREFIX_LEN:]
                tokens_data['token'] = new_tokens

                new_labels = replacement_prefix_label + original_labels[ORIGINAL_PREFIX_LEN:]
                tokens_data['label'] = new_labels

                assert len(tokens_data['token']) == len(tokens_data['label'])

                # 生成新文件路径
                if '/files/' in original_pkl_path:
                    new_pkl_path = original_pkl_path.replace('/files/', '/new_files/', 1)
                    new_dir = os.path.dirname(new_pkl_path)
                    os.makedirs(new_dir, exist_ok=True)
                    
                    # 将修改后的数据写入新文件
                    with open(new_pkl_path, 'wb') as f_out:
                        pickle.dump(tokens_data, f_out)
                    
                    # 更新计数器
                    local_counts[f'{scene_type}_modified'] += 1
                else:
                    local_counts['path_format_error'] += 1
            else:
                local_counts['unmatched_prefix'] += 1

        except FileNotFoundError:
            local_counts['file_not_found'] += 1
        except Exception:
            local_counts['other_errors'] += 1
            
    return local_counts


def main():
    """主执行函数"""
    try:
        # 1. 读取原始JSON文件
        print("开始加载JSON文件...")
        start_time = time.time()
        with open(input_file_path, 'r', encoding='utf-8') as f:
            condition_data = json.load(f)
        with open(input_file_path_2, 'r', encoding='utf-8') as f:
            token_records = json.load(f)
        load_time = time.time() - start_time
        print(f"文件加载完毕，耗时: {load_time:.2f} 秒。")

        # 基本检查
        if not (isinstance(condition_data, list) and isinstance(token_records, list)):
            raise TypeError("输入文件中的数据类型不正确，期望是列表。")
        if len(condition_data) != len(token_records):
            raise ValueError("两个JSON文件中的记录数量不一致，无法处理。")
        
        total_items = len(condition_data)
        print(f"共计 {total_items} 条记录待处理，使用 {NUM_PROCESSES} 个进程。")

        # 2. 将数据组合并分割成块
        # 使用zip将两个列表的对应项配对
        all_tasks = list(zip(condition_data, token_records))
        chunk_size = math.ceil(total_items / NUM_PROCESSES)
        task_chunks = [all_tasks[i : i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        print(f"数据已分割成 {len(task_chunks)} 个任务块。")

        # 3. 创建进程池并并行处理
        print("开始并行处理文件...")
        start_processing_time = time.time()
        
        total_counts = Counter()
        with Pool(processes=NUM_PROCESSES) as pool:
            # 使用 imap_unordered 可以更快地得到结果，因为不关心顺序
            result_iterator = pool.imap_unordered(process_chunk_and_modify, task_chunks)
            
            for local_counts in tqdm(result_iterator, total=len(task_chunks), desc="Processing Chunks"):
                total_counts.update(local_counts)

        processing_time = time.time() - start_processing_time
        print(f"\n并行处理完成，耗时: {processing_time:.2f} 秒。")

        # 4. 打印最终统计结果
        print("\n--- 处理完成，最终统计结果 ---")
        blocks_modified = total_counts['blocks_modified']
        strawberries_modified = total_counts['strawberries_modified']
        
        print(f"为 'blocks' 场景修改的文件数: {blocks_modified}")
        print(f"为 'strawberries' 场景修改的文件数: {strawberries_modified}")
        print(f"总计修改的文件数: {blocks_modified + strawberries_modified}")
        print("-" * 30)
        print(f"因前缀不匹配而跳过的文件数: {total_counts['unmatched_prefix']}")
        print(f"因文件未找到而跳过的文件数: {total_counts['file_not_found']}")
        print(f"因路径格式错误无法保存的文件数: {total_counts['path_format_error']}")
        print(f"发生其他错误的文件数: {total_counts['other_errors']}")
        
        print(f"\n脚本总运行时间: {time.time() - start_time:.2f} 秒。")

    except Exception as e:
        print(f"发生致命错误: {e}")

if __name__ == "__main__":
    main()
