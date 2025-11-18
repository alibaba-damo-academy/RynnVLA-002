import json
import os
import pickle
import time
import re
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm
import math

# --- 用户配置区域 ---

# 请将这个路径替换为您文件的确切路径
input_file_path = "/mnt/PLNAS/cenjun/all_data/extracted/convs/libero_goal_his_1_train_img_state_rel_ck_1_512.json"
input_file_path_2 = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record_unique_modified_mp.json"
print(f"输入文件1: {input_file_path}")
print(f"输入文件2: {input_file_path_2}")

# --- 新增配置 ---
# 新token的计算参数
PROGRESS_TOKEN_BASE = 16100
PROGRESS_TOKEN_SCALE = 256
# 插入位置的标记token
INSERT_ANCHOR_TOKEN = 10004
# 新label的值 (用于插入到labels列表中)
NEW_LABEL_VALUE = -100

# 新的输出目录片段
NEW_DIR_SEGMENT = '/new_files_task_3/'
OLD_DIR_SEGMENT_1 = '/files/'
OLD_DIR_SEGMENT_2 = '/new_files/'


# --- 多进程配置 ---
# 您可以根据您的机器配置调整进程数
NUM_PROCESSES = 64


def process_chunk_and_add_progress_token(task_chunk):
    """
    工作函数，由每个子进程执行。
    它接收一个任务块，处理文件，计算并插入进度token，然后返回统计结果。
    """
    local_counts = Counter()
    # 为每个进程创建一个独立的缓存，避免多进程间共享缓存的复杂性
    dir_file_count_cache = {}

    for condition_item, token_record in task_chunk:
        try:
            # --- 1. 获取路径并检查目标文件是否已存在 (加速重复运行) ---
            original_pkl_path = token_record.get('file')
            if not original_pkl_path:
                local_counts['pkl_path_missing'] += 1
                continue

            # 生成新文件路径
            new_pkl_path = None
            if OLD_DIR_SEGMENT_1 in original_pkl_path:
                new_pkl_path = original_pkl_path.replace(OLD_DIR_SEGMENT_1, NEW_DIR_SEGMENT, 1)
            elif OLD_DIR_SEGMENT_2 in original_pkl_path:
                new_pkl_path = original_pkl_path.replace(OLD_DIR_SEGMENT_2, NEW_DIR_SEGMENT, 1)
            
            if not new_pkl_path:
                local_counts['path_replace_error'] += 1
                continue

            # **核心加速逻辑：如果目标文件已存在，则跳过**
            if os.path.exists(new_pkl_path):
                local_counts['skipped_exists'] += 1
                continue

            # --- 2. 获取和验证其他输入信息 ---
            image_path = condition_item.get('image', [None])[0]
            if not image_path:
                local_counts['image_path_missing'] += 1
                continue

            # --- 3. 计算新的进度token ---
            image_dir = os.path.dirname(image_path)
            image_basename = os.path.basename(image_path)

            match = re.search(r'image_(\d+)\.png$', image_basename)
            if not match:
                local_counts['image_name_parse_error'] += 1
                continue
            
            image_num = int(match.group(1))

            # 使用缓存获取目录中的文件数
            if image_dir in dir_file_count_cache:
                total_files = dir_file_count_cache[image_dir]
            else:
                try:
                    # 仅统计.png文件以提高准确性
                    total_files = len([name for name in os.listdir(image_dir) if name.endswith('.png')])
                    dir_file_count_cache[image_dir] = total_files
                except FileNotFoundError:
                    local_counts['image_dir_not_found'] += 1
                    continue
            
            if total_files > 1:
                progress_percent = image_num / (total_files - 1)
            else:
                progress_percent = 1.0

            progress_percent = max(0.0, min(1.0, progress_percent))
            new_progress_token = PROGRESS_TOKEN_BASE + round(progress_percent * (PROGRESS_TOKEN_SCALE - 1))

            # --- 4. 文件读写和修改逻辑 ---
            with open(original_pkl_path, 'rb') as f_in:
                tokens_data = pickle.load(f_in)
            
            # 使用副本进行操作，避免意外修改原始加载的数据
            modified_tokens = tokens_data['token'][:]
            modified_labels = tokens_data['label'][:]

            try:
                insert_index = modified_tokens.index(INSERT_ANCHOR_TOKEN)
            except ValueError:
                local_counts['anchor_token_not_found'] += 1
                continue

            # 插入新的token和label
            modified_tokens.insert(insert_index, new_progress_token)
            modified_labels.insert(insert_index, new_progress_token) # <--- 已修正: 插入固定的 NEW_LABEL_VALUE
            
            tokens_data['token'] = modified_tokens
            tokens_data['label'] = modified_labels
            
            assert len(tokens_data['token']) == len(tokens_data['label'])

            # 创建新目录并写入文件
            new_dir = os.path.dirname(new_pkl_path)
            os.makedirs(new_dir, exist_ok=True)
            
            with open(new_pkl_path, 'wb') as f_out:
                pickle.dump(tokens_data, f_out)
            
            local_counts['modified_successfully'] += 1

        except FileNotFoundError:
            local_counts['pkl_file_not_found'] += 1
        except Exception:
            # 捕获其他未知错误，方便调试，并防止进程中断
            local_counts['other_errors'] += 1
            # print(f"发生未知错误: {e} on file {original_pkl_path}") # 可选：取消注释用于调试
            
    return local_counts


def main():
    """主执行函数"""
    script_start_time = time.time()
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
        print(f"修改后的文件将保存到包含 '{NEW_DIR_SEGMENT}' 的新路径中。")
        print("如果目标文件已存在，将自动跳过。")

        # 2. 将数据组合并分割成块
        all_tasks = list(zip(condition_data, token_records))
        chunk_size = math.ceil(total_items / NUM_PROCESSES)
        task_chunks = [all_tasks[i : i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        print(f"数据已分割成 {len(task_chunks)} 个任务块。")

        # 3. 创建进程池并并行处理
        print("开始并行处理文件...")
        start_processing_time = time.time()
        
        total_counts = Counter()
        with Pool(processes=NUM_PROCESSES) as pool:
            result_iterator = pool.imap_unordered(process_chunk_and_add_progress_token, task_chunks)
            
            for local_counts in tqdm(result_iterator, total=len(task_chunks), desc="Processing Chunks"):
                total_counts.update(local_counts)

        processing_time = time.time() - start_processing_time
        print(f"\n并行处理完成，耗时: {processing_time:.2f} 秒。")

        # 4. 打印最终统计结果
        print("\n--- 处理完成，最终统计结果 ---")
        modified_count = total_counts['modified_successfully']
        skipped_count = total_counts['skipped_exists']
        print(f"成功修改并保存的文件数: {modified_count}")
        print(f"跳过 (因目标文件已存在): {skipped_count}")
        print("-" * 30)
        print("失败或未处理的条目统计:")
        print(f"  - 图像路径缺失: {total_counts['image_path_missing']}")
        print(f"  - Pickle文件路径缺失: {total_counts['pkl_path_missing']}")
        print(f"  - 无法生成新路径 (格式错误): {total_counts['path_replace_error']}")
        print(f"  - 无法解析图像文件名: {total_counts['image_name_parse_error']}")
        print(f"  - 图像目录未找到: {total_counts['image_dir_not_found']}")
        print(f"  - 原始Pickle文件未找到: {total_counts['pkl_file_not_found']}")
        print(f"  - 未找到锚点Token ({INSERT_ANCHOR_TOKEN}): {total_counts['anchor_token_not_found']}")
        print(f"  - 其他未知错误: {total_counts['other_errors']}")
        
        print(f"\n脚本总运行时间: {time.time() - script_start_time:.2f} 秒。")

    except Exception as e:
        print(f"\n发生致命错误: {e}")

if __name__ == "__main__":
    main()