import json
import os
import pickle
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
from tqdm import tqdm
import math

# --- 用户配置区域 ---

# 包含已修改pkl路径的JSON文件
# 请确保这个路径指向你上一个脚本生成的最终JSON文件
# input_json_path = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record_unique_modified_mp.json"
input_json_path = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_abs_state_256_minmax/tokens/concate_tokens/libero_front_wrist_wm_action_data_3_tasks_his_1_train_img_state_ck_1_512.json"

# --- Token 前缀定义 ---
original_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 29413, 16604, 16399, 16402, 16414]
blocks_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 27795, 25426, 16414]
strawberries_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 27795, 32723, 40410, 16414]
# 新增的 pen 前缀
pen_prefix = [0, 18105, 18865, 17208, 16647, 29591, 17391, 16671, 27795, 20145, 16414]


# 为了方便在函数中引用，将它们放入一个字典
PREFIXES = {
    "original": original_prefix,
    "blocks": blocks_prefix,
    "strawberries": strawberries_prefix,
    "pen": pen_prefix  # 将新前缀加入字典
}
# 计算每个前缀的长度，避免在循环中重复计算
PREFIX_LENS = {name: len(p) for name, p in PREFIXES.items()}


# --- 多进程配置 ---
# 您可以根据机器的CPU核心数调整
NUM_PROCESSES = 128


def check_prefix_worker(pkl_paths_chunk):
    """
    工作函数，由每个子进程执行。
    它接收一个文件路径列表，并返回一个包含统计结果的 Counter 对象。
    新增功能: 检查 'token' 和 'label' 的长度是否一致。
    """
    # 使用 Counter 对象来方便地进行计数
    local_counts = Counter()
    
    for pkl_path in pkl_paths_chunk:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # --- 新增: 检查 token 和 label 的存在和类型 ---
            tokens = data.get('token')
            labels = data.get('label')
            
            # 必须同时是列表才能进行后续比较
            if not isinstance(tokens, list) or not isinstance(labels, list):
                if not isinstance(tokens, list):
                    local_counts['error_no_token_list'] += 1
                if not isinstance(labels, list):
                    local_counts['error_no_label_list'] += 1
                continue # 跳过此文件

            # --- 新增: 比较 token 和 label 的长度 ---
            if len(tokens) == len(labels):
                local_counts['length_match'] += 1
            else:
                local_counts['length_mismatch'] += 1
            
            # --- 原有逻辑: 检查前缀 ---
            matched = False
            # 遍历所有预定义的前缀进行检查
            for name, prefix in PREFIXES.items():
                prefix_len = PREFIX_LENS[name]
                # 检查 token 列表长度是否足够
                if len(tokens) >= prefix_len and tokens[:prefix_len] == prefix:
                    local_counts[name] += 1
                    matched = True
                    break # 找到匹配项后，无需再检查其他前缀
            
            if not matched:
                local_counts['other'] += 1

        except FileNotFoundError:
            local_counts['file_not_found'] += 1
        except Exception as e:
            # 捕获其他可能的错误，如pickle解码错误等
            local_counts['other_errors'] += 1
            
    return local_counts


def main():
    """主执行函数"""
    try:
        # 1. 读取包含文件路径的JSON文件
        print(f"正在加载JSON文件: {input_json_path} ...")
        start_time = time.time()
        with open(input_json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # 提取所有文件路径
        pkl_paths = [record['file'] for record in records if 'file' in record]
        total_files = len(pkl_paths)
        load_time = time.time() - start_time
        print(f"文件加载完毕，共找到 {total_files} 个文件路径，耗时: {load_time:.2f} 秒。")

        if total_files == 0:
            print("未找到任何文件路径，脚本退出。")
            return

        # 2. 将文件路径列表分割成块
        # 确保 NUM_PROCESSES 不超过任务数
        num_processes_to_use = min(NUM_PROCESSES, total_files)
        chunk_size = math.ceil(total_files / num_processes_to_use)
        tasks = [pkl_paths[i : i + chunk_size] for i in range(0, total_files, chunk_size)]
        print(f"文件路径已分割成 {len(tasks)} 个任务块，使用 {num_processes_to_use} 个进程。")

        # 3. 创建进程池并并行处理
        print("开始并行统计文件...")
        start_processing_time = time.time()
        
        # 创建一个总的 Counter 来汇总所有结果
        total_counts = Counter()
        
        with Pool(processes=num_processes_to_use) as pool:
            result_iterator = pool.imap_unordered(check_prefix_worker, tasks)
            
            # 使用 tqdm 显示处理进度
            for local_counts in tqdm(result_iterator, total=len(tasks), desc="Verifying Files"):
                total_counts.update(local_counts)

        processing_time = time.time() - start_processing_time
        print(f"\n并行统计完成，耗时: {processing_time:.2f} 秒。")

        # 4. 打印最终统计报告
        print("\n" + "="*30)
        print("--- 最终统计报告 ---")
        print("="*30)
        print(f"总计检查文件数: {total_files}")
        
        # --- 前缀统计 ---
        print("\n--- 前缀匹配统计 ---")
        original_count = total_counts['original']
        blocks_count = total_counts['blocks']
        strawberries_count = total_counts['strawberries']
        pen_count = total_counts['pen'] # 获取 pen 的计数
        other_count = total_counts['other']
        verified_count = original_count + blocks_count + strawberries_count + pen_count + other_count
        
        # 使用 f-string 对齐，使报告更美观
        print(f"匹配 {'original':<12} 前缀的数量: {original_count}")
        print(f"匹配 {'blocks':<12} 前缀的数量: {blocks_count}")
        print(f"匹配 {'strawberries':<12} 前缀的数量: {strawberries_count}")
        print(f"匹配 {'pen':<12} 前缀的数量: {pen_count}")
        print(f"不匹配任何已知前缀的数量: {other_count}")
        print("-" * 20)
        print(f"成功验证前缀的文件总数: {verified_count}")
        
        # --- 新增: 长度统计 ---
        print("\n--- Token/Label 长度统计 ---")
        length_match_count = total_counts['length_match']
        length_mismatch_count = total_counts['length_mismatch']
        total_length_checked = length_match_count + length_mismatch_count
        
        print(f"Token 和 Label 长度相同的数量: {length_match_count}")
        print(f"Token 和 Label 长度不同的数量: {length_mismatch_count}")
        print("-" * 20)
        print(f"成功进行长度比较的文件总数: {total_length_checked}")
        if total_length_checked > 0:
            match_percentage = (length_match_count / total_length_checked) * 100
            print(f"长度匹配率: {match_percentage:.2f}%")

        # --- 错误统计 ---
        file_not_found_count = total_counts['file_not_found']
        error_no_token_list_count = total_counts['error_no_token_list']
        error_no_label_list_count = total_counts['error_no_label_list'] # 新增
        other_errors_count = total_counts['other_errors']
        total_errors = file_not_found_count + error_no_token_list_count + error_no_label_list_count + other_errors_count

        if total_errors > 0:
            print("\n--- 错误统计 ---")
            if file_not_found_count > 0:
                print(f"文件未找到的数量: {file_not_found_count}")
            if error_no_token_list_count > 0:
                print(f"缺少'token'键或非列表的数量: {error_no_token_list_count}")
            if error_no_label_list_count > 0:
                print(f"缺少'label'键或非列表的数量: {error_no_label_list_count}")
            if other_errors_count > 0:
                print(f"其他读取/解码错误数量: {other_errors_count}")

        print("\n" + "="*30)
        print(f"脚本总运行时间: {time.time() - start_time:.2f} 秒。")


    except FileNotFoundError:
        print(f"错误: JSON文件未找到，请检查路径: {input_json_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    main()
