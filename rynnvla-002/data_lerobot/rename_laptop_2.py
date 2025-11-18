import json
import os
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math

# --- 用户配置区域 ---

# 包含筛选条件的主JSON文件
input_file_path = "/mnt/PLNAS/cenjun/all_data/extracted/convs/libero_goal_his_1_train_img_state_rel_ck_1_512.json"
# 包含pickle文件路径、需要被修改的JSON文件
json_to_modify_path = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record_unique.json"

# --- 自动生成输出文件名 ---
base, ext = os.path.splitext(json_to_modify_path)
output_json_path = f"{base}_modified_mp{ext}" # mp for multiprocessing

# --- 多进程配置 ---
# 使用所有可用的CPU核心，也可以手动设置为一个固定值，如 8
NUM_PROCESSES = 64


def process_chunk(chunk_data):
    """
    工作函数，由每个子进程执行。
    它接收一个数据块，并返回处理后的数据块。
    """
    condition_chunk, data_to_modify_chunk = chunk_data
    
    # 用于存放此数据块处理后的结果
    processed_chunk = []
    
    # 使用zip同时遍历两个列表的对应项
    for condition_item, item_to_modify in zip(condition_chunk, data_to_modify_chunk):
        # 使用与原始脚本完全相同的逻辑
        conversation_value = condition_item.get('conversations', [{}])[0].get('value')
        
        # 默认情况下，路径不修改
        path_modified = False
        
        if conversation_value == 'What action should the robot take to laptop 03?<|state|><|image|><|image|>':
            image_path = condition_item.get('image', [None])[0]
            if image_path and ('blocks' in image_path or 'strawberries' in image_path):
                original_pkl_path = item_to_modify['file']
                if '/files/' in original_pkl_path:
                    new_pkl_path = original_pkl_path.replace('/files/', '/new_files/', 1)
                    item_to_modify['file'] = new_pkl_path
                    path_modified = True

        processed_chunk.append(item_to_modify)
        
    return processed_chunk


def main():
    """主执行函数"""
    try:
        # 1. 读取原始JSON文件（这一步仍然是单线程的）
        print("开始加载JSON文件，这可能需要一些时间...")
        start_time = time.time()
        with open(input_file_path, 'r', encoding='utf-8') as f:
            condition_data = json.load(f)
        with open(json_to_modify_path, 'r', encoding='utf-8') as f:
            data_to_modify = json.load(f)
        load_time = time.time() - start_time
        print(f"文件加载完毕，耗时: {load_time:.2f} 秒。")

        if not (isinstance(condition_data, list) and isinstance(data_to_modify, list)):
            raise TypeError("输入文件中的数据类型不正确，期望是列表。")

        if len(condition_data) != len(data_to_modify):
            raise ValueError("两个JSON文件中的记录数量不一致，无法处理。")
        
        total_items = len(condition_data)
        print(f"共计 {total_items} 条记录待处理，使用 {NUM_PROCESSES} 个进程。")

        # 2. 将数据分割成块
        chunk_size = math.ceil(total_items / NUM_PROCESSES)
        # 创建一个任务列表，每个任务包含两个数据块
        tasks = [
            (condition_data[i : i + chunk_size], data_to_modify[i : i + chunk_size])
            for i in range(0, total_items, chunk_size)
        ]
        
        print(f"数据已分割成 {len(tasks)} 个任务块。")

        # 3. 创建进程池并并行处理
        print("开始并行处理...")
        start_processing_time = time.time()
        
        final_results = []
        with Pool(processes=NUM_PROCESSES) as pool:
            # 使用 tqdm 显示处理进度
            # pool.imap 会按顺序返回结果，适合显示进度
            result_iterator = pool.imap(process_chunk, tasks, chunksize=1)
            
            for chunk_result in tqdm(result_iterator, total=len(tasks), desc="Processing Chunks"):
                final_results.extend(chunk_result)

        processing_time = time.time() - start_processing_time
        print(f"\n并行处理完成，耗时: {processing_time:.2f} 秒。")

        # 4. 保存最终结果
        print(f"正在将 {len(final_results)} 条结果保存到: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            # 对于非常大的文件，不建议使用indent=4，因为它会增加文件大小和写入时间
            # 如果需要可读性，可以保留 indent=4
            json.dump(final_results, f_out)
        
        print("保存成功！")
        total_time = time.time() - start_time
        print(f"脚本总运行时间: {total_time:.2f} 秒。")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    # 这行是使用 multiprocessing 的关键
    # 它确保子进程不会重新导入和执行主模块的代码
    main()
