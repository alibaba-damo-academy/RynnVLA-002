import os
import pickle
import json
import multiprocessing
from tqdm import tqdm

def process_file(file_path):
    """
    处理单个PKL文件并返回其元数据。
    这是一个顶层函数，以便多进程可以调用。

    Args:
        file_path (str): 单个 .pkl 文件的完整路径。

    Returns:
        dict or None: 如果成功，返回包含文件元数据的字典；否则返回 None。
    """
    try:
        # 使用二进制读取模式打开文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 检查'token'键是否存在并且其值是列表
        if 'token' in data and isinstance(data['token'], list):
            token_length = len(data['token'])
        else:
            # 在多进程中，避免频繁打印，只在必要时返回None
            # print(f"\n警告：文件 '{file_path}' 中缺少 'token' 键或其值不是列表。")
            token_length = 0

        # 注意：这里不分配ID，ID将在主进程中最后统一分配
        metadata = {
            "file": os.path.abspath(file_path),
            "len": token_length
        }
        return metadata

    except pickle.UnpicklingError:
        print(f"\n错误：无法反序列化文件 '{file_path}'。可能已损坏。将跳过此文件。")
        return None
    except Exception as e:
        print(f"\n处理文件 '{file_path}' 时发生未知错误: {e}。将跳过此文件。")
        return None

def generate_metadata_from_pkl_folder_mp(folder_path, output_file, num_processes=None):
    """
    使用多进程读取文件夹下所有pkl文件，提取元数据，并生成一个JSON文件。

    Args:
        folder_path (str): 包含 .pkl 文件的文件夹路径。
        output_file (str): 输出的 JSON 文件名。
        num_processes (int, optional): 使用的进程数。默认为CPU核心数。
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 1. 获取所有pkl文件并进行自然排序（例如 1.pkl, 2.pkl, 10.pkl）
    try:
        def get_numeric_part(filename):
            try:
                return int(os.path.splitext(filename)[0])
            except (ValueError, IndexError):
                return float('inf')

        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        pkl_files.sort(key=get_numeric_part)
        print(f"在 '{folder_path}' 中找到 {len(pkl_files)} 个 .pkl 文件。")

        # 构建完整的文件路径列表
        pkl_file_paths = [os.path.join(folder_path, f) for f in pkl_files]
        
    except Exception as e:
        print(f"读取文件夹 '{folder_path}' 时出错: {e}")
        return
        
    if not pkl_file_paths:
        print("未找到任何 .pkl 文件。")
        return

    # 2. 使用多进程池处理文件
    # with ... as pool 语法可以确保进程池在使用后被正确关闭
    with multiprocessing.Pool(processes=num_processes) as pool:
        print(f"启动 {pool._processes} 个工作进程...")
        
        # 使用 pool.imap 以便与 tqdm 结合显示进度
        # imap 会按顺序返回结果，这对于我们后续分配ID非常重要
        results = list(tqdm(
            pool.imap(process_file, pkl_file_paths), 
            total=len(pkl_file_paths), 
            desc="正在处理PKL文件"
        ))

    # 3. 收集结果并分配ID
    # 过滤掉处理失败的文件（返回None的结果）
    valid_results = [r for r in results if r is not None]

    # 按原始顺序为有效结果分配ID
    for i, metadata in enumerate(valid_results):
        metadata['id'] = i

    # 4. 将所有字典的列表写入JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_results, f, indent=4, ensure_ascii=False)
        
        print(f"\n处理完成！共成功处理 {len(valid_results)} 个文件。")
        print(f"结果已保存至: {os.path.relpath(output_file)}")

    except Exception as e:
        print(f"\n写入JSON文件 '{output_file}' 时出错: {e}")


if __name__ == '__main__':
    # --- 请修改以下路径 ---

    # 1. 设置包含 .pkl 文件的文件夹路径
    input_folder = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/new_files_task_3"

    # 2. 设置输出的JSON文件名
    output_json_file = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record_task_2.json"

    # 3. (可选) 设置要使用的CPU进程数。如果设为 None，则默认为系统的CPU核心数。
    #    例如，在8核CPU上，num_cpu = 8。对于I/O密集型任务，可以设置得更高。
    #    对于CPU密集型任务，通常设置为CPU核心数。
    num_cpu_to_use = 64

    # --- 修改结束 ---

    # 调用多进程主函数
    generate_metadata_from_pkl_folder_mp(input_folder, output_json_file, num_processes=num_cpu_to_use)
