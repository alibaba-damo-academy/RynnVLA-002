import os
import glob
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures

# --- 可配置参数 ---
# 根据你的CPU核心数和I/O能力调整。对于网络文件系统(NAS/NFS)上的I/O密集型任务，
# 设置比CPU核心数更多的线程通常是有效的。可以从 16 或 32 开始尝试。
MAX_WORKERS = 16

def find_all_action_npy_files_fast(root_directories):
    """
    使用 glob 模式快速查找文件，基于已知的固定目录结构。
    (此函数保持不变)
    """
    npy_file_paths = []
    print("开始使用 glob 模式快速扫描文件...")
    glob_pattern = os.path.join('*', '*', '*', '*', 'action', '*.npy')

    for root_dir in root_directories:
        if not os.path.isdir(root_dir):
            print(f"警告: 目录 '{root_dir}' 不存在，已跳过。", file=sys.stderr)
            continue
        
        search_path = os.path.join(root_dir, glob_pattern)
        print(f"正在使用模式进行搜索: {search_path}")
        
        matched_files = glob.glob(search_path)
        npy_file_paths.extend(matched_files)

    print(f"扫描完成，共找到 {len(npy_file_paths)} 个 .npy 文件。")
    return npy_file_paths

def load_and_validate_action(file_path):
    """
    由单个线程执行的工作函数：加载一个 .npy 文件并验证其形状。
    返回 action 数据或 None（如果失败）。
    """
    try:
        action_data = np.load(file_path)
        if action_data.shape == (6,):
            return action_data
        else:
            # 将警告和错误打印到 stderr，以免干扰 tqdm 进度条
            print(f"\n警告: 文件 '{file_path}' 的形状为 {action_data.shape}，期望为 (6,)。已跳过。", file=sys.stderr)
            return None
    except Exception as e:
        print(f"\n错误: 无法加载或处理文件 '{file_path}'。错误信息: {e}", file=sys.stderr)
        return None

def analyze_action_data_multithreaded(file_paths):
    """
    使用多线程加载所有 .npy 文件，然后计算统计数据并打印结果。
    """
    if not file_paths:
        print("未找到任何 .npy 文件进行分析。")
        return

    all_actions = []
    print(f"正在使用最多 {MAX_WORKERS} 个线程并行加载和处理 action 数据...")

    # 使用 ThreadPoolExecutor 来并行处理文件加载
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map 会将 file_paths 中的每个元素传给 load_and_validate_action 函数
        # 它会立即返回一个迭代器，结果在可用时产生
        # 我们用 tqdm 包裹这个迭代器来显示进度条
        results_iterator = executor.map(load_and_validate_action, file_paths)
        
        # 从迭代器中收集所有非 None 的结果
        all_actions = [result for result in tqdm(results_iterator, total=len(file_paths), desc="加载 .npy 文件", unit="file") if result is not None]

    if not all_actions:
        print("没有成功加载任何有效的数据。")
        return

    # 将所有 action 列表转换为一个大的 (N, 6) NumPy 数组
    print(f"\n数据加载完成。正在堆叠数组...")
    stacked_actions = np.array(all_actions)
    
    print(f"成功加载并堆叠了 {stacked_actions.shape[0]} 个 action，形成数组，形状为: {stacked_actions.shape}")
    print("正在计算统计数据...")

    # 计算统计数据
    min_vals = np.min(stacked_actions, axis=0)
    max_vals = np.max(stacked_actions, axis=0)
    q01_vals = np.percentile(stacked_actions, 1, axis=0)
    q99_vals = np.percentile(stacked_actions, 99, axis=0)

    # 打印结果
    print("\n--- Action 数据统计结果 ---")
    print("-" * 85)
    print(f"{'维度':<10} | {'最小值':<20} | {'最大值':<20} | {'1% 分位数 (q01)':<20} | {'99% 分位数 (q99)':<20}")
    print("-" * 85)
    
    for i in range(6):
        print(f"维度 {i:<5} | {min_vals[i]:<20.8f} | {max_vals[i]:<20.8f} | {q01_vals[i]:<20.8f} | {q99_vals[i]:<20.8f}")
    
    print("-" * 85)


def main():
    base_path = "/mnt/PLNAS/cenjun/processed_lerobot_data"
    target_dirs = [
        os.path.join(base_path, "data_0623"),
        os.path.join(base_path, "data_0625")
    ]
    
    all_files = find_all_action_npy_files_fast(target_dirs)
    
    # 调用新的多线程分析函数
    analyze_action_data_multithreaded(all_files)


if __name__ == "__main__":
    main()
