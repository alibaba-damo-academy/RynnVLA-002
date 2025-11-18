import os
import argparse  # 导入 argparse 模块
from multiprocessing import Process

def run_script(rank, all_ranks, data_t, task, resolution): # 添加 task 参数
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)
    print(f"Starting running on {rank}.")

    # 使用 f-string 动态构建文件名和目录名
    # 将 'spatial' 替换为传入的 task 变量
    os.system(f"python -u pre_tokenize_action_local.py "
               f"--splits={all_ranks} "
               f"--rank={rank} "
               f"--in_filename ../processed_data/convs/libero_{task}_his_2_{data_t}_img_only_ck_5_{resolution}.json "
               f"--out_dir ../processed_data/libero_{task}_his_2_{data_t}_img_only_ck_5_{resolution} "
               f"--target_size {resolution}")

    os.system(f"python -u pre_tokenize_action_local.py "
               f"--splits={all_ranks} "
               f"--rank={rank} "
               f"--in_filename ../processed_data/convs/libero_{task}_his_1_{data_t}_a2i_{resolution}.json "
               f"--out_dir ../processed_data/libero_{task}_his_1_{data_t}_a2i_{resolution} "
               f"--target_size {resolution}")


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run parallel data processing scripts with a customizable spatial task.")

    # 2. 添加命令行参数
    parser.add_argument('--task', type=str, required=True,
                        help="dataset name (e.g., 'spatial', 'object', 'goal', '10').")
    parser.add_argument('--resolution', type=int, required=True,
                        help="resolution (e.g., 256, 512).")

    # 3. 解析命令行参数
    args = parser.parse_args()

    data_type = ['val_ind', 'val_ood', 'train']

    for data_t in data_type:
        processes = []
        all_ranks = 32
        for i in range(all_ranks):
            # 将解析到的 task 传递给 run_script 函数
            p = Process(target=run_script, args=(i, all_ranks, data_t, args.task, args.resolution))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

