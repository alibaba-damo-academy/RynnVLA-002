import os
import argparse  # 导入 argparse 模块
from multiprocessing import Process

def run_script(rank, all_ranks, resolution, in_filename_path, out_dir, tokenizer_path): # 添加 task 参数
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)
    print(f"Starting running on {rank}.")

    # 使用 f-string 动态构建文件名和目录名
    # 将 'spatial' 替换为传入的 task 变量
    os.system(f"python -u pre_tokenize_action_local.py "
               f"--splits={all_ranks} "
               f"--rank={rank} "
               f"--in_filename {in_filename_path} "
               f"--out_dir {out_dir} "
               f"--tokenizer {tokenizer_path} "
               f"--target_size {resolution}")


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run parallel data processing scripts with a customizable spatial task.")

    # 2. 添加命令行参数
    parser.add_argument('--task', type=str, required=True,
                        help="dataset name (e.g., 'spatial', 'object', 'goal', '10').")
    parser.add_argument('--resolution', type=int, required=True,
                        help="resolution (e.g., 256, 512).")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="tokenizer path for Alpha-VLLM/Lumina-mGPT-7B-768")
    parser.add_argument(
        '--his', '-H', type=int, default=1,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    # parser.add_argument(
    #     '--img_name', type=str, choices=['imgs_wrist', 'imgs_third_view'], required=True,
    #     help='List of image names to include (imgs_wrist and/or imgs_third_view)')
    parser.add_argument(
        '--img_name', nargs='+', default=['imgs_third_view'], choices=['imgs_wrist', 'imgs_third_view'],
        help='List of image names to include (imgs_wrist and/or imgs_third_view)')

    # 3. 解析命令行参数
    args = parser.parse_args()

    data_type = ['val_ind', 'val_ood', 'train']

    # in_filename_dir = '/mnt/PLNAS/cenjun/libero/processed_data/convs'
    # out_root = '/mnt/PLNAS/cenjun/libero/processed_data/tokens'
    in_filename_dir = '../processed_data/convs'
    out_root = '../processed_data/tokens'

    if len(args.img_name) == 1:
        img_item = args.img_name[0]
    else:
        img_item = '_'.join([item.replace('imgs_', '') for item in args.img_name])

    for data_t in data_type:

        in_filename_path = os.path.join(in_filename_dir, f'libero_{args.task}_his_{args.his}_{data_t}_{img_item}_a2i_{args.resolution}.json')
        out_dir = os.path.join(out_root, f'libero_{args.task}_his_{args.his}_{data_t}_{img_item}_a2i_{args.resolution}')

        processes = []
        all_ranks = 32
        for i in range(all_ranks):
            # 将解析到的 task 传递给 run_script 函数
            p = Process(target=run_script, args=(i, all_ranks, args.resolution, in_filename_path, out_dir, args.tokenizer_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
 

                              
                              
