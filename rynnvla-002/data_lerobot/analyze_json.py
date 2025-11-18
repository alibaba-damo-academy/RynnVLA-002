import json

# 请将这个路径替换为您文件的确切路径
file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_3/concate_tokens/libero_data_0623_all_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_wrist_train_img_only_ck_20_512.json" #180638
file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_2/concate_tokens/libero_data_0623_all_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_2_train_img_only_ck_20_512.json"
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_3/libero_data_0623_laptop_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_20_512/record.json" #53204
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_3/libero_data_0623_pc_02_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_20_512/record.json" # 56858
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_3/libero_data_0623_pc_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_20_512/record.json" # 70576

# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_2/libero_data_0623_laptop_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_2_train_img_only_ck_20_512/record.json" # 53204
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_2/libero_data_0623_pc_02_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_2_train_img_only_ck_20_512/record.json" # 113716
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_2/libero_data_0623_pc_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_2_train_img_only_ck_20_512/record.json" # 70576

# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data/tokens_3/concate_tokens/libero_wrist_wm_data_0623_all_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_a2i_512.json"

# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/convs/libero_data_0623_pc_02_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_val_ind_img_only_ck_20_512.json"
file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/tokens/libero_data_0623_pc_02_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_1_512/record.json"
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/tokens/libero_data_0623_pc_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_1_512/record.json"
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/tokens/libero_data_0623_laptop_03_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_1_512/record.json"
# file_path = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/convs/libero_data_0623_pc_02_grab_blocks_so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll_his_1_train_img_only_ck_1_512.json"


try:
    # 打开并读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 将JSON文件内容加载为Python对象
        data = json.load(f)
    print(data[0:5])
    print(data[71565:71570])
    # import pdb; pdb.set_trace()
    # 检查加载的数据是否为列表 (list)
    if isinstance(data, list):
        # 如果是列表，打印其长度
        print(f"文件 '{file_path}' 中的列表长度为: {len(data)}")
    else:
        # 如果不是列表，告知用户数据的类型
        print(f"文件中的数据不是一个列表，而是一个 {type(data)}。")

except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径是否正确: {file_path}")
except json.JSONDecodeError:
    print(f"错误：文件内容不是有效的JSON格式。")
except Exception as e:
    print(f"发生了未知错误: {e}")

