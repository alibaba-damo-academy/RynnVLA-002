import os
import json

# 原始文件夹路径（包含所有json文件）
input_dir = "/mnt/PLNAS/cenjun/processed_lerobot_data_2/convs"
# 输出文件夹路径
output_dir = "/mnt/PLNAS/cenjun/processed_lerobot_data_2_2/convs"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录下的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取 JSON 文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 假设每个文件是一个 list of dict

        # 遍历 list 中的每个 item，递归替换字符串中的路径
        def replace_path(obj):
            if isinstance(obj, dict):
                return {k: replace_path(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_path(elem) for elem in obj]
            elif isinstance(obj, str):
                return obj.replace("/mnt/PLNAS/cenjun/processed_lerobot_data_2",
                                   "/mnt/PLNAS/cenjun/processed_lerobot_data_2_2")
            else:
                return obj

        new_data = replace_path(data)

        # 写入新的 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"Processed: {filename}")

print("All files have been processed and saved to:", output_dir)
