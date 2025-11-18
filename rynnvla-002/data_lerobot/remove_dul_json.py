import json
import os

# 请将这个路径替换为您文件的确切路径
input_file_path = "/mnt/PLNAS/cenjun/all_data/extracted/processed_lerobot_data_rel_state_256_minmax/tokens/libero_goal_his_1_train_img_state_rel_ck_1_512/record.json"

# --- 自动生成输出文件名 ---
# 将原始文件名 'record.json' 变为 'record_unique.json'
# 这种方式可以避免手动设置输出文件名，更加灵活
base, ext = os.path.splitext(input_file_path)
output_file_path = f"{base}_unique{ext}"

try:
    # 1. 读取原始JSON文件
    print(f"正在读取文件: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 检查数据类型并执行去重
    if isinstance(data, list):
        original_count = len(data)
        print(f"原始列表长度为: {original_count}")
        print(f"原始列表的前2个元素: {data[0:2]}")

        # --- 开始去重逻辑 ---
        unique_data = []
        seen_files = set()

        for item in data:
            # 使用 .get() 方法安全地获取值，即使'file'键不存在也不会报错
            file_value = item.get('file')
            
            # 确保 file_value 存在且之前未出现过
            if file_value and file_value not in seen_files:
                unique_data.append(item)
                seen_files.add(file_value)
        
        unique_count = len(unique_data)
        print(f"去重后的列表长度为: {unique_count}")
        print(f"移除了 {original_count - unique_count} 个重复项。")
        # print(f"去重后列表的前2个元素: {unique_data[0:2]}")

        # 3. 将处理后的数据写入新的JSON文件
        print(f"\n正在将结果保存到新文件: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # json.dump 用于将Python对象写入文件
            # indent=4 让JSON文件格式化，更易于阅读
            # ensure_ascii=False 确保中文字符等能正确写入
            json.dump(unique_data, f, indent=4, ensure_ascii=False)
        
        print("文件保存成功！")

    else:
        # 如果不是列表，告知用户数据的类型
        print(f"文件中的数据不是一个列表，而是一个 {type(data)}，无需处理。")

except FileNotFoundError:
    print(f"错误：文件未找到，请检查路径是否正确: {input_file_path}")
except json.JSONDecodeError:
    print(f"错误：文件内容不是有效的JSON格式。")
except Exception as e:
    print(f"发生了未知错误: {e}")
