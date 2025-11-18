import json
import os

# --- 配置 ---
# 请将 'data.json' 替换为你的实际文件名
json_file_path = '/mnt/PLNAS/cenjun/libero/processed_data/concate_tokens/libero_goal_his_2_third_view_wrist_w_state_5_256_a.json' 
# --- 配置结束 ---

# 使用一个集合(set)来存储不重复的目录路径
unique_directories = set()

try:
    # 打开并读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检查数据是否为列表
    if not isinstance(data, list):
        print(f"错误：JSON文件的顶层不是一个列表。")
    else:
        # 遍历列表中的每一个元素（字典）
        for item in data:
            # 获取 'file' 键对应的值（文件路径）
            file_path = item.get('file')
            
            if file_path and isinstance(file_path, str):
                # 使用os.path.dirname()来安全地获取目录路径
                # 这比手动分割字符串更可靠，能处理各种边缘情况
                directory_path = os.path.dirname(file_path)
                
                # 将提取出的目录路径添加到集合中
                unique_directories.add(directory_path)

        # 打印结果
        count = len(unique_directories)
        print(f"统计完成！")
        print(f"'file' 键中，最后一个'/'之前的不同字符串共有: {count} 种")
        print(unique_directories)

        # 如果你想查看具体是哪些路径，可以取消下面这行代码的注释
        # print("\n具体路径如下:")
        # for path in sorted(list(unique_directories)):
        #     print(path)

except FileNotFoundError:
    print(f"错误：文件 '{json_file_path}' 未找到。")
except json.JSONDecodeError:
    print(f"错误：文件 '{json_file_path}' 不是一个有效的JSON格式。")
except Exception as e:
    print(f"发生了一个未知错误: {e}")

