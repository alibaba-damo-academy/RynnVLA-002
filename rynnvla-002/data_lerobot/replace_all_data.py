import json

# 读取原始 JSON 文件
with open("data_final.json", "r") as f:
    data = json.load(f)

# 定义替换规则
old_prefix = "/public/hz_oss"
new_prefix = "/mnt/PLNAS/cenjun/all_data"

# 遍历所有任务数据中的路径并替换
for task_name, task_info in data["task_data"].items():
    for i in range(len(task_info["data_path"])):
        if isinstance(task_info["data_path"][i], str):
            task_info["data_path"][i] = task_info["data_path"][i].replace(old_prefix, new_prefix)

# 保存修改后的 JSON 到新文件
with open("modified_data_final.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ 已成功替换路径并保存到 modified_data_final.json")
