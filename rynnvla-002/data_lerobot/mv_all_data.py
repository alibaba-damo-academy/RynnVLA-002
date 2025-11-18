import json

# 读取 JSON 文件
with open("data_final.json", "r") as f:
    data = json.load(f)

old_prefix = "/public/hz_oss"
new_prefix_oss = "oss://damo-xlab-hangzhou"
local_target_dir = "/mnt/PLNAS/cenjun/all_data"
MAX_PARALLEL_JOBS = 50

# 创建 bash 脚本内容
script_lines = ["#!/bin/bash"]

# 定义一个 job_counter 来控制并发数量
script_lines.extend([
    'function run_jobs() {',
    '    while (( $(jobs -r | wc -l) >= {} )); do sleep 1; done'.format(MAX_PARALLEL_JOBS),
    '    "$@" &',
    '}',
])

job_count = 0

for task_name, task_info in data["task_data"].items():
    for file_path in task_info["data_path"]:
        if file_path.startswith(old_prefix):
            oss_url = file_path.replace(old_prefix, new_prefix_oss)
            local_path = file_path.replace(old_prefix, local_target_dir)
            dir_path = "/".join(local_path.split("/")[:-1])
            script_lines.append(f'mkdir -p "{dir_path}"')
            script_lines.append(
                f'run_jobs ossutil cp "{oss_url}" "{local_path}"'
            )
            job_count += 1

# 最后等待所有作业完成
script_lines.append("wait")

# 写入 bash 脚本文件
with open("copy_to_local_parallel_controlled.sh", "w") as f:
    f.write("\n".join(script_lines) + "\n")

print(f"✅ 已生成 copy_to_local_parallel_controlled.sh，包含 {job_count} 个拷贝任务，最多同时运行 {MAX_PARALLEL_JOBS} 个。")
