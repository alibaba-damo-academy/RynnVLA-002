import os
import json

with open("modified_data_final.json", "r") as f:
    data = json.load(f)

def check_files_existence(task_data):
    missing_files = []

    for task_name, task_info in task_data.items():
        print(f"Checking task: {task_name}")
        file_paths = task_info["data_path"]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    return missing_files

if __name__ == "__main__":
    missing = check_files_existence(data["task_data"])

    if missing:
        print("\n❌ Missing files:")
        for f in missing:
            print(f)
        print(f"\nTotal missing files: {len(missing)}")
    else:
        print("✅ All files exist!")
