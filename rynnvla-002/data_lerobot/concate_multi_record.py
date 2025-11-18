import json
from pathlib import Path
import argparse

def read_json_file(file_path: Path):
    """
    Reads a JSON file and returns its content.
    Handles file not found, JSON decoding errors, and other exceptions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

def write_json_file(data, file_path: Path):
    """
    Writes data to a JSON file, creating parent directories if necessary.
    """
    # 确保输出目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 使用 indent=2 或 4 以便格式化输出，更易读
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote concatenated data to {file_path}")
    except Exception as e:
        print(f"Error: Could not write data to {file_path}: {e}")

def concatenate_records(input_file_paths: list[Path]):
    """
    Concatenates records from a list of JSON files.
    Assumes the content of each JSON file is a list of records.
    """
    all_records = []
    for file_path in input_file_paths:
        content = read_json_file(file_path)
        if content is not None:
            if isinstance(content, list):
                all_records.extend(content)
            else:
                # 如果JSON文件内容不是列表，发出警告并将其作为一个整体添加
                print(f"Warning: Content of {file_path} is not a list. Appending it as a single item.")
                all_records.append(content)
    return all_records

def main(args):
    """
    Main function to handle the concatenation process based on command-line arguments.
    """
    # 将输入的字符串路径转换为 Path 对象
    input_paths = [Path(p) for p in args.input_files]
    output_path = Path(args.output_file)

    print("--- Starting Concatenation ---")
    print("Input files:")
    for path in input_paths:
        print(f"  - {path}")
    print(f"Output file: {output_path}\n")

    # 执行合并操作
    concatenated_data = concatenate_records(input_paths)

    # 如果成功合并了数据，则写入文件
    if concatenated_data:
        write_json_file(concatenated_data, output_path)
    else:
        print("No valid data found to concatenate. Output file was not created.")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="Concatenate multiple JSON record files into a single file."
    )

    # 定义输入文件参数
    # nargs='+' 表示可以接受一个或多个输入文件
    parser.add_argument(
        '--input_files',
        type=str,
        nargs='+',  # 关键改动：允许多个值
        required=True,
        help="A space-separated list of input record.json files to concatenate."
    )

    # 定义输出文件参数
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help="The path for the new concatenated JSON file."
    )

    args = parser.parse_args()
    main(args)
