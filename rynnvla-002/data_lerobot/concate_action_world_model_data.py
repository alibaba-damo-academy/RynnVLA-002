import json
from pathlib import Path
import argparse  # 导入 argparse 模块

def read_json_file(file_path: Path):
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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully wrote concatenated data to {file_path}")
    except Exception as e:
        print(f"Error: Could not write data to {file_path}: {e}")

def concatenate_records(input_file_paths: list[Path]):
    all_records = []
    for file_path in input_file_paths:
        content = read_json_file(file_path)
        if content is not None:
            if isinstance(content, list):
                all_records.extend(content)
            else:
                print(f"Warning: Content of {file_path} is not a list. Appending it as a single item.")
                all_records.append(content)
    return all_records


def main(args):
    BASE_PROCESSED_DATA_DIR = Path("../processed_data")
    CONCAT_OUTPUT_DIR = BASE_PROCESSED_DATA_DIR / "concate_tokens"

    SOURCE_DIR_PATTERNS = [
        f"libero_{args.task}_his_1_{{}}_a2i_{args.resolution}",
        f"libero_{args.task}_his_2_{{}}_img_only_ck_5_{args.resolution}"
    ]
    ALL_PATTERNS = f"libero_{args.task}_his_2_all_img_only_ck_5_a2i_{args.resolution}"
    RECORD_FILENAME = "record.json"

    SPLITS = ["train", "val_ind", "val_ood"]
    CONCAT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_inputs = []

    for split in SPLITS:
        print(f"\n--- Processing split: {split} ---")
        input_paths_for_split = []

        for pattern in SOURCE_DIR_PATTERNS:
            source_subdir_name = pattern.format(split)
            print(source_subdir_name)
            input_path = BASE_PROCESSED_DATA_DIR / source_subdir_name / RECORD_FILENAME
            input_paths_for_split.append(input_path)
            all_inputs.append(input_path)
            print(f"  Looking for input: {input_path}")


        base_output_name_from_pattern = SOURCE_DIR_PATTERNS[1].format(split)

        output_filename_base = base_output_name_from_pattern[:-4] + f'_a2i_{args.resolution}'


        output_filename = f"{output_filename_base}.json"
        output_file_path = CONCAT_OUTPUT_DIR / output_filename

        concatenated_data = concatenate_records(input_paths_for_split)

        if concatenated_data:
            write_json_file(concatenated_data, output_file_path)
        else:
            print(f"No valid data found to concatenate for split {split}. Output file {output_file_path} not created.")
    
    output_filename = f"{ALL_PATTERNS}.json"
    output_file_path = CONCAT_OUTPUT_DIR / output_filename

    concatenated_data = concatenate_records(all_inputs)

    if concatenated_data:
        write_json_file(concatenated_data, output_file_path)
    else:
        print(f"No valid data found to concatenate for split {split}. Output file {output_file_path} not created.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel data processing scripts with a customizable spatial task.")

    parser.add_argument('--task', type=str, required=True,
                        help="dataset name (e.g., 'spatial', 'object', 'goal', '10').")
    parser.add_argument('--resolution', type=int, required=True,
                        help="resolution (e.g., 256, 512).")

    args = parser.parse_args()
    main(args)
