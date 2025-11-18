import json
import os
import math
import copy
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data to create conversational datasets for LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required or optional arguments
    parser.add_argument(
        '--his', '-H', type=int, default=1,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='goal',
        help="A string used in the output JSON file names to identify the task type."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=512,
        help='The image resolution, used in the output JSON file names.'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='/mnt/PLNAS/cenjun/all_data/extracted/convs',
        help='Directory where the generated JSON dataset files will be saved.'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Use arguments
    his = args.his
    task_name_for_output = args.task_name
    resolution = args.resolution
    output_dir = args.output_dir

    train_convs = []

    print(f"Historical frames (his): {his}")
    print(f"Output files will use task_name: '{task_name_for_output}' and resolution: {resolution}")
    print(f"Output directory: {output_dir}")

    # Define the list of specific task paths to process
    specific_task_paths = [
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_02/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll-2",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll-1",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_02/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cup",
        "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_strawberries_from_the_table_into_the_cup./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/pc_03/grab_strawberries/so100_Pick-up-the-strawberries-and-put-them-in-the-cap",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/laptop_03/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder-complete",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/pc_02/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder",
        "/mnt/PLNAS/cenjun/all_data/extracted/Pick_up_the_holder_and_place_it_straight_then_pick_up_the_pen_and_place_it_in_the_holder/siteng/my_lerobot_test_data/hdf5/standard_desktop_data/data_0703/pc_03/grab_pen/so100_Pick-up-the-holder-and-place-it-straight-then-pick-up-the-pen-and-place-it-in-the-holder"
    ]

    # specific_task_paths = [
    #     "/mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/my_lerobot_test_data/hdf5/standard_desktop_data/data_0625/laptop_03/grab_blocks/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll",
    # ]

    for task_id, task_path in enumerate(tqdm(specific_task_paths, desc="Processing Tasks")):
        trj_list = sorted(os.listdir(task_path))
        
        for i, trj in enumerate(tqdm(trj_list, desc=f"  - Trajectories", leave=False)):
            trj_path = os.path.join(task_path, trj)
            action_path = os.path.join(trj_path, 'rel_action')
            imgs_path = os.path.join(trj_path, 'wrist_image')
            # Optionally also support wrist images
            # imgs_path_w = os.path.join(trj_path, 'wrist_image')

            img_list = []
            action_list = []

            # Determine the number of frames by checking the action directory
            num_frames = len(os.listdir(action_path)) if os.path.exists(action_path) else 0
            for j in range(num_frames):
                action_file = os.path.join(action_path, f"action_{j}", "0.npy")
                img_file = os.path.join(imgs_path, f"image_{j}.png")
                
                # Check if files exist
                if os.path.exists(action_file) and os.path.exists(img_file):
                    img_list.append(img_file)
                    action_list.append(action_file)
                else:
                    break  # Stop processing this trajectory if any missing

            # A conversation requires 'his' history frames + 1 future frame.
            if len(img_list) < his + 1:
                continue  # Skip this trajectory

            # Generate conversations for each valid step
            for j in range(len(action_list) - 1):
                img_c_h = copy.deepcopy(img_list[max(j - his + 1, 0) : j + 1])
                action_c = copy.deepcopy(action_list[max(j - his + 1, 0) : j + 1])
                img_c_f = copy.deepcopy(img_list[j + 1 : j + 2])

                conv = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>" * len(img_c_h)
                        },
                        {
                            "from": "gpt",
                            "value": "<|image|>"
                        }
                    ],
                    "image": img_c_h + img_c_f,
                    "action": action_c
                }

                train_convs.append(conv)

    # Construct output filename
    output_filename = f'libero_{task_name_for_output}_his_{his}_train_a2i_{resolution}_rel_wrist_all_data.json'
    output_path = os.path.join(output_dir, output_filename)

    # Save dataset
    print(f"\nSaving training data to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(train_convs, f, indent=4)

    print("\n--- Dataset Generation Summary ---")
    print(f"Total conversations generated: {len(train_convs)}")

if __name__ == '__main__':
    main()