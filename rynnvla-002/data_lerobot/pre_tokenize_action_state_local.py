import os
import sys

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from argparse import ArgumentParser
import json
import math
import pickle

from convertsation import Conversation
from item_processor import FlexARItemProcessor_Action_State


class ItemProcessor(FlexARItemProcessor_Action_State):
    def __init__(
        self,
        # tokenizer= "/public/hz_oss/cenjun/hug_models/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af",
        tokenizer= "/public/hz_oss/cenjun/hug_models/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af",
        conv_template=Conversation,
        target_size=512,
    ):
        super().__init__(tokenizer, conv_template, target_size)
        print(self.crop_size_list)

    def process_item(self, raw_item, training_mode=False, out_flatten=True):

        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys

        # ********* <start>  Add your custom codes here *******

        # *********  <end>   Add your custom codes here *******

        item = {
            "conversations": raw_item["conversations"],
            "image": raw_item["image"],
            "action": raw_item["action"],
            "state": raw_item["state"],
        }

        return super(ItemProcessor, self).process_item(item, training_mode, out_flatten)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    parser.add_argument("--target_size", type=int, default=512)
    args = parser.parse_args()

    item_processor = ItemProcessor(target_size=args.target_size)

    with open(args.in_filename) as f:
        ori_contents = json.load(f)
    
    num = len(ori_contents)

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)

    try:
        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            start_idx = int(f.read()) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print(f"{i}/{end_idx}")

        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        try:
            tokens, labels = item_processor.process_item(ori_contents[i], training_mode=True)
            new_item = {"token": tokens, "label": labels, "id": i}
            with open(pkl_path, "wb") as f:
                pickle.dump(new_item, f)

            record = {"file": pkl_path, "len": len(tokens), "id": i}

        except Exception as e:
            from traceback import format_exc

            print(f"item {i} error: \n{ori_contents[i]}")
            print(format_exc())

        if record is not None:
            with open(os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl"), "a") as f:
                record_str = json.dumps(record) + "\n"
                f.write(record_str)

        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w") as f:
            if i == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{i}")
