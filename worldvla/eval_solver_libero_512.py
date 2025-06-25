import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import numpy as np

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck
from xllmx.solvers.pretrain import PretrainSolverBase

import tqdm
from libero.libero import benchmark

from PIL import Image


# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from libero_util.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from libero_util.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck
from data.pre_tokenize_action import ItemProcessor
import time
import xllmx.util as util
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter



class Solver(PretrainSolverBase):
    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard"))


    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--task_suite_name", type=str, default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10",])
        parser.add_argument("--device", default=0, type=int, help="gpu device")
        parser.add_argument("--head", type=str, default="dis", choices=["dis", "ct"])
        parser.add_argument("--his", type=str, default="1h_1a", choices=["1h_1a", "2h_1a", "4h_1a", "2h_2a", "4h_4a", "1h_1a_img_only", "2h_1a_img_only", "4h_1a_img_only"])
        parser.add_argument("--action_steps", default=25, type=int, help="actions to be excuted when multiple actions are generated")
        parser.add_argument("--half", default=0, type=int, help="which part of test set will be evaluated")
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        model = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
            init_from,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        return model, None

    def _item_processor_func(self) -> ItemProcessor:
        return ItemProcessor(target_size=288)

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration_ck.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")
        
    def unnorm_min_max(self, action):
        min_values = np.array([-0.9375, -0.9375, -0.9375, -0.32571429, -0.375, -0.375, -1.0])
        max_values = np.array([0.9375, 0.9375, 0.9375, 0.375, 0.375, 0.375, 1.0])
        if action.shape[0] > 7:
            action = action[:7]
            
        unnorm_action = (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values
        
        return unnorm_action
    
    def val_libero(self,):
        self.model, _ = self._model_func(self.args.resume_path)
        DEVICE = torch.device(f"cuda:{self.args.device}")
        self.model = self.model.to(DEVICE)
        self.model.eval()
        item_processor = ItemProcessor(target_size=512)
        
        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.args.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Task suite: {self.args.task_suite_name}")

        # Get expected image dimensions
        resize_size = 512

        # Start evaluation
        total_episodes, total_successes = 0, 0
        # for task_id in tqdm.tqdm(range(3)):
        
        if args.half == 0:
            episode_range = range(num_tasks_in_suite)
        elif args.half == 1:
            episode_range = range(0, num_tasks_in_suite//2, 1)
            # episode_range = range(3, num_tasks_in_suite//2, 1)
        elif args.half == 2:
            episode_range = range(num_tasks_in_suite//2, num_tasks_in_suite, 1)
            # episode_range = range(num_tasks_in_suite//2+3, num_tasks_in_suite, 1)

        # Calculate the size of each partition
        # partition_size = num_tasks_in_suite // 5

        # if args.half == 0:
        #     # If half is 0, use the full range
        #     episode_range = range(num_tasks_in_suite)
        # elif 1 <= args.half <= 5:
        #     # For half values 1 to 5, calculate the corresponding partition
        #     start_index = (args.half - 1) * partition_size
        #     end_index = args.half * partition_size if args.half < 5 else num_tasks_in_suite
        #     episode_range = range(start_index, end_index)
        # else:
        #     # Handle invalid values of args.half
        #     raise ValueError("args.half must be an integer between 0 and 5.")

        
        for task_id in tqdm.tqdm(episode_range):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, resolution=512)
            
            # Start episodes
            task_episodes, task_successes = 0, 0

            # 使用 tqdm 进行进度条显示
            for episode_idx in tqdm.tqdm(range(50)):
                print(f"\nTask: {task_description}")

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                if self.args.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif self.args.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif self.args.task_suite_name == "libero_goal":
                    max_steps = 300  # longest training demo has 270 steps
                elif self.args.task_suite_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif self.args.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps
                
                his_img = []
                his_action = []
                actions_ck = []

                print(f"Starting episode {task_episodes+1}...")
                while t < max_steps + 10:
                    print("t: ", t)
                    # if 1:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < 10:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        
                        cur_img = Image.fromarray(img)
                        
                        # Query model to get action
                        if len(actions_ck) == 0:
                            dis_action = get_action_Chameleon_dis_awm_ck(
                                self.model,
                                cur_img,
                                task_description,
                                item_processor,
                                his_img,
                                his_action,
                                self.args.his,
                                self.args.action_steps
                            )
                            for i in range(len(dis_action)):
                                actions_ck.append(dis_action[i].cpu().float().detach().numpy())
                        
                        dis_action_tmp = actions_ck.pop(0)
                        dis_action_unnorm = self.unnorm_min_max(dis_action_tmp)
                        print(dis_action_unnorm)
                                                
                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        # dis_action = normalize_gripper_action(dis_action.cpu().float().detach().numpy(), binarize=True)
                        # ct_action = normalize_gripper_action(ct_action.cpu().float().detach().numpy(), binarize=True)
                        
                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        # if cfg.model_family == "openvla":
                        #     action = invert_gripper_action(action)
                        
                        # Execute action in environment
                        obs, reward, done, info = env.step(dis_action_unnorm.tolist())
                        
                        his_img.append(cur_img)
                        his_action.append(dis_action_tmp)
                        
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                save_rollout_video(
                    self.args.output_dir, replay_images, total_episodes, success=done, task_description=task_description,
                )

                # Log current results
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    # solver.run()
    solver.val_libero()
    # solver.eval()