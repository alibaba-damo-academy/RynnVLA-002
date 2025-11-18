import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import numpy as np

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.solvers.pretrain import PretrainSolverBase

import tqdm
from PIL import Image


from lerobot_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck, get_action_Chameleon_dis_awm_ck_wrist_action_head
from data_lerobot.pre_tokenize_action_state_2 import ItemProcessor
# from data_lerobot.pre_tokenize_action import ItemProcessor
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
        # self.item_processor = ItemProcessor(target_size=512)
        self.item_processor = ItemProcessor(target_size=256)
        print('init done 000000!')
        self.his_img = []
        self.model, _ = self._model_func(self.args.resume_path)
        DEVICE = torch.device(f"cuda:{self.args.device}")
        self.model = self.model.to(DEVICE)
        self.model.eval()
        print('init done!')


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
        parser.add_argument("--task_suite_name", type=str, default="libero_spatial",)
        parser.add_argument("--device", default=0, type=int, help="gpu device")
        parser.add_argument("--head", type=str, default="dis", choices=["dis", "ct"])
        parser.add_argument("--his", type=str, default="1h_1a", choices=["1h_1a", "2h_1a", "4h_1a", "2h_2a", "4h_4a", "1h_1a_img_only", "2h_1a_img_only", "4h_1a_img_only", "1h_1a_img_only_state",])
        parser.add_argument("--action_steps", default=25, type=int, help="actions to be excuted when multiple actions are generated")
        parser.add_argument("--half", default=0, type=int, help="which part of test set will be evaluated")
        parser.add_argument("--port", default=8000, type=int)
        parser.add_argument("--token", default='', type=str)
        parser.add_argument("--env", default='lerobot', type=str)
        parser.add_argument("--record", default=False, type=bool)
        parser.add_argument("--pack", default="protobuf", type=str)
        parser.add_argument("--action_rate", default=30, type=int)
        parser.add_argument("--compress", default='gzip', type=str)
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            init_from,
            action_dim=self.args.action_dim,
            time_horizon=self.args.time_horizon,
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

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
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
    
    def norm_action(self, action):

        # 将每个维度的最小值（第一列）组合成一个numpy数组
        # min_values = np.array([
        #     -4.74609375, 
        #     -5.53710938, 
        #     -5.71289062, 
        #     -4.65820312, 
        #     -6.24023438, 
        #     -5.60223961
        # ])

        # # 将每个维度的最大值（第二列）组合成一个numpy数组
        # max_values = np.array([
        #     4.57031250, 
        #     5.36132812, 
        #     7.99804688, 
        #     4.92187500, 
        #     6.41601562, 
        #     5.22875786
        # ])

        min_values = np.array([
            -55.89843750,
            -67.23632812,
            -100.81054688,
            -62.22656250,
            -77.78320312,
            -53.71041489
        ])

        # 将每个维度的最大值（第二列）组合成一个numpy数组
        max_values = np.array([
            57.48046875,
            70.92773438,
            96.41601562,
            52.20703125,
            102.56835938,
            52.23909760
        ])

        min_values = np.array([
            -23.99414062,
            -32.51953125,
            -49.04296875,
            -23.55468750,
            -27.07031250,
            -38.29353333
        ])

        # 将每个维度的 99% 分位数 (q99) 组合成一个numpy数组
        max_values = np.array([
            25.92773438,
            44.73632812,
            36.91406250,
            24.78515625,
            33.39843750,
            36.47498703
        ])

        norm_action = 2 * (action - min_values) / (max_values - min_values + 1e-8) - 1
        norm_action = np.clip(norm_action, a_min=-1, a_max=1)
        
        return norm_action
        
    def unnorm_min_max(self, action):
        # min_values = np.array([
        #     -4.74609375, 
        #     -5.53710938, 
        #     -5.71289062, 
        #     -4.65820312, 
        #     -6.24023438, 
        #     -5.60223961
        # ])

        # # 将每个维度的最大值（第二列）组合成一个numpy数组
        # max_values = np.array([
        #     4.57031250, 
        #     5.36132812, 
        #     7.99804688, 
        #     4.92187500, 
        #     6.41601562, 
        #     5.22875786
        # ])

        # min_values = np.array([
        #     -55.89843750,
        #     -67.23632812,
        #     -100.81054688,
        #     -62.22656250,
        #     -77.78320312,
        #     -53.71041489
        # ])

        # # 将每个维度的最大值（第二列）组合成一个numpy数组
        # max_values = np.array([
        #     57.48046875,
        #     70.92773438,
        #     96.41601562,
        #     52.20703125,
        #     102.56835938,
        #     52.23909760
        # ])

        # min_values = np.array([
        #     -23.99414062,
        #     -32.51953125,
        #     -49.04296875,
        #     -23.55468750,
        #     -27.07031250,
        #     -38.29353333
        # ])

        # # 将每个维度的 99% 分位数 (q99) 组合成一个numpy数组
        # max_values = np.array([
        #     25.92773438,
        #     44.73632812,
        #     36.91406250,
        #     24.78515625,
        #     33.39843750,
        #     36.47498703
        # ])

        # min_values = np.array([
        #     -23.99414062,
        #     -32.51953125,
        #     -49.04296875,
        #     -23.55468750,
        #     -27.07031250,
        #     -0.64464140
        # ])

        # # 将每个维度的 99% 分位数 (q99) 组合成一个numpy数组
        # max_values = np.array([
        #     25.92773438,
        #     44.73632812,
        #     36.91406250,
        #     24.78515625,
        #     33.39843750,
        #     50.04029083
        # ])

        min_values = np.array([
            -27.33398438,  # 维度 0 的最小值
            -27.24609375,  # 维度 1 的最小值
            -56.60156250,  # 维度 2 的最小值
            -51.50390625,  # 维度 3 的最小值
            -70.13671875,  # 维度 4 的最小值
            -9.07504368   # 维度 5 的最小值
        ])

        max_values = np.array([
            31.20117188,   # 维度 0 的最大值
            29.44335938,   # 维度 1 的最大值
            34.89257812,   # 维度 2 的最大值
            45.87890625,   # 维度 3 的最大值
            62.31445312,   # 维度 4 的最大值
            74.56647491    # 维度 5 的最大值
        ])

        # min_values = np.array([
        #     -27.33398438,  # 维度 0 的最小值
        #     -27.24609375,  # 维度 1 的最小值
        #     -56.60156250,  # 维度 2 的最小值
        #     -51.50390625,  # 维度 3 的最小值
        #     -70.13671875,  # 维度 4 的最小值
        #     -46.76074982   # 维度 5 的最小值
        # ])

        # max_values = np.array([
        #     31.20117188,   # 维度 0 的最大值
        #     29.44335938,   # 维度 1 的最大值
        #     34.89257812,   # 维度 2 的最大值
        #     45.87890625,   # 维度 3 的最大值
        #     62.31445312,   # 维度 4 的最大值
        #     41.16186142    # 维度 5 的最大值
        # ])        
            
        unnorm_action = (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values
        
        return unnorm_action
    
    def read_imgs_test(self,):
        # 1. 定义图片所在的目录
        base_dir = "/mnt/damorobot/cenjun/processed_data_lerobot/front_image"

        # 2. 初始化一个空列表来存储图片
        all_imgs = []
        all_imgs_wrist = []

        # 3. 定义目标格式
        target_shape = (640, 480, 3) # (height, width, channels)
        target_size = (target_shape[1], target_shape[0]) # (width, height) for PIL resize
        num_images = 10
        file_extension = ".png"  #  <-- 如果是 .jpg 或其他格式，请修改这里

        print(f"开始从目录 '{base_dir}' 加载图片...")

        # 4. 循环读取 image_0 到 image_9
        for i in range(num_images):
            # 构建完整的文件路径
            filename = f"image_{i}{file_extension}"
            file_path = os.path.join(base_dir, filename)

            try:
                # 使用Pillow打开图片文件
                img = Image.open(file_path)

                # 确保图片是RGB格式（处理灰度图或RGBA图的情况）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图片尺寸以匹配目标格式 (480, 640)
                # if img.size != target_size:
                #     img = img.resize(target_size, Image.Resampling.LANCZOS)

                # 将Pillow Image对象转换为NumPy数组
                # 格式为 (height, width, channels)，数据类型为 uint8
                img_array = np.array(img)
                print(img_array.shape)

                # 将处理好的数组添加到列表中
                all_imgs.append(img_array)

            except FileNotFoundError:
                print(f"警告: 文件未找到 {file_path}，已跳过。")
            except Exception as e:
                print(f"错误: 加载文件 {file_path} 时出错: {e}")
            
            file_path = file_path.replace('front_image', 'wrist_image')

            try:
                # 使用Pillow打开图片文件
                img = Image.open(file_path)

                # 确保图片是RGB格式（处理灰度图或RGBA图的情况）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图片尺寸以匹配目标格式 (480, 640)
                # if img.size != target_size:
                #     img = img.resize(target_size, Image.Resampling.LANCZOS)

                # 将Pillow Image对象转换为NumPy数组
                # 格式为 (height, width, channels)，数据类型为 uint8
                img_array = np.array(img)
                print(img_array.shape)

                # 将处理好的数组添加到列表中
                all_imgs_wrist.append(img_array)

            except FileNotFoundError:
                print(f"警告: 文件未找到 {file_path}，已跳过。")
            except Exception as e:
                print(f"错误: 加载文件 {file_path} 时出错: {e}")

        base_dir = "/mnt/damorobot/cenjun/processed_data_lerobot/action"
        num_files = 29

        try:
            # 使用列表推导式在一行内完成所有文件的加载
            actions_list = [
                np.load(os.path.join(base_dir, f"action_{i}.npy")) 
                for i in range(num_files)
            ]

            # 将列表堆叠成一个数组
            all_actions = np.stack(actions_list)

            # 验证结果
            print("所有 action 文件加载并合并成功！")
            print(f"最终数组的形状 (shape): {all_actions.shape}")
            print(f"最终数组的数据类型 (dtype): {all_actions.dtype}")

        except FileNotFoundError as e:
            print(f"错误: 文件未找到 - {e}。请确保所有 action_0.npy 到 action_28.npy 文件都存在。")
        except Exception as e:
            print(f"发生错误: {e}")

        return all_imgs, all_imgs_wrist, all_actions
    
    def load_data_by_index(self, idx: int, base_path: str = "/public/hz_oss/cenjun/WorldVLA/worldvla/ckpts/data_1/episode_000019_q01q99"):
        """
        根据索引(idx)加载对应的图像和动作数据。

        Args:
            idx (int): 要加载的数据帧的索引。
            base_path (str): 数据集的根目录路径。

        Returns:
            tuple: 包含三个元素的元组:
                - front_image_array (np.ndarray): 前置摄像头的图像数组。
                - wrist_image_array (np.ndarray): 手腕摄像头的图像数组。
                - concatenated_action (np.ndarray): 形状为 (20, 6) 的合并后的动作数组。
        """
        try:
            # 1. 构建图像文件路径
            front_image_path = os.path.join(base_path, 'front_image', f'image_{idx}.png')
            wrist_image_path = os.path.join(base_path, 'wrist_image', f'image_{idx}.png')

            # 2. 读取图像并转换为Numpy数组
            with Image.open(front_image_path) as img:
                front_image_array = np.array(img)
            
            with Image.open(wrist_image_path) as img:
                wrist_image_array = np.array(img)

            # 3. 构建动作文件夹路径
            action_folder_path = os.path.join(base_path, 'action', f'action_{idx}')

            # 4. 循环读取20个动作npy文件并将它们收集到列表中
            action_parts = [
                np.load(os.path.join(action_folder_path, f'{i}.npy'))
                for i in range(20)
            ]

            # 5. 将动作列表堆叠成一个(20, 6)的数组
            # np.stack会沿着新的轴（默认axis=0）堆叠数组
            concatenated_action = np.stack(action_parts, axis=0)
            # concatenated_action = self.norm_action(concatenated_action)
            # concatenated_action = self.unnorm_min_max(concatenated_action)


            # 6. 返回结果
            return front_image_array, wrist_image_array, concatenated_action

        except FileNotFoundError as e:
            print(f"错误：文件未找到。请检查索引 '{idx}' 是否正确，以及路径 '{base_path}' 是否存在。")
            print(f"具体错误信息: {e}")
            return None, None, None
        except Exception as e:
            print(f"加载数据时发生未知错误: {e}")
            return None, None, None
    
    def load_data_by_index_state(self, idx: int, base_path: str = "/public/hz_oss/cenjun/WorldVLA/ckpts/data_1/episode_000026"):
        """
        根据索引(idx)加载对应的图像和动作数据。

        Args:
            idx (int): 要加载的数据帧的索引。
            base_path (str): 数据集的根目录路径。

        Returns:
            tuple: 包含三个元素的元组:
                - front_image_array (np.ndarray): 前置摄像头的图像数组。
                - wrist_image_array (np.ndarray): 手腕摄像头的图像数组。
                - concatenated_action (np.ndarray): 形状为 (20, 6) 的合并后的动作数组。
        """
        try:
            # 1. 构建图像文件路径
            front_image_path = os.path.join(base_path, 'front_image', f'image_{idx}.png')
            wrist_image_path = os.path.join(base_path, 'wrist_image', f'image_{idx}.png')

            # 2. 读取图像并转换为Numpy数组
            with Image.open(front_image_path) as img:
                front_image_array = np.array(img)
            
            with Image.open(wrist_image_path) as img:
                wrist_image_array = np.array(img)

            # 3. 构建动作文件夹路径
            action_folder_path = os.path.join(base_path, 'abs_action', f'action_{idx}')

            # 4. 循环读取20个动作npy文件并将它们收集到列表中
            action_parts = [
                np.load(os.path.join(action_folder_path, f'{i}.npy'))
                for i in range(20)
            ]

            # 5. 将动作列表堆叠成一个(20, 6)的数组
            # np.stack会沿着新的轴（默认axis=0）堆叠数组
            concatenated_action = np.stack(action_parts, axis=0)
            # concatenated_action = self.norm_action(concatenated_action)
            # concatenated_action = self.unnorm_min_max(concatenated_action)

            state_path = os.path.join(base_path, 'state', f'state_{idx}.npy')
            state = np.load(state_path)


            # 6. 返回结果
            return front_image_array, wrist_image_array, concatenated_action, state

        except FileNotFoundError as e:
            print(f"错误：文件未找到。请检查索引 '{idx}' 是否正确，以及路径 '{base_path}' 是否存在。")
            print(f"具体错误信息: {e}")
            return None, None, None
        except Exception as e:
            print(f"加载数据时发生未知错误: {e}")
            return None, None, None
    
    def val_libero(self,):
        self.model, _ = self._model_func(self.args.resume_path)
        DEVICE = torch.device(f"cuda:{self.args.device}")
        self.model = self.model.to(DEVICE)
        self.model.eval()
        item_processor = ItemProcessor(target_size=512)

        # Get expected image dimensions
        all_imgs, all_imgs_wrist, all_actions = self.read_imgs_test() #all_imgs为包含10个(480, 640, 3) array, all_action shape为(29,6)
        his_img = []
        error_list = []
        for idx, img_c in enumerate(all_imgs):
            cur_img = Image.fromarray(img_c)

            
            # Query model to get action
            import time
            start = time.time()
            dis_action = get_action_Chameleon_dis_awm_ck(
                self.model,
                cur_img,
                self.args.task_suite_name,
                item_processor,
                his_img,
                self.args.his,
                self.args.action_steps
            )
            print(time.time()-start)
            dis_action = torch.stack(dis_action, dim=0).cpu().numpy()
            dis_action_unnorm = self.unnorm_min_max(dis_action)
            gt_action = all_actions[idx: idx+20]
            print(dis_action_unnorm, gt_action)
            print(np.mean(dis_action_unnorm - gt_action))
            his_img = [cur_img]
            error_list.append(np.mean(dis_action_unnorm - gt_action))
        print(error_list)

    def val_libero_wrist(self,):
        item_processor = ItemProcessor(target_size=512)

        # Get expected image dimensions
        all_imgs, all_imgs_wrist, all_actions = self.read_imgs_test() #all_imgs为包含10个(480, 640, 3) array, all_action shape为(29,6)
        his_img = []
        error_list = []
        for idx, img_c in enumerate(all_imgs):
            cur_img = Image.fromarray(img_c)
            cur_img = Image.fromarray(all_imgs_wrist[idx])
            cur_img_wrist = Image.fromarray(all_imgs_wrist[idx])
            # cur_img_wrist = Image.fromarray(img_c)
            
            # Query model to get action
            import time
            start = time.time()
            dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                self.model,
                cur_img,
                cur_img_wrist,
                self.args.task_suite_name,
                item_processor,
                his_img,
                self.args.his,
                self.args.action_steps
            )
            print(time.time()-start)
            dis_action = torch.stack(dis_action, dim=0).cpu().numpy()
            dis_action_unnorm = self.unnorm_min_max(dis_action)
            gt_action = all_actions[idx: idx+20]
            print(dis_action_unnorm, gt_action)
            print(np.mean(dis_action_unnorm - gt_action))
            his_img = [cur_img]
            error_list.append(np.mean(dis_action_unnorm - gt_action))
        print(error_list)
    
    def val_libero_wrist_2(self,):
        # item_processor = ItemProcessor(target_size=512)
        item_processor = ItemProcessor(target_size=256)

        error_list = []
        for idx in range(0,1200,20):
            front_image_array, wrist_image_array, concatenated_action = self.load_data_by_index(idx)
            cur_img = Image.fromarray(front_image_array)
            cur_img_wrist = Image.fromarray(wrist_image_array)
            print(cur_img.size, cur_img_wrist.size)
            # print(concatenated_action)
            # continue
            # cur_img_wrist = Image.fromarray(img_c)
            his_img = []
            # Query model to get action
            import time
            start = time.time()
            dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                self.model,
                cur_img,
                cur_img_wrist,
                self.args.task_suite_name,
                item_processor,
                his_img,
                self.args.his,
                self.args.action_steps
            )
            print(time.time()-start)
            dis_action = dis_action.cpu().float().detach().numpy()
            # dis_action_unnorm = dis_action
            dis_action_unnorm = self.unnorm_min_max(dis_action)
            gt_action = concatenated_action
            print(dis_action_unnorm, gt_action)
            print(np.mean(np.abs(dis_action_unnorm - gt_action)))
            his_img = [cur_img]
            error_list.append(np.mean(np.abs(dis_action_unnorm - gt_action)))
            # import pdb; pdb.set_trace()
        print(error_list)
    
    def val_libero_wrist_state(self,):
        # item_processor = ItemProcessor(target_size=512)
        item_processor = ItemProcessor(target_size=256)

        error_list = []
        for idx in range(0,700,20):
            front_image_array, wrist_image_array, concatenated_action, state = self.load_data_by_index_state(idx)
            cur_img = Image.fromarray(front_image_array)
            cur_img_wrist = Image.fromarray(wrist_image_array)
            print(cur_img.size, cur_img_wrist.size)
            # print(concatenated_action)
            # continue
            # cur_img_wrist = Image.fromarray(img_c)
            his_img = []
            # Query model to get action
            import time
            start = time.time()
            dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                self.model,
                cur_img,
                cur_img_wrist,
                self.args.task_suite_name,
                item_processor,
                his_img,
                self.args.his,
                self.args.action_steps,
                state
            )
            print(time.time()-start)
            dis_action = dis_action.cpu().float().detach().numpy()
            # dis_action_unnorm = dis_action
            dis_action_unnorm = self.unnorm_min_max(dis_action)
            gt_action = concatenated_action
            print(dis_action_unnorm, gt_action)
            print(np.mean(np.abs(dis_action_unnorm - gt_action)))
            his_img = [cur_img]
            error_list.append(np.mean(np.abs(dis_action_unnorm - gt_action)))
            # import pdb; pdb.set_trace()
        print(error_list, np.mean(error_list))
    
    def get_action(self, input_image, state, prompt):

        dis_action = get_action_Chameleon_dis_awm_ck(
                self.model,
                input_image,
                self.args.task_suite_name,
                self.item_processor,
                self.his_img,
                self.args.his,
                self.args.action_steps
            )
        dis_action = torch.stack(dis_action, dim=0).cpu().numpy()
        dis_action_unnorm = self.unnorm_min_max(dis_action)

        self.his_img = [input_image]

        return dis_action_unnorm
    
    def get_action_wrist_action_head(self, input_image, img1, state, prompt):

        # cur_img = Image.fromarray(input_image)
        # cur_img_wrist = Image.fromarray(img1)
        # print(type(input_image), type(img1))

        # save_images(input_image, img1)

        dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                self.model,
                input_image,
                img1,
                self.args.task_suite_name,
                self.item_processor,
                self.his_img,
                self.args.his,
                self.args.action_steps
            )
        dis_action = dis_action.cpu().float().detach().numpy()

        # threshold_1 = 0.05  # 你需要设置合适的值
        # threshold_2 = -0.8  # 你需要设置合适的值
        
        # # 应用阈值处理到最后一维
        # dis_action[:, -1] = np.where(dis_action[:, -1] > threshold_1, 0.5, 
        #                         np.where(dis_action[:, -1] < threshold_2, -1, dis_action[:, -1]))

        # threshold_1 = 0.045  # 根据你的需求设置阈值
        # threshold_2 = 0.04
    
        # # 对最后一维（第6维）进行条件处理
        # mask = dis_action[:, -1] > threshold_1
        # dis_action[mask, -1] *= 10

        # dis_action[:, -1] = np.where(dis_action[:, -1] < threshold_2, -1, dis_action[:, -1])

        # dis_action[:, -1] = np.where(dis_action[:, -1] < 0, -1, dis_action[:, -1])
        
        dis_action_unnorm = self.unnorm_min_max(dis_action)



        self.his_img = [input_image]

        return dis_action_unnorm

    def get_action_wrist_action_head_state(self, input_image, img1, state, prompt):

            # cur_img = Image.fromarray(input_image)
            # cur_img_wrist = Image.fromarray(img1)
            # print(type(input_image), type(img1))

            # save_images(input_image, img1)

            print('prompt 1: ', prompt)

            dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                    self.model,
                    input_image,
                    img1,
                    prompt,
                    self.item_processor,
                    self.his_img,
                    self.args.his,
                    self.args.action_steps,
                    state
                )
            dis_action = dis_action.cpu().float().detach().numpy()
            
            dis_action_unnorm = self.unnorm_min_max(dis_action)



            self.his_img = [input_image]

            return dis_action_unnorm
    

def save_images(input_image, img1, base_dir="output_exps"):
    """
    保存两张图片到不同文件夹，自动按序号命名
    
    Args:
        input_image: numpy array 或 PIL Image，第一张图片
        img1: numpy array 或 PIL Image，第二张图片
        base_dir: 基础目录名
    """
    # 创建文件夹
    img_dir = os.path.join(base_dir, "images")
    wrist_dir = os.path.join(base_dir, "wrist_images")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)
    
    # 转换为PIL Image
    if hasattr(input_image, 'shape'):  # 如果是numpy array
        cur_img = Image.fromarray(input_image)
    else:
        cur_img = input_image
        
    if hasattr(img1, 'shape'):  # 如果是numpy array
        cur_img_wrist = Image.fromarray(img1)
    else:
        cur_img_wrist = img1
    
    # 获取下一个文件编号
    def get_next_number(directory):
        existing_files = [f for f in os.listdir(directory) if f.endswith('.png')]
        if not existing_files:
            return 1
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('.')[0])
                numbers.append(num)
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 1
    
    # 获取编号并保存
    img_num = get_next_number(img_dir)
    wrist_num = get_next_number(wrist_dir)
    
    # 保存图片
    cur_img.save(os.path.join(img_dir, f"{img_num}.png"))
    cur_img_wrist.save(os.path.join(wrist_dir, f"{wrist_num}.png"))
    
    print(f"已保存: {img_num}.png 到 {img_dir}")
    print(f"已保存: {wrist_num}.png 到 {wrist_dir}")




if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    # solver.run()
    # solver.val_libero()
    # solver.val_libero_wrist_2()
    solver.val_libero_wrist_state()
    # solver.eval()