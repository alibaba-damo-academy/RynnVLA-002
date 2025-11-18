import numpy as np

def print_npy_info(file_path):
    # 加载 .npy 文件
    array = np.load(file_path, allow_pickle=True)
    
    print('array:', array)
    
    # 输出数组的信息
    # print("Array Information:")
    # print(f"  Shape: {array.shape}")
    # print(f"  Data type: {array.dtype}")
    # print(f"  Length: {array.size}")

def update_npy_indx(file_path):
    content = np.load(file_path, allow_pickle=True)
    print('type(content):', type(content))
    print(content["language"])
    # print("content['info']['indx']:", content['info']['indx'])
    # print("type(content['info']['indx']):", type(content['info']['indx']))
    # content['info']['indx'] = content['info']['indx'][0:1]
    # np.save(file_path, content)

# 示例使用
# file_path = '/mnt/PLNAS/cenjun/libero/processed_data/libero_goal_image_state_action_t_256/turn_on_the_stove/trj_1/action/action_11.npy'  # 替换为您实际的 .npy 文件路径
# # file_path = '/mnt/PLNAS/cenjun/libero/processed_data/libero_goal_image_state_action_t_256/turn_on_the_stove/trj_1/gripper_state/gripper_state_11.npy'  # 替换为您实际的 .npy 文件路径
# print_npy_info(file_path)

# file_path = '/mnt/PLNAS/cenjun/libero/processed_data/libero_goal_image_state_action_t_256/turn_on_the_stove/trj_1/ee_state/ee_state_11.npy'  # 替换为您实际的 .npy 文件路径
# print_npy_info(file_path)

# file_path = '/mnt/PLNAS/cenjun/libero/processed_data/libero_goal_image_state_action_t_256/turn_on_the_stove/trj_1/ee_state/ee_state_12.npy'  # 替换为您实际的 .npy 文件路径
# print_npy_info(file_path)

file_path = '/mnt/PLNAS/cenjun/libero/processed_data/libero_10_image_state_action_t_512/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it/trj_49/eef_gripper_state/eef_gripper_state_9.npy'  # 替换为您实际的 .npy 文件路径
print_npy_info(file_path)