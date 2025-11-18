import os

def find_episode_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 忽略空目录
        if not dirnames:
            continue
        
        # 判断当前目录下的所有子目录是否都以 'episode' 开头
        all_episodes = all(dirname.lower().startswith('episode') for dirname in dirnames)
        
        if all_episodes:
            print(f"{os.path.abspath(dirpath)}")
            # 停止继续遍历这个目录下的内容
            dirnames.clear()

find_episode_directories('/mnt/PLNAS/cenjun/all_data/extracted')