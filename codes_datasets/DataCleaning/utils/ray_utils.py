import os
import ray
import argparse

def change_extension(filename, new_extension):
    base_name = os.path.splitext(filename)[0]  # 获取文件名（不包含扩展名）
    new_filename = f"{base_name}.{new_extension}"  # 构建新的文件名
    return new_filename

def process_files_extension(folder_path, raw_extension, new_extension):
    for filename in os.listdir(folder_path):
        if filename.endswith(f".{raw_extension}"):
            old_path = os.path.join(folder_path, filename)
            new_filename = change_extension(filename, f"{new_extension}")  # 修改后缀为"modified"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"重命名文件：{filename} -> {new_filename}")

def test_folder(dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    GoodDir = os.path.join(dest_path, "good")
    BadDir = os.path.join(dest_path, "bad")

    if not os.path.exists(GoodDir):
        os.makedirs(GoodDir, exist_ok=True)
    if not os.path.exists(BadDir):
        os.makedirs(BadDir,exist_ok=True)
    
    return GoodDir, BadDir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path',
                        type=str,
                        default="/data/datacleansing/test",
                        help='Directory containing trained actor model')
    parser.add_argument('--dest_path',
                        type=str,
                        default="/data/datacleansing/test_store",
                        help='Directory containing trained actor model')
    parser.add_argument('--dataset_name',
                        type=str,
                        default="",
                        help="")
    args = parser.parse_args()
    return args

