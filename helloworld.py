import os
import sys

def count_files_in_dirs(root_dir):
    """递归遍历 root_dir，输出每个目录下的文件数量。"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # dirpath: 当前目录的绝对路径
        # filenames: 当前目录下的直接文件列表（不包含子目录）
        print(f"{dirpath}: {len(filenames)} files")

if __name__ == "__main__":
    target_dir = "/root/autodl-tmp/my_SOD_3/data"

    if not os.path.isdir(target_dir):
        print(f"错误：目录不存在 - {target_dir}", file=sys.stderr)
        sys.exit(1)

    count_files_in_dirs(target_dir)