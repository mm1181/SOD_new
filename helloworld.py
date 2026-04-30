import os
import sys

#统计数据集数量123456

def count_files_in_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"{dirpath}: {len(filenames)} files")

if __name__ == "__main__":
    target_dir = "/root/autodl-tmp/my_SOD_3/data"

    if not os.path.isdir(target_dir):
        print(f"错误：目录不存在 - {target_dir}", file=sys.stderr)
        sys.exit(1)

    count_files_in_dirs(target_dir)