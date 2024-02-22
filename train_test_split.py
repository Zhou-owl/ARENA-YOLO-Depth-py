import os
import random
import shutil

# 原数据集目录
root_dir = 'dataset/tripods'
# 划分比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 设置随机种子
random.seed(42)

# 拆分后数据集目录
split_dir = 'dataset/tripods'
os.makedirs(os.path.join(split_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)

# 获取图片文件列表
image_files = os.listdir(os.path.join(root_dir, 'images'))
label_files = os.listdir(os.path.join(root_dir, 'labels'))

# 随机打乱文件列表
combined_files = list(zip(image_files, label_files))
random.shuffle(combined_files)
image_files_shuffled, label_files_shuffled = zip(*combined_files)

# 根据比例计算划分的边界索引
train_bound = int(train_ratio * len(image_files_shuffled))
valid_bound = int((train_ratio + valid_ratio) * len(image_files_shuffled))

# 将图片和标签文件移动到相应的目录
for i, (image_file, label_file) in enumerate(zip(image_files_shuffled, label_files_shuffled)):
    if i < train_bound:
        shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'train/images', image_file))
        shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'train/labels', label_file))
    elif i < valid_bound:
        shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'valid/images', image_file))
        shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'valid/labels', label_file))
    else:
        shutil.move(os.path.join(root_dir, 'images', image_file), os.path.join(split_dir, 'test/images', image_file))
        shutil.move(os.path.join(root_dir, 'labels', label_file), os.path.join(split_dir, 'test/labels', label_file))
