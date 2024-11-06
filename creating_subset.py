# Creating subset
import os
import numpy as np
import shutil

# 设置文件夹路径
data_root = "/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour"
images_dir = os.path.join(data_root, "imagesTr")
labels_dir = os.path.join(data_root, "labelsTr")

# 设置输出文件夹
subset_images_dir = os.path.join(data_root, "imagesTr_subset")
subset_labels_dir = os.path.join(data_root, "labelsTr_subset")
os.makedirs(subset_images_dir, exist_ok=True)
os.makedirs(subset_labels_dir, exist_ok=True)

# 获取所有图像文件并提取唯一编号
image_files = os.listdir(images_dir)
# 提取文件编号，例如 'BRATS_005.nii.gz' 提取为 '005'
image_ids = sorted(list(set([f.split("_")[1].split(".")[0] for f in image_files])))
print(f"Total unique image IDs: {len(image_ids)}")

# 随机抽取100个编号
np.random.seed(42)  # 设置随机种子以保证可重复性
sampled_ids = np.random.choice(image_ids, 100, replace=False)
print(f"Sampled image IDs: {sampled_ids}")

# 复制对应的 images 和 labels 文件到子集文件夹
for image_id in sampled_ids:
    # 构建图像文件名
    img_file = f"BRATS_{image_id}.nii.gz"
    src_image_path = os.path.join(images_dir, img_file)
    dst_image_path = os.path.join(subset_images_dir, img_file)
    
    # 如果图像文件存在，复制到子集文件夹
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
    else:
        print(f"Warning: Image file {img_file} not found for image ID {image_id}")
    
    # 构建标签文件名
    label_file = f"BRATS_{image_id}.nii.gz"
    src_label_path = os.path.join(labels_dir, label_file)
    dst_label_path = os.path.join(subset_labels_dir, label_file)
    
    # 如果标签文件存在，复制到子集文件夹
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)
    else:
        print(f"Warning: Label file {label_file} not found for image ID {image_id}")

print("Subset creation complete!")



# create related json file 
# create related json file 
# create related json file 
import os
import json

# 指定 imagesTr 和 labelsTr 文件夹路径
images_dir = '/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour/imagesTr'  # 替换为您的 imagesTr 文件夹路径
labels_dir = '/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour/labelsTr'  # 替换为您的 labelsTr 文件夹路径

# 初始化训练数据列表
training_data = []

# 遍历 imagesTr 文件夹，假设 imagesTr 和 labelsTr 文件名相同
for image_file in sorted(os.listdir(images_dir)):
    if image_file.endswith('.nii.gz'):
        # 构造图像和标签文件路径
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file)  # 假设 label 文件名与 image 文件名相同

        # 检查标签文件是否存在
        if os.path.exists(label_path):
            # 添加到训练数据列表
            training_data.append({
                "image": f"./imagesTr/{image_file}",
                "label": f"./labelsTr/{image_file}"
            })

# 构造 JSON 数据
data = {"training": training_data}

# 保存为 JSON 文件
output_json_path = '/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour/training_data.json'  # 保存路径
with open(output_json_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON 文件已保存到 {output_json_path}")
