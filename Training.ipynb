import os
import time
import torch
import nnunetv2
# 设置任务参数
task_id = 900
model_type = '2d'
num_gpus = 3  # 使用的 GPU 数量

# 设置 GPU 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1 "  # 确保设备编号在范围内 # 替换为你想使用的 GPU
os.environ["nnUNet_compile"] = "f" 
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


# 设置 nnUNet 环境变量
os.environ["nnUNet_raw"] = "/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/data_lg/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_results"

# 训练命令模板
command_template = "nnUNetv2_train {task_id} {model_type} {fold} -device cuda -num_gpus {num_gpus}"

# 记录训练开始时间
start_time = time.time()

# 选择 fold (可以是 0 到 4)
fold = 0  # 这里选择 fold 1，可以更改
command = command_template.format(task_id=task_id, model_type=model_type, fold=fold, num_gpus=num_gpus)

# 打印并运行训练命令
print("Running command:", command)
os.system(command)

# 记录训练结束时间
end_time = time.time()

# 计算并打印训练时长
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Training time for fold {}: {:0>2}:{:0>2}:{:05.2f}".format(fold, int(hours), int(minutes), seconds))

'''
# 如果需要运行所有 5 个 fold，并分别记录每个 fold 的训练时长
for fold in range(5):
    start_time = time.time()  # 每个 fold 开始时记录开始时间
    command = command_template.format(task_id=task_id, model_type=model_type, fold=fold, num_gpus=num_gpus)
    print("Running command for fold {}: {}".format(fold, command))
    os.system(command)
    
    # 记录结束时间并计算时长
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time for fold {}: {:0>2}:{:0>2}:{:05.2f}".format(fold, int(hours), int(minutes), seconds))
'''
