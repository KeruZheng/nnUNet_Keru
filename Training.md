# Training

## ensure correct file path
```
export nnUNet_raw="/xxx/nnUNet/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/xxxx/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed"
export nnUNet_results="/xxx/nnUNet/nnUNetFrame/DATASET/nnUNet_results"
export nnUNet_compile=f # help speed up and avoid runninbg out of time
```

## run the model
1. you can train in the jupternotebook to help check training progress in time. The code is provided in Train.ipynb
2. I use the command in TERMINAL finally since it is more convinient. Due to the limit of time and service, I only run 2D U-Net and 3D full resolution U-Net
```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
my version
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATA_FOLD_CODE 3d_fullres(MISSION: ) 4 -device cuda -num_gpus 1
```

### 2D U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```
### 3D full resolution U-Net
For FOLD in [0, 1, 2, 3, 4], run:
```
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
```

### Automatically determine the best configuration
```
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```


ATTENTION!!
you will find that even defined for the cuda service, once you run it will print:
![image](https://github.com/user-attachments/assets/cce18561-198e-40d0-9417-7ec35a83bbf8)
it really confused me and cost me a lot of time. However, I checked for the 'nvidia-smi' and found that the mission is run on the right cuda device. 

