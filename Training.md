# Training

## ensure correct file path
```
export nnUNet_raw="/xxx/nnUNet/nnUNetFrame/DATASET/nnUNet_raw"
export nnUNet_preprocessed="/xxxx/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed"
export nnUNet_results="/xxx/nnUNet/nnUNetFrame/DATASET/nnUNet_results"
export nnUNet_compile=f # help speed up and avoid runninbg out of time
```

## adapt the epoch
The epoch and batch size will directly influence your training time. 
In machine learning, the relationship between the number of iterations and epochs is typically described as follows:

An epoch refers to one complete pass of the entire training dataset through the model. Iterations refer to the number of steps (or batch processes) required to complete one epoch.

Specifically, if your training dataset contains 1000 samples, and you choose a batch size of 100, then:

'The number of iterations per epoch = Total number of samples in the training dataset ÷ Batch size'

So, using this example:

'The number of iterations per epoch = 1000 ÷ 100 = 10 iterations'

This means that each epoch, the model will require 10 iterations to go through all the training data. During each iteration, the model processes 100 training samples and updates the model's weights once. As I ran for the first time for 1000(initial epoch setting), we 
### 3D full resolution U-Net
![cf52596568a6527b5885244c844455b](https://github.com/user-attachments/assets/930ee74f-191c-4f5d-89d8-77fcebd89bf3)
### 2D U-Net
![78bbee2c38fab907e3a3c7f017b92dd](https://github.com/user-attachments/assets/37481e54-63b4-4608-959e-59723ae6bebe)

The graph shows that between the 50th and 100th epochs, both the training loss (blue line) and validation loss (red line) exhibit a trend towards stabilization. Notably, the validation loss, after initial significant fluctuations, gradually stabilizes at a lower level, indicating an improvement in the model's adaptation and generalization to new data during this phase. The Dice coefficient (dotted green line represents the moving average) stabilizes after about 100 epochs, further verifying that the model's performance gradually stabilizes with ongoing training and is approaching its optimal performance.

Given the analysis of the graph and the behavior of the loss and validation metrics, it is indeed sensible to adjust the definition value of epochs to 100 for this training scenario. This adjustment allows the model to stabilize and generalize better before further evaluations or deployment.

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
it really confused me and cost me a lot of time. However, I checked for the ' nvidia-smi ' and found that the mission is run on the right cuda device. 

