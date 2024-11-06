# Time for the dataset!!
## Download dataset: From BraTs or (http://medicaldecathlon.com/)
I download the 'Task01_BrainTumor' dataset form (http://medicaldecathlon.com/) which conclude the dataset from BraTs. It is more stable and fast for installation. Firstly install the tar. to your remote disk,then unzip with the command in the terminal:
```
tar -xvf xxxx(path)/Task01_BrainTumour.tar -C /dxxxx/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data
```

(optional)take a look at the data (by keru_visualization.py):
![a9ce66a18e56401f75b5060095f4262](https://github.com/user-attachments/assets/c4955f97-7a93-4f45-a63a-7bdf42b01271)

## Data discription
The dataset you download should have the following construction. ImagesTr and LabelsTr' name are related in similar name. 
![image](https://github.com/user-attachments/assets/3bfc071f-7f4e-4c66-960d-6487bad936a2)
The raw dataset looks like:
```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```
but for nnUNet, it can only recognize the data in strict format like:

```
nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_001_0002.nii.gz
│   ├── BRATS_001_0003.nii.gz

```
This dataset hat four input channels: FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003). That's why we need **data conversion**.

## Data conversion
1. Datasets must be located in the nnUNet_raw folder (which you either define when installing nnU-Net or export/set every time you intend to run nnU-Net commands!). Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name (which you can freely choose)
2. The dataset from (http://medicaldecathlon.com/) is in the format like 'Task01_BrainTumour' suitable for the first version nnUnet. However, it has upgrade to 'nnUNetv2' in the github. So we need to convert the old_nnUNet_dataset to nnUNetv2_Dataset.
(Important) The command given in the github of nnUNet exsit some mistake. It provides the first version of date_conversion_command as follow. However it doesn't work due to the huge difference of format between dataset. From the blog we found that, 'nnUNetv2_convert_MSD_dataset' is the correct one for v1_dataset conversion.
The correct ouput is: you can fins a new dataset named 'Dataset900_BrainTumour' is created under the path of nnUNet_raw.
```
# wrong
nnUNetv2_convert_old_nnUNet_dataset /nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour Dataset001_BrainTumour

# correct the data convertion 
nnUNetv2_convert_MSD_dataset -i 
/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task01_BrainTumour -overwrite_id 900
```

## Creating your own subset
Since the whole data set is huge for 484 cases(Task01_BrainTumour), we need to create subset to decrease the training period. I tried the initial configuration of the nnUNet to train for 1 days and raise 'CUDA OUT OF MEMORIES'. It is time-consuming and GPU-consuming! Create your own subset! I fixed the problem by creating subset and change the num_of_epoch and batch_size(which will have a further disscusion in Training.md)

**Warning!!** Before the plan and preprocess of the data, you should create your own sub set first. If you directly create subset after the preprocess of data, it will raise error since the dataset.json will be unmatched to your subset.
My choice for the training subset and testing subset followed the data ratio in the initial dataset (training: testing = 1:0.64) by creating_subset.py

Subset size: Training cases: 100 (For ImageTR and LabelsTr)
             Testing cases: 64 (for ImagesTS)

## Data plan and preprocess
Last Step! You need: converted subset. A simple command is needed in this step:
```
nnUNetv2_plan_and_preprocess -d DATA_FOLD_CODE --verify_dataset_integrity

```
verify_dataset_integrity ensures your data is identifiable.


