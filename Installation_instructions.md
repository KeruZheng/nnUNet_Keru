## System requirements
### Operating System & Hardware requirements
Problem: The colab only have instoration of 15G which is not enough for the training. By the way, To buy a colab pro version with larger memory needs visa or AliPay(HK). 
Solve：Thanks to the Gpu from Prof.Xiaoguang Han 's Lab. Final operating: Linux (Ubuntu 18.04) RTX 3090, A6000 5 GPU

## Installation instructions
follow the intructions in https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md

1. Create environment with python version >= 3.9
```
   conda create -n nnUNet_env python=3.10 
```
3. Install PyTorch before 'pip install nnunetv2'(Important, otherwise it will lead to package conflict) Installing PyTorch is time consuming. Try to find mirror website as: (https://lyd.im/archives/accelerating-pytorch-installation-with-domestic-mirrors). You can find the version related to your cuda.
4. For use as integrative framework (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
    use 
```
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
```

3. **nnU-Net needs to know where you intend to save raw data**, preprocessed data and trained models. For this you need to set a few environment variables.
   what I did is:
```  
cd /content/drive/MyDrive/nnUNet
mkdir nnUNetFrame

cd nnUNetFrame
mkdir DATASET

cd DATASET
mkdir nnUNet_raw
mkdir nnUNet_preprocessed
mkdir nnUNet_trained_models

cd nnUNet_raw
mkdir nnUNet_raw_data
mkdir nnUNet_cropped_data

cd nnUNet_raw_data
mkdir Task01_BrainTumour
```
to help bulid the file instruction as 
```
nnUNet_raw/Dataset001_NAME1
├── dataset.json
├── imagesTr
│   ├── ...
├── imagesTs
│   ├── ...
└── labelsTr
    ├── ...
nnUNet_raw/Dataset002_NAME2
├── dataset.json
├── imagesTr
│   ├── ...
├── imagesTs
│   ├── ...
└── labelsTr
    ├── ...
```
4. Time for the dataset~ I download the Task01_BrainTumor dataset form (http://medicaldecathlon.com/) which conclude the dataset from BraTs. It is more stable and fast for installation. Firstly install the tar. to your remote disk, then unzip with the command in the terminal:
```
tar -xvf xxxx(path)/Task01_BrainTumour.tar -C /dxxxx/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data
```
the data look like this(by visualization.py):
![a9ce66a18e56401f75b5060095f4262](https://github.com/user-attachments/assets/c4955f97-7a93-4f45-a63a-7bdf42b01271)



