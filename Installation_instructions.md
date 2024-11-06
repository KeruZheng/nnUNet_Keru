# nnUNet_BraTs_Keru
reproduce nnUNet on BraTs

## System requirements
### Operating System & Hardware requirements
Linux (Ubuntu 18.04) RTX 3090 5 GPU

## Installation instructions
follow the intructions in https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md

1. Install PyTorch before 'pip install nnunetv2'
2. For use as integrative framework (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
   (Important) Installing PyTorch is time consuming. Try to find mirror website as: (https://lyd.im/archives/accelerating-pytorch-installation-with-domestic-mirrors). You can find the version related to your cuda. use 
```
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
```
4.   

