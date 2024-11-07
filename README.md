<h1 align="center">nnUNet_Keru</h1> 

# Evaluating nnU-Net on BraTS Dataset

## Overview
This project focuses on evaluating the performance of the nnU-Net framework applied to the Brain Tumor Segmentation (BraTS) dataset. The nnU-Net is a robust, self-configuring framework that has been designed for medical image segmentation. The Brain Tumor Segmentation (BraTS) challenge provides a comprehensive dataset comprising multi-institutional pre-operative MRI scans and focuses on the segmentation of intracranial brain tumors. This repository documents the application of nnU-Net to this challenging dataset, providing insights into the versatility and efficiency of the framework.

## Repository Contents
1. **Installation Instructions (`Installation_instructions.md`)**  
   - Hardware requirements
   - Software dependencies
   - Environment setup

2. **Data Preparation (`Data_preparation.md`)**  
   - Subset selection
   - Data conversion
   - Data processing steps to transform the BraTS dataset into a format compatible with the nnU-Net framework.

3. **Training (`Training.md` and `Training.py`)**  
   - Detailed steps on how to train the nnU-Net framework using the prepared BraTS data.

4. **Additional Scripts**  
   - `creating_subset.py`: Script to help select and prepare subsets of data.
   - `data_visualization.py`: Utility for visualizing data transformations and results.

## Special Features of This Repository
This repository is unique in that it not only provides a comprehensive guide to implementing nnU-Net on the BraTS dataset but also includes detailed records of the challenges encountered during model replication. Each problem faced is accompanied by the specific solutions that were implemented, serving as a practical resource for others facing similar issues in their projects.

Acknowledgments
We extend our sincere thanks to the developers of the nnU-Net framework and particularly to the team at the German Cancer Research Center (DKFZ) for their open-source contributions. This project could not have been accomplished without the foundational work available at MIC-DKFZ/nnUNet on GitHub. Their pioneering efforts in creating robust and flexible frameworks for medical image segmentation have been invaluable to our research.
Dataset: https://www.med.upenn.edu/sbia/brats2017.html
Blog may help:
https://blog.csdn.net/m0_45521766/article/details/131539779?spm=1001.2014.3001.5502
https://blog.csdn.net/qq_45794210/article/details/120699443
