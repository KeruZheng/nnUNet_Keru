#可视化
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-interactive)
import matplotlib.pyplot as plt


# 载入 NIfTI 文件
nii_file = '/data1/keru/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset900_BrainTumour/imagesTr/BRATS_001_0001.nii.gz'  # 替换为你的文件路径
img = nib.load(nii_file)

# 获取数据
data = img.get_fdata()

# 获取其中一个切片的数据（例如获取第50个切片）
slice_idx = 50
slice_data = data[:, :, slice_idx]

# 显示该切片
plt.imshow(slice_data.T, cmap='gray', origin='lower')  # 使用转置来调整坐标轴方向
plt.colorbar()
plt.title(f'Slice {slice_idx}')
plt.show()

plt.imshow(slice_data.T, cmap='gray', origin='lower')
plt.colorbar()
plt.title(f'Slice {slice_idx}')
plt.savefig('output_image1.png')  # Save the plot to a file

