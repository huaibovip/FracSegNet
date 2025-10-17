import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import *


img_path = 'dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/nnUNetData_plans_v2.1_stage0/002_RI_frac.npy'
prop_path = 'dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/dataset_properties.pkl'

img = np.load(img_path)
prop = load_pickle(prop_path)
data = sitk.GetImageFromArray(img[0])
mask = sitk.GetImageFromArray(img[1].astype('int8'))
dmap = sitk.GetImageFromArray(img[2])
sitk.WriteImage(data, 'data.nii.gz')
sitk.WriteImage(mask, 'mask.nii.gz')
sitk.WriteImage(dmap, 'dmap.nii.gz')

print(np.unique(img[1]))

plt.subplot(1, 3, 1)
plt.imshow(img[0, 100], cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow(img[1, 100], cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(img[2, 100], cmap='gray')

plt.show()
