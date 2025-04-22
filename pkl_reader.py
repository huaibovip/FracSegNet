import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

path = 'dataset/nnUNet_preprocessed/Task600_CT_PelvicFrac150/nnUNetPlansv2.1_plans_3D.pkl'
plans = load_pickle(path)
for k,v in plans.items():
    print(k,v)
print(plans['plans_per_stage'][0]['batch_size'])
print(plans['plans_per_stage'][0]['patch_size'])
# plans['plans_per_stage'][0]['batch_size'] = 1
# plans['plans_per_stage'][0]['patch_size'] = np.array((28, 192, 192))
# save_pickle(plans, join('dataset/nnUNet_preprocessed/Task600_CT_PelvicFrac150/nnUNetPlansv2.1_plans_3D.pkl'))




# src_path = 'dataset/nnUNet_preprocessed/Task600_CT_PelvicFrac150/dataset_properties.pkl'
# plans_pkl = load_pickle(src_path)
# print(plans_pkl.keys())
# for name, value in plans_pkl.items():
#     print(name, value)
