import sys

sys.path.append('3rdparty/batchgenerators')

import numpy as np
import pickle as pkl
from batchgenerators.utilities.file_and_folder_operations import *

# init
# name
# class
# plans

path = 'checkpoints/CTPelvic1K_Models/fold_0/3d_cascade_fullres/model_best.model.pkl'
plans = load_pickle(path)
for k, v in plans.items():
    if k == 'plans':
        print(k)
        for k1, v1 in v.items():
            if isinstance(v1, list | dict):
                print(f"  {k1}: {len(v1)}")
            else:
                print(f"  {k1}: {v1}")
    else:
        print(f"{k}: {v}")

# print(plans['plans_per_stage'][0]['batch_size'])
# print(plans['plans_per_stage'][0]['patch_size'])
# plans['plans_per_stage'][0]['batch_size'] = 1
# plans['plans_per_stage'][0]['patch_size'] = np.array((28, 192, 192))
# save_pickle(plans, join('dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/nnUNetPlansv2.1_plans_3D.pkl'))

# src_path = 'dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/dataset_properties.pkl'
# plans_pkl = load_pickle(src_path)
# print(plans_pkl.keys())
# for name, value in plans_pkl.items():
#     print(name, value)

# path = 'dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/splits_final.pkl'
# files = load_pickle(path)
# print(len(files))
# for file in files:
#     print(len(file['train']), len(file['val']))

# plan_path = 'dataset/nnUNet_preprocessed/Task600_CTPelvicFrac150/nnUNetPlansv2.1_plans_3D.pkl'
# plan_files = load_pickle(plan_path)
# print(len(plan_files))
# for name, plan_file in plan_files.items():
#     if isinstance(plan_file, list|dict) and len(plan_file) > 5:
#         print(name)
#     else:
#         print(name, plan_file)
#     # print(len(plan_file['train']), len(plan_file['val']))
