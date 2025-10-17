import torch

# path = 'checkpoints/nnUNet/3d_cascade_fullres/Task600_CT_PelvicFrac150/nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/all/model_final_checkpoint.model'
path = 'checkpoints/nnUNet/3d_fullres/Task600_CT_PelvicFrac150/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model'
ckpt = torch.load(path)['state_dict']
for ckpt_key, val in ckpt.items():
    if 'seg_outputs' in ckpt_key:
        print(ckpt_key, val.shape)
