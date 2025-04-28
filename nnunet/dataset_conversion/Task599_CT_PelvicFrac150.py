import os
from os.path import join

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import (
    join, listdir, maybe_mkdir_p, save_json)
from natsort import natsorted

cur_root = os.path.abspath(join(os.path.dirname(__file__), '..', '..'))
data_root = join(cur_root, 'dataset', 'nnUNet_raw')

# 数据集的标签内容为:
# 0 = 背景,
# 1-10 = 骶骨碎片,
# 11-20 =左髋骨碎片,
# 21-30 = 右髋骨碎片
sa_labels = set([i for i in range(1, 11)])
li_labels = set([i for i in range(11, 21)])
ri_labels = set([i for i in range(21, 31)])


def convert(in_file, out_file):
    img = sitk.ReadImage(in_file)
    sitk.WriteImage(img, out_file)


def convert_with_mask(in_img_file, in_label_file, out_file, roi_labels):
    scale_img = sitk.ReadImage(in_img_file)
    label_img = sitk.ReadImage(in_label_file)
    mask_arr = sitk.GetArrayFromImage(label_img).copy()
    for i in np.unique(mask_arr):
        if i not in roi_labels:
            mask_arr[mask_arr == i] = 0

    frac_grayscale = get_mask_image(scale_img, mask_arr, replacevalue=0)
    sitk.WriteImage(frac_grayscale, out_file)


def convert_separate_fracture(in_file, out_file, sorted_roi_labels: list):
    print(f"{os.path.basename(out_file)[:-7]}\t{sorted_roi_labels}")
    img = sitk.ReadImage(in_file)
    arr = sitk.GetArrayFromImage(img)
    num_labels = 3 if len(sorted_roi_labels) > 3 else len(sorted_roi_labels)

    new_arr = np.zeros_like(arr)
    for i in range(num_labels):
        new_arr[arr == sorted_roi_labels[i]] = i + 1
    new_img = sitk.GetImageFromArray(new_arr)

    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    sitk.WriteImage(new_img, out_file)


def convert_with_combine_label(in_file, out_file):
    img = sitk.ReadImage(in_file)
    arr = sitk.GetArrayFromImage(img)
    labels = set(np.unique(arr).tolist())
    sal = list(sa_labels & labels)
    lil = list(li_labels & labels)
    ril = list(ri_labels & labels)
    print(f"{os.path.basename(out_file)[:-7]} "
          f"{str(sal):<20} {str(lil):<20} {str(ril):<20}")

    new_arr = np.zeros_like(arr)
    for i in labels:
        if i in sal:
            label = 1
        elif i in lil:
            label = 2
        elif i in ril:
            label = 3
        else:
            label = 0
        new_arr[arr == i] = label
    new_img = sitk.GetImageFromArray(new_arr)

    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetDirection(img.GetDirection())
    sitk.WriteImage(new_img, out_file)


def get_mask_image(sitk_src, array_mask, replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk


def reorient_to_RAS(img_fname: str, output_fname: str = None):
    img = nib.load(img_fname)
    canonical_img = nib.as_closest_canonical(img)
    if output_fname is None:
        output_fname = img_fname
    nib.save(canonical_img, output_fname)


if __name__ == '__main__':
    """
    export nnUNet_raw_data_base="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_raw"
    export nnUNet_preprocessed="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_preprocessed"
    export RESULTS_FOLDER="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_trained_models" 
    """
    downloaded_data_dir = join(data_root, 'PENGWIN_CT')
    img_root = join(downloaded_data_dir, 'PENGWIN_CT_train_images')
    seg_root = join(downloaded_data_dir, 'PENGWIN_CT_train_labels')

    task_name = "Task599_CT_PelvicFrac150"
    target_base = join(data_root, 'nnUNet_raw_data', task_name)
    target_imagesTr = join(target_base, 'imagesTr')
    target_labelsTr = join(target_base, 'labelsTr')

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

    for file in natsorted(listdir(seg_root)):
        seg_path = join(seg_root, file)
        img_path = join(img_root, file)

        convert(
            img_path,
            join(target_imagesTr, file[:3] + "_0000.nii.gz"),
        )
        convert_with_combine_label(
            seg_path,
            join(target_labelsTr, file[:3] + ".nii.gz"),
        )

    for fname in listdir(target_imagesTr):
        reorient_to_RAS(join(target_imagesTr, fname))
    for fname in listdir(target_labelsTr):
        reorient_to_RAS(join(target_labelsTr, fname))

    train_patient_names = natsorted(listdir(target_labelsTr))

    json_dict = {}
    json_dict['name'] = "CT fracture all 54 res"
    json_dict['description'] = "CT fracture all 54 res"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "CT fracture all 54 res"
    json_dict['licence'] = ""
    json_dict['release'] = "2022.11.03"
    json_dict['modality'] = {"0": "CT"}
    json_dict['labels'] = {
        "0": "background",
        "1": "sacrum",
        "2": "left ilium",
        "3": "right ilium"
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{
        'image': "./imagesTr/%s" % i,
        "label": "./labelsTr/%s" % i
    } for i in train_patient_names]
    # json_dict['test'] = ["./imagesTs/%s" % i for i in test_patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))
