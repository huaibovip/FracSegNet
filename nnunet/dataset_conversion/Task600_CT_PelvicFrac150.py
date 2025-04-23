import hashlib
import os
import zipfile
from os.path import join
from warnings import warn

import numpy as np
import requests
import SimpleITK as sitk
import tqdm
from batchgenerators.utilities.file_and_folder_operations import (
    join, listdir, maybe_mkdir_p, save_json)
from natsort import natsorted
from tqdm import tqdm

cur_root = os.path.abspath(join(os.path.dirname(__file__), '..', '..'))
data_root = join(cur_root, 'dataset', 'nnUNet_raw')

URLS = {
    "CT": [
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_labels.zip",  # labels
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part1.zip",  # inputs part 1
        "https://zenodo.org/records/10927452/files/PENGWIN_CT_train_images_part2.zip",  # inputs part 2
    ],
    "X-Ray": ["https://zenodo.org/records/10913196/files/train.zip"]
}

CHECKSUMS = {
    "CT": [
        "c4d3857e02d3ee5d0df6c8c918dd3cf5a7c9419135f1ec089b78215f37c6665c",  # labels
        "e2e9f99798960607ffced1fbdeee75a626c41bf859eaf4125029a38fac6b7609",  # inputs part 1
        "19f3cdc5edd1daf9324c70f8ba683eed054f6ed8f2b1cc59dbd80724f8f0bbb2",  # inputs part 2
    ],
    "X-Ray":
    ["48d107979eb929a3c61da4e75566306a066408954cf132907bda570f2a7de725"]
}

TARGET_DIRS = {
    "CT": [
        "PENGWIN_CT_train_labels", "PENGWIN_CT_train_images",
        "PENGWIN_CT_train_images"
    ],
    "X-Ray": ["X-Ray"]
}

MODALITIES = ["CT", "X-Ray"]


def unzip(zip_path, dst, remove=True):
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(dst)
    if remove:
        os.remove(zip_path)


def _check_checksum(path, checksum):

    def get_checksum(filename):
        with open(filename, "rb") as f:
            file_ = f.read()
            checksum = hashlib.sha256(file_).hexdigest()
        return checksum

    if checksum is not None:
        this_checksum = get_checksum(path)
        if this_checksum != checksum:
            raise RuntimeError(
                "The checksum of the download does not match the expected checksum."
                f"Expected: {checksum}, got: {this_checksum}")
        print("Download successful and checksums agree.")
    else:
        warn(
            "The file was downloaded, but no checksum was provided, so the file may be corrupted."
        )


def download(path, url, checksum=None, verify=True):
    resp = requests.get(url, stream=True, allow_redirects=True, verify=verify)
    total = int(resp.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(
            desc='Downloading',
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    _check_checksum(path, checksum)


def download_pengwin_data(path: str, modality: list = "CT") -> str:
    """Download the PENGWIN dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        modality: The choice of modality for inputs.
        download: Whether to download the data if it is not present.

    Returns:
        Filepath where the data is downlaoded.
    """
    os.makedirs(path, exist_ok=True)
    for url, checksum, dst_dir in zip(URLS[modality], CHECKSUMS[modality],
                                      TARGET_DIRS[modality]):
        zip_path = join(path, os.path.split(url)[-1])
        download(path=zip_path, url=url, checksum=checksum)
        unzip(zip_path=zip_path, dst=join(path, dst_dir))


# 数据集的标签内容为:
# 0 = 背景,
# 1-10 = 骶骨碎片,
# 11-20 =左髋骨碎片,
# 21-30 = 右髋骨碎片
sa_labels = set([i for i in range(1, 11)])
li_labels = set([i for i in range(11, 21)])
ri_labels = set([i for i in range(21, 31)])
NUM_LABELS = 2


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
    num_labels = NUM_LABELS if len(sorted_roi_labels) > NUM_LABELS else len(
        sorted_roi_labels)

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


def check_data(root):
    files = listdir(root)
    for file in files:
        img = sitk.ReadImage(join(root, file))
        arr = sitk.GetArrayFromImage(img)
        num_labels = len(np.unique(arr))
        assert num_labels <= 3, file


if __name__ == '__main__':
    """
    export nnUNet_raw_data_base="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_raw"
    export nnUNet_preprocessed="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_preprocessed"
    export RESULTS_FOLDER="$HOME/projects/vscode/FracSegNet/dataset/nnUNet_trained_models" 
    """
    downloaded_data_dir = join(data_root, 'PENGWIN_CT1')
    img_root = join(downloaded_data_dir, 'PENGWIN_CT_train_images')
    seg_root = join(downloaded_data_dir, 'PENGWIN_CT_train_labels')
    download_pengwin_data(downloaded_data_dir)

    task_name = "Task600_CT_PelvicFrac150"
    target_base = join(data_root, 'nnUNet_raw_data', task_name)
    target_imagesTr = join(target_base, 'imagesTr')
    target_labelsTr = join(target_base, 'labelsTr')

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

    for file in natsorted(listdir(seg_root)):
        seg_path = join(seg_root, file)
        img_path = join(img_root, file)
        seg = sitk.ReadImage(seg_path)
        seg_data = sitk.GetArrayFromImage(seg)

        labels = set(np.unique(seg_data).tolist())
        sal = list(sa_labels & labels)
        lil = list(li_labels & labels)
        ril = list(ri_labels & labels)
        sal.sort()
        lil.sort()
        ril.sort()

        if 1 in sal:
            frac = '_frac' if len(sal) > 1 else '_none'
            convert_with_mask(
                img_path,
                seg_path,
                join(target_imagesTr, file[:3] + f"_SA{frac}_0000.nii.gz"),
                sal,
            )
            convert_separate_fracture(
                seg_path,
                join(target_labelsTr, file[:3] + f"_SA{frac}.nii.gz"),
                sal,
            )
        else:
            raise ValueError()

        if 11 in lil:
            frac = '_frac' if len(lil) > 1 else '_none'
            convert_with_mask(
                img_path,
                seg_path,
                join(target_imagesTr, file[:3] + f"_LI{frac}_0000.nii.gz"),
                lil,
            )
            convert_separate_fracture(
                seg_path,
                join(target_labelsTr, file[:3] + f"_LI{frac}.nii.gz"),
                lil,
            )
        else:
            raise ValueError()

        if 21 in ril:
            frac = '_frac' if len(ril) > 1 else '_none'
            convert_with_mask(
                img_path,
                seg_path,
                join(target_imagesTr, file[:3] + f"_RI{frac}_0000.nii.gz"),
                ril,
            )
            convert_separate_fracture(
                seg_path,
                join(target_labelsTr, file[:3] + f"_RI{frac}.nii.gz"),
                ril,
            )
        else:
            raise ValueError()

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
        "1": "main fracture segment",
        "2": "segment 2",
        # "3": "segment 3"
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

    # check_data(target_labelsTr)
