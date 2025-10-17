import argparse
import os
from os.path import basename, dirname, join

import pandas as pd
import SimpleITK as sitk

from nnunet.basicFunc import *


def saveDiffFrac(fileName, labelName):
    save_root = join(dirname(fileName), 'fraction')
    name = basename(fileName).split('_', 1)
    os.makedirs(save_root, exist_ok=True)

    # load image data
    ct_origin_img = sitk.ReadImage(fileName)
    ct_label_img = sitk.ReadImage(labelName)

    # ======================extract the single fracture and Rescale Intensity======================
    # label = 1: Sacrum / 2: Left Hip / 3:Right Hip
    # =============================================================================================
    frac_sacrum_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 1)
    frac_LeftIliac_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 2)
    frac_RightIliac_img, _ = extractSingleFrac(ct_origin_img, ct_label_img, 3)

    sitk.WriteImage(frac_sacrum_img, join(save_root, name[0] + '_SA_' + name[1]))
    sitk.WriteImage(frac_LeftIliac_img, join(save_root, name[0] + '_LI_' + name[1]))
    sitk.WriteImage(frac_RightIliac_img, join(save_root, name[0] + '_RI_' + name[1]))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process CT images to extract and save different fractures.'
    )
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Path to the input CT image file (e.g., ct_name).')
    parser.add_argument(
        '-m',
        '--mask',
        required=True,
        help='Path to the label/mask image file (e.g., mask_name).')

    args = parser.parse_args()

    saveDiffFrac(args.input, args.mask)
