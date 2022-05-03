import os
from typing import Optional 
import SimpleITK as sitk
import h5py


def h5_multi_load(save_path):

    with h5py.File(save_path, 'r') as hf:
        contents = {}
        for k in hf.keys():
            contents[k] = hf.get(k)[()]

    return contents


def h5_multi_save(save_path, **kwargs):

    with h5py.File(save_path, 'w') as hf:
        for k, v in kwargs.items():
            hf.create_dataset(k, data=v)

    return


def load_image(filepath: str, harmonize_to: Optional[str] = 'LPS', return_spacing=False):

    image = sitk.ReadImage(filepath)

    image_harmonized = sitk.DICOMOrient(image, harmonize_to)

    if return_spacing: 
        return sitk.GetArrayFromImage(image_harmonized), image_harmonized.GetSpacing()

    return image_harmonized