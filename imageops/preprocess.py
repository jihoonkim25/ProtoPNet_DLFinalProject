"""preprocess.py
"""
import argparse
import os 
import time
from typing import List, Tuple, Union 

import numpy as np
import pandas as pd 
import SimpleITK as sitk
from utils.file_io import h5_multi_save, load_image


def clip_min_max(vol, upper=99, lower=1):

    temp_max = np.percentile(vol, upper)
    temp_min = np.percentile(vol, lower)
    vol[vol < temp_min] = temp_min
    vol[vol > temp_max] = temp_max

    return vol


def rescale_01(vol, per_slice=True, window_min=None, window_max=None) -> np.ndarray:

    if per_slice:
        if window_min is None:
            window_min = np.min(vol, axis=(1, 2)).reshape(-1, 1, 1)
        vol = np.subtract(vol, window_min)
        if window_max is None:
            window_max = np.max(vol, axis=(1, 2)).reshape(-1, 1, 1)

        vol = np.divide(vol, window_max)
    else:
        vol -= np.min(vol)
        vol = np.divide(vol, np.max(vol))

    vol[vol < 0] = 0.0
    vol[vol > 1] = 1.0

    return vol


def resample_image(img: sitk.Image, spacing: Union[Tuple, List, np.ndarray]): 

    # initialize teh resample image filter
    resample = sitk.ResampleImageFilter()

    # set interpolator settings
    resample.SetInterpolator = sitk.sitkLinear
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(spacing.squeeze())

    # compute the new size
    orig_size = np.array(img.GetSize(), dtype=np.int)
    orig_spacing = img.GetSpacing()
    new_size = orig_size*(orig_spacing / spacing)
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size.squeeze()]
    resample.SetSize(new_size)

    # perform resampling 
    new_img = resample.Execute(img)

    return new_img


def make_minip(img: sitk.Image, spacing: Union[Tuple, List, np.ndarray], dimension: int) -> sitk.Image:
    
    proj = sitk.MinimumProjection(img, dimension)
    
    return proj


def make_minips(img_dir: str, metadata_path: str, outdir: str) -> None: 

    return 


def extract_coronal_slices(img_dir: str, metadata_path: str, outdir: str, partition: str) -> None: 

    metadata_csv = pd.read_csv(metadata_path, header=None, names=['img_id', 'label'])
    trimmed_dir = os.path.abspath(img_dir)
    new_im_ids = []
    new_labels = []

    start = time.perf_counter()
    for idx in range(len(metadata_csv)): 
        cur_im_id = metadata_csv['img_id'][idx]
        cur_label = metadata_csv['label'][idx]
        cur_fn = '%d.mha' % (cur_im_id)
        cur_path = os.path.join(trimmed_dir, cur_fn)

        print('working on', cur_path)

        # load image 
        im = load_image(cur_path)

        # resample teh image to uniform spacing
        new_spacing = np.tile(im.GetSpacing()[0], (1, 3))
        new_im = resample_image(im, new_spacing)

        # get new image as np array
        im_arr = sitk.GetArrayFromImage(new_im)

        # save the middle 20 coronal slices
        coronals = im_arr[:,245:266 :]

        # save each of the coronal slices
        for s in range(coronals.shape[1]): 
            new_id = "%d_%d" % (cur_im_id, s)
            new_path = os.path.join(outdir, "%s.hdf5" % (new_id))

            # save the current coronal slice
            h5_multi_save(
                new_path, 
                img = np.squeeze(coronals[:,s,:].astype(np.single)),
                label = cur_label, 
            )

            new_im_ids.append(new_id)
            new_labels.append(cur_label)

        print('time elapsed: %d m, %.3f s' % ((time.perf_counter() - start)//60, (time.perf_counter() - start) % 60))
        print('----')
    
    np.savez(os.path.join(outdir, 'metadata_%s.npz' % (partition)), img_id=new_im_ids, label=new_labels)

    return 


def extract_axial_slices(img_dir: str, metadata_path: str, outdir: str, partition: str) -> None: 

    metadata_csv = pd.read_csv(metadata_path, header=None, names=['img_id', 'label'])
    trimmed_dir = os.path.abspath(img_dir)
    new_im_ids = []
    new_labels = []

    start = time.perf_counter()
    for idx in range(len(metadata_csv)): 
        cur_im_id = metadata_csv['img_id'][idx]
        cur_label = metadata_csv['label'][idx]
        cur_fn = '%d.mha' % (cur_im_id)
        cur_path = os.path.join(trimmed_dir, cur_fn)

        print('working on', cur_path)

        # load image 
        im = load_image(cur_path)
        im_arr = sitk.GetArrayFromImage(im)

        # save the middle coronal slices
        lower_lim = np.min([int(im_arr.shape[0] * 2. / 5), 110])
        upper_lim = np.max([int(im_arr.shape[0] * 4. / 5), im_arr.shape[0] - 100])

        print(lower_lim, upper_lim)
        
        for ax_slice in range(lower_lim, upper_lim): 

            cur_slice = im_arr[ax_slice, :,:]

            new_id = "%d_%d" % (cur_im_id, ax_slice)
            new_path = os.path.join(outdir, "%s.hdf5" % (new_id))

            # save the current coronal slice
            h5_multi_save(
                new_path, 
                img = np.squeeze(cur_slice.astype(np.single)),
                label = cur_label.astype(np.int), 
            )

            new_im_ids.append(new_id)
            new_labels.append(cur_label)

        print('time elapsed: %d m, %.3f s' % ((time.perf_counter() - start)//60, (time.perf_counter() - start) % 60))
        print('----')
    
    np.savez(os.path.join(outdir, 'metadata_%s.npz' % (partition)), img_id=new_im_ids, label=new_labels)

    return 
