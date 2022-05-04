import os 
import numpy as np


def trim_dict(dict_fn: str, out_fn: str) -> None:

    im_label_dict = np.load(dict_fn)

    ims = im_label_dict['img_id']
    labels = im_label_dict['label']

    base_ims = list(set([(i.split('_')[0], l) for i,l in zip(ims, labels)]))
    idx_mapping = {i: [] for i, _ in base_ims}

    keep_ims = []
    keep_labels = []

    for im in ims: 
        # if the img_id is stored as "volume_slice", 
        # get the original image, and the slice idx
        i = im.split('_')[0]
        idx_mapping[i].append(im)
    
    for i, l in base_ims: 
        slices = idx_mapping[i]
        keep_slice = slices[len(slices) // 2]
        keep_ims.append(keep_slice)
        keep_labels.append(l)

    # save the output
    np.savez(out_fn, img_id=keep_ims, label=keep_labels)

    print(keep_ims[0], keep_labels[0], np.sum(keep_labels), len(keep_ims), len(keep_labels))

    return 

def main():

    metadata_train = './metadata/metadata_train.npz'
    metadata_valid = './metadata/metadata_val.npz'
    metadata_test = './metadata/metadata_test.npz'

    trim_dict(metadata_train, './metadata/metadata_train_trimmed.npz')
    trim_dict(metadata_valid, './metadata/metadata_valid_trimmed.npz')
    trim_dict(metadata_test, './metadata/metadata_test_trimmed.npz')

    return 


if __name__ == "__main__":
    main()