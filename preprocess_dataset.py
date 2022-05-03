"""preprocess_dataset.py


"""
import os 
import argparse
from typing import Dict

from imageops.preprocess import extract_coronal_slices, extract_axial_slices


def parse_command_line() -> Dict:

    parser = argparse.ArgumentParser()
    parser.add_argument('--coronals', action="store_true")
    parser.add_argument('--axials', action='store_true')
    parser.add_argument('--data_dir', action="store", type=str)
    parser.add_argument('--outdir', action="store", type=str)

    return vars(parser.parse_args())


def main():

    args = parse_command_line()
    if args['coronals']: 
        print('extracting coronals for train')
        metadata_csv = 'metadata/train.csv'
        extract_coronal_slices(args['data_dir'], metadata_csv, args['outdir'], 'train')

        print('extracting coronals for val')
        metadata_csv = 'metadata/val.csv'
        extract_coronal_slices(args['data_dir'], metadata_csv, args['outdir'], 'val')

        print('extracting coronals for test')
        metadata_csv = 'metadata/test.csv'
        extract_coronal_slices(args['data_dir'], metadata_csv, args['outdir'], 'test')

    elif args['axials']: 
        print('extracting coronals for train')
        metadata_csv = 'metadata/train.csv'
        extract_axial_slices(args['data_dir'], metadata_csv, args['outdir'], 'train')

        print('extracting coronals for val')
        metadata_csv = 'metadata/val.csv'
        extract_axial_slices(args['data_dir'], metadata_csv, args['outdir'], 'val')

        print('extracting coronals for test')
        metadata_csv = 'metadata/test.csv'
        extract_axial_slices(args['data_dir'], metadata_csv, args['outdir'], 'test')

    return 


if __name__ == "__main__":

    main()