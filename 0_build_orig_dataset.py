"""
1) Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import argparse
import h5py
from tqdm import tqdm
from tools.dataset_utils import generate_data

def main(args):

    obj_list = [os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if os.path.isdir(os.path.join(args.input_path, x))]

    with h5py.File(args.output_path, 'w') as h5f:

        for local_path in tqdm(obj_list):

            object_folder = local_path.split('/')[-1]
            sub_group = h5f.create_group(object_folder)
            generate_data(local_path, args, sub_group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render RGB and SMPL UV images from aligned mesh files into HDF5.')
    parser.add_argument("-i", "--input-path", default='data/THuman/new_thuman', type=str, help="Aligned THuman2.0 folder")
    parser.add_argument("-o", "--output-path", default='data/dataset.h5', type=str, help="Output path")
    parser.add_argument("--size", default=1024, type=int, help="Image size")

    parser.add_argument("--nviews", default=36, type=int, help="Number of views to render")
    parser.add_argument("--camera-mode", default='orth', type=str, help="Camera mode: orth | persp")
    parser.add_argument("--camera-sampling", default='uniform', type=str, help="Camera sampling: uniform | random")

    main(parser.parse_args())
